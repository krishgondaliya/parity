"""Pieces every Tier 1 check shares: the base class, cached references, and the zone-based mask report.

Which reference a check compares against:

- `t1.template_parity` compares against the model's own (official) template.
  A divergence is a `difference` only if a known-differences entry matches it
  exactly.
- The other checks compare against the template the framework declares it
  uses (`Declaration.chat_template`), because their question is about masks,
  stop tokens, padding, truncation or prefixes, not about which template was
  chosen. If the framework's tokens don't match its own declared template's
  render, they say so and return inconclusive (template parity owns that).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from typing import Any, ClassVar

from transformers import PreTrainedTokenizerBase

from parity.adapters.base import LABEL_IGNORE, Declaration, Encoded, TrainCfg
from parity.engine.evidence import decode_each, token_evidence
from parity.engine.plants import Plant
from parity.engine.verdict import Verdict, make_verdict
from parity.fixtures import Conversation, Fixture
from parity.checks.known_differences import KnownDifference, describe, entries_in_scope, known_zones
from parity.references.properties import prefill_property_id
from parity.serving import resolve_mode, template_kwargs
from parity.references.template import (
    TemplateReference,
    TemplateReferenceError,
    Zone,
    model_chat_template,
    render_reference,
    template_sha256,
    template_token_ids,
)
from parity.release import LoadedRelease

Declare = Callable[[TrainCfg], Declaration]

NO_ALIGNMENT = (
    "the framework's tokens differ from its own declared template's render, so this check can't line "
    "them up with the reference; see t1.template_parity"
)


class Tier1Check:
    """Base for Tier 1 checks. Subclasses set `id` and implement `run` and `plants`.

    run(adapter, release, fixture) -> Verdict
        The comparison for one fixture, with no self-test. The engine
        (`parity.engine.runner.evaluate`) adds the planted faults.
    plants(release, fixture, declare) -> list[Plant]
        The planted faults that apply to this fixture. `declare` is the
        adapter's declare(): a check asks it about the exact config each
        planted fault will be encoded under. Each `plant.apply(adapter)` is
        the contract's `plant(adapter) -> Adapter`.
    not_applicable(fixture) -> reason | None
        Decided from the fixture alone.
    """

    id: ClassVar[str]
    tier: ClassVar[int] = 1

    def __init__(self, serving_mode: str | None = None):
        # None: the model's documented default (parity.serving). The mode picks the reference's template
        # arguments; frameworks encode the same rows in every mode.
        self.serving_mode = serving_mode

    @property
    def base_cfg(self) -> TrainCfg:
        return TrainCfg(objective="assistant_only", serving_mode=getattr(self, "serving_mode", None))

    def mode(self, release: LoadedRelease) -> str:
        """The serving mode this check judges in, for this model."""
        return resolve_mode(release, getattr(self, "serving_mode", None))

    def not_applicable(self, fixture: Fixture) -> str | None:
        return None

    def run(self, adapter: Any, release: LoadedRelease, fixture: Fixture) -> Verdict:
        raise NotImplementedError

    def plants(self, release: LoadedRelease, fixture: Fixture, declare: Declare) -> list[Plant]:
        raise NotImplementedError

    def reference_properties(self, release: LoadedRelease, fixture: Fixture, declaration: Declaration) -> list[str]:
        """Model-level template properties this check's reference relies on for this fixture.

        Checks that judge against spans of the declared template depend on how those spans were found;
        a span anchored past a missing pre-fill depends on the "pre-fill missing from history" property.
        """
        try:
            ref = declared_reference(release, fixture, declaration, self.mode(release))
        except TemplateReferenceError:
            return []
        if any(span.prefill_missing for span in ref.spans):
            return [prefill_property_id(release, ref.template_sha256, self.mode(release))]
        return []

    def verdict(
        self,
        adapter: Any,
        fixture: Fixture,
        result: str,
        *,
        declared: Declaration | None = None,
        verified: dict[str, str] | None = None,
        cfg: TrainCfg | None = None,
        other_calls: list[dict[str, Any]] | None = None,
        flags: list[str] | None = None,
        **fields: Any,
    ) -> Verdict:
        """Build a verdict that records the declaration for `cfg` (default: this check's base config).

        `other_calls` holds the declaration records of any other encode calls the check made
        under a different config. A contradicted declaration anywhere flags the verdict as a
        declared/actual mismatch for /triage.
        """
        flags = list(flags or [])
        if declared is not None:
            record = declared.record(verified, cfg if cfg is not None else self.base_cfg)
            if other_calls:
                record["other_calls"] = other_calls
            fields["declaration"] = record
            statuses = list(record["verified"].values())
            statuses += [s for call in other_calls or [] for s in call["verified"].values()]
            if any(is_contradicted(s) for s in statuses):
                flags.append("declared_actual_mismatch")
                fields["evidence"] = {**fields.get("evidence", {}), "triage": "declared/actual mismatch"}
        fields["flags"] = sorted(set(flags))
        fields.setdefault("serving_mode", self.mode(adapter.release))
        return make_verdict(self.id, adapter, fixture.id, result, **fields)


# Verifying declarations against behavior. A declaration is contradicted only when
# the behavior exactly matches a different value of the same field; that is an
# adapter error until proven otherwise (rule 4), so the verdict is inconclusive.

CONSISTENT = "consistent"
CONTRADICTED = "contradicted: "
MISMATCH = "mismatch: "


def is_contradicted(status: str) -> bool:
    return status.startswith(CONTRADICTED)


def adapter_error(field_name: str, status: str) -> str:
    return (
        f"declared/actual mismatch: the declared {field_name} is contradicted by what the framework did "
        f"({status.removeprefix(CONTRADICTED)}). Route to /triage: either the adapter's declare() misreads "
        "the framework's config (our bug, rule 4), or the framework ignores its own config (a real bug class)."
    )


def template_status(
    release: LoadedRelease, conv: Conversation, declaration: Declaration, ids: Sequence[int], mode: str | None = None
) -> str:
    """Do the framework's tokens confirm the declared chat template (rendered in `mode`)?"""
    ids = tuple(ids)
    if ids == token_ids_on(release, conv, declaration.chat_template, mode):
        return CONSISTENT
    official = official_template(release)
    if declaration.chat_template != official and ids == token_ids_on(release, conv, official, mode):
        return CONTRADICTED + "the tokens are exactly the model's own template's render, not the declared template's"
    return MISMATCH + "the tokens match neither the declared template's render nor the model's own"


def unaligned_verdict(
    check: Tier1Check,
    adapter: Any,
    release: LoadedRelease,
    fixture: Fixture,
    declaration: Declaration,
    ids: Sequence[int],
    expected: Sequence[int],
    **fields: Any,
) -> Verdict:
    """Inconclusive: the framework's tokens don't line up with its declared template's render."""
    status = template_status(release, fixture, declaration, ids, check.mode(release))
    reason = adapter_error("chat_template", status) if is_contradicted(status) else NO_ALIGNMENT
    return check.verdict(
        adapter,
        fixture,
        "inconclusive",
        reason=reason,
        declared=declaration,
        verified={"chat_template": status},
        evidence=token_evidence(release.tokenizer, expected, ids),
        **fields,
    )


# References, cached per (release, template, conversation content, serving mode). They are pure functions of those.
# `mode=None` means the model's documented default (parity.serving).

_REFERENCES: dict[tuple[str, str, str, str, str, str], TemplateReference] = {}
_TOKEN_IDS: dict[tuple[str, str, str, str], tuple[int, ...]] = {}


def official_template(release: LoadedRelease) -> str:
    return model_chat_template(release.tokenizer)


def reference_on(
    release: LoadedRelease, conv: Conversation, template: str, source: str, mode: str | None = None
) -> TemplateReference:
    resolved = resolve_mode(release, mode)
    key = (release.release.spec, template_sha256(template), conv.content_key, conv.id, source, resolved)
    if key not in _REFERENCES:
        kwargs = template_kwargs(release, resolved)
        _REFERENCES[key] = render_reference(release, conv, template, source, kwargs)
    return _REFERENCES[key]


def official_reference(release: LoadedRelease, conv: Conversation, mode: str | None = None) -> TemplateReference:
    return reference_on(release, conv, official_template(release), "model", mode)


def declared_reference(
    release: LoadedRelease, conv: Conversation, declaration: Declaration, mode: str | None = None
) -> TemplateReference:
    template = declaration.chat_template
    source = "model" if template == official_template(release) else declaration.template_source
    return reference_on(release, conv, template, source, mode)


def token_ids_on(release: LoadedRelease, conv: Conversation, template: str, mode: str | None = None) -> tuple[int, ...]:
    """Just the token ids of a template render, for conversations that need no spans (e.g. prefixes)."""
    resolved = resolve_mode(release, mode)
    key = (release.release.spec, template_sha256(template), conv.content_key, resolved)
    if key not in _TOKEN_IDS:
        kwargs = template_kwargs(release, resolved)
        _TOKEN_IDS[key] = tuple(template_token_ids(release.tokenizer, conv, template, **kwargs))
    return _TOKEN_IDS[key]


def zone_difference(
    check: Tier1Check,
    adapter: Any,
    release: LoadedRelease,
    fixture: Fixture,
    declaration: Declaration,
    ref: TemplateReference,
    explains: Callable[[TemplateReference], tuple[bool, dict[str, Any]]],
) -> tuple[KnownDifference | None, list[dict[str, Any]]]:
    """The zone-signature known difference that explains the whole divergence, if one does.

    `explains(ref)` is the check's own rule, judged against a reference whose zones the entry's documented
    behavior gives. Returns (entry, []) when one leaves nothing over; otherwise (None, what each in-scope
    entry left unexplained), so triage can link the rest to its own root cause.
    """
    leftovers = []
    for entry in entries_in_scope(
        check_id=check.id,
        framework=adapter.name,
        version=adapter.version,
        official_template=official_template(release),
        declared_template=declaration.chat_template,
        mode=check.mode(release),
    ):
        ok, evidence = explains(known_zones(entry, release, fixture, ref))
        if ok:
            return entry, []
        leftovers.append({**describe(entry), "unexplained": evidence})
    return None, leftovers


def library_versions() -> dict[str, str]:
    out = {}
    for name in ("transformers", "tokenizers", "trl"):
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = "none"
    return out


def reference_info(release: LoadedRelease, ref: TemplateReference) -> dict[str, Any]:
    """What the verdict was compared against. The declaration is recorded separately (Verdict.declaration)."""
    return {
        "template_source": ref.template_source,
        "template_sha256": ref.template_sha256,
        "official_template_sha256": template_sha256(official_template(release)),
        "marker_check": ref.marker_check,
        "template_kwargs": dict(ref.template_kwargs),
        **library_versions(),
    }


def special_ids(tokenizer: PreTrainedTokenizerBase) -> frozenset[int]:
    """Every token id the tokenizer marks as special (not merely added, like Qwen3's <think>)."""
    ids = set(tokenizer.all_special_ids)
    ids.update(i for i, token in tokenizer.added_tokens_decoder.items() if token.special)
    return frozenset(ids)


def replacement_special(release: LoadedRelease, old: int) -> int:
    """Another special token to swap in: another stop token if there is one, else the pad token, else any special."""
    others = sorted(release.stop_token_ids - {old})
    if others:
        return others[0]
    pad = release.tokenizer.pad_token_id
    if pad is not None and pad != old:
        return pad
    return min(special_ids(release.tokenizer) - {old})


def closing_token_id(ref: TemplateReference) -> int:
    """The stop token that closes the first assistant turn in the reference."""
    if not ref.spans:
        raise ValueError("reference has no assistant span")
    return ref.input_ids[ref.spans[0].token_end - 1]


# The zone rule shared by t1.assistant_mask and t1.truncation (docs/DECISIONS.md, "Mask zones").


@dataclass
class MaskReport:
    must_train_unsupervised: list[int] = field(default_factory=list)
    must_not_train_supervised: list[int] = field(default_factory=list)
    label_mismatch: list[int] = field(default_factory=list)  # supervised, but the label isn't the token itself
    neutral_trained: int = 0
    neutral_untrained: int = 0

    @property
    def violated(self) -> bool:
        return bool(self.must_train_unsupervised or self.must_not_train_supervised or self.label_mismatch)

    @property
    def neutral_convention(self) -> str:
        if self.neutral_trained and self.neutral_untrained:
            return "mixed"
        if self.neutral_trained:
            return "trained"
        if self.neutral_untrained:
            return "not_trained"
        return "none"

    def conventions(self) -> dict[str, Any]:
        return {
            "neutral": self.neutral_convention,
            "neutral_trained": self.neutral_trained,
            "neutral_untrained": self.neutral_untrained,
        }

    def evidence(self, tokenizer: PreTrainedTokenizerBase, row: Encoded, zones: Sequence[Zone]) -> dict[str, Any]:
        def show(positions: list[int]) -> list[dict[str, Any]]:
            return [
                {
                    "index": i,
                    "token": decode_each(tokenizer, [row.input_ids[i]])[0],
                    "zone": zones[i].name.lower(),
                    "label": row.labels[i],
                }
                for i in positions[:8]
            ]

        return {
            "counts": {
                "must_train_unsupervised": len(self.must_train_unsupervised),
                "must_not_train_supervised": len(self.must_not_train_supervised),
                "label_mismatch": len(self.label_mismatch),
            },
            "must_train_unsupervised": show(self.must_train_unsupervised),
            "must_not_train_supervised": show(self.must_not_train_supervised),
            "label_mismatch": show(self.label_mismatch),
        }


def mask_report(zones: Sequence[Zone], row: Encoded) -> MaskReport:
    """Compare a row's supervision with the reference zones. The row must already line up with them."""
    if len(zones) != len(row.input_ids):
        raise ValueError("row and zones differ in length")
    report = MaskReport()
    for i, zone in enumerate(zones):
        supervised = row.labels[i] != LABEL_IGNORE
        if supervised and row.labels[i] != row.input_ids[i]:
            report.label_mismatch.append(i)
        if zone == Zone.MUST_TRAIN and not supervised:
            report.must_train_unsupervised.append(i)
        elif zone == Zone.MUST_NOT_TRAIN and supervised:
            report.must_not_train_supervised.append(i)
        elif zone == Zone.NEUTRAL:
            if supervised:
                report.neutral_trained += 1
            else:
                report.neutral_untrained += 1
    return report
