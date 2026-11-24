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

from collections.abc import Sequence
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from typing import Any, ClassVar

from transformers import PreTrainedTokenizerBase

from parity.adapters.base import LABEL_IGNORE, Declaration, Encoded, TrainCfg
from parity.engine.evidence import decode_each, token_evidence
from parity.engine.plants import Plant
from parity.engine.verdict import Verdict, make_verdict
from parity.fixtures import Conversation, Fixture
from parity.references.template import (
    TemplateReference,
    Zone,
    model_chat_template,
    render_reference,
    template_sha256,
    template_token_ids,
)
from parity.release import LoadedRelease

NO_ALIGNMENT = (
    "the framework's tokens differ from its own declared template's render, so this check can't line "
    "them up with the reference; see t1.template_parity"
)


class Tier1Check:
    """Base for Tier 1 checks. Subclasses set `id` and implement `run` and `plants`.

    run(adapter, release, fixture) -> Verdict
        The comparison for one fixture, with no self-test. The engine
        (`parity.engine.runner.evaluate`) adds the planted faults.
    plants(release, fixture, declaration) -> list[Plant]
        The planted faults that apply to this fixture. Each `plant.apply(adapter)`
        is the contract's `plant(adapter) -> Adapter`.
    not_applicable(fixture) -> reason | None
        Decided from the fixture alone.
    """

    id: ClassVar[str]
    tier: ClassVar[int] = 1
    base_cfg: ClassVar[TrainCfg] = TrainCfg(objective="assistant_only")

    def not_applicable(self, fixture: Fixture) -> str | None:
        return None

    def run(self, adapter: Any, release: LoadedRelease, fixture: Fixture) -> Verdict:
        raise NotImplementedError

    def plants(self, release: LoadedRelease, fixture: Fixture, declaration: Declaration) -> list[Plant]:
        raise NotImplementedError

    def verdict(
        self,
        adapter: Any,
        fixture: Fixture,
        result: str,
        *,
        declared: Declaration | None = None,
        verified: dict[str, str] | None = None,
        **fields: Any,
    ) -> Verdict:
        if declared is not None:
            fields["declaration"] = declared.record(verified)
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
        f"adapter error: the declared {field_name} is contradicted by what the framework did "
        f"({status.removeprefix(CONTRADICTED)}). Check the adapter's declare() before treating this as a "
        "framework finding (rule 4)."
    )


def template_status(release: LoadedRelease, conv: Conversation, declaration: Declaration, ids: Sequence[int]) -> str:
    """Do the framework's tokens confirm the declared chat template?"""
    ids = tuple(ids)
    if ids == token_ids_on(release, conv, declaration.chat_template):
        return CONSISTENT
    official = official_template(release)
    if declaration.chat_template != official and ids == token_ids_on(release, conv, official):
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
    status = template_status(release, fixture, declaration, ids)
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


# References, cached per (release, template, conversation content). They are pure functions of those.

_REFERENCES: dict[tuple[str, str, str, str, str], TemplateReference] = {}
_TOKEN_IDS: dict[tuple[str, str, str], tuple[int, ...]] = {}


def official_template(release: LoadedRelease) -> str:
    return model_chat_template(release.tokenizer)


def reference_on(release: LoadedRelease, conv: Conversation, template: str, source: str) -> TemplateReference:
    key = (release.release.spec, template_sha256(template), conv.content_key, conv.id, source)
    if key not in _REFERENCES:
        _REFERENCES[key] = render_reference(release, conv, template, source)
    return _REFERENCES[key]


def official_reference(release: LoadedRelease, conv: Conversation) -> TemplateReference:
    return reference_on(release, conv, official_template(release), "model")


def declared_reference(release: LoadedRelease, conv: Conversation, declaration: Declaration) -> TemplateReference:
    template = declaration.chat_template
    source = "model" if template == official_template(release) else declaration.template_source
    return reference_on(release, conv, template, source)


def token_ids_on(release: LoadedRelease, conv: Conversation, template: str) -> tuple[int, ...]:
    """Just the token ids of a template render, for conversations that need no spans (e.g. prefixes)."""
    key = (release.release.spec, template_sha256(template), conv.content_key)
    if key not in _TOKEN_IDS:
        _TOKEN_IDS[key] = tuple(template_token_ids(release.tokenizer, conv, template))
    return _TOKEN_IDS[key]


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
        **library_versions(),
    }


def special_ids(tokenizer: PreTrainedTokenizerBase) -> frozenset[int]:
    """Every token id the tokenizer marks as special (not merely added, like Qwen3's <think>)."""
    ids = set(tokenizer.all_special_ids)
    ids.update(i for i, token in tokenizer.added_tokens_decoder.items() if token.special)
    return frozenset(ids)


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
