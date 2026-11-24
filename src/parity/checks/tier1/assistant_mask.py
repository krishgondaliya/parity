"""t1.assistant_mask: under an assistant-only objective, is loss on exactly the assistant's own tokens?

Bug class: a framework trains on the prompt (system, user or tool text, or a
role header), skips part of an assistant turn, trains on nothing at all, or
hands over labels that are shifted by one, so the model learns the wrong
thing or nothing.

Reference: the template the framework declares it uses, rendered through
transformers. Assistant spans come from the prefix diff, cross-checked against
the template's generation markers when it has them. Every reference token is
in one zone (docs/DECISIONS.md, "Mask zones"): must-train (the assistant's
content through its stop token), neutral (the separator after the stop
token), or must-not-train (everything else).

Pass rule: exact, per token. Fail if any must-train token has no loss, any
must-not-train token has loss, or any supervised label is not the token
itself (labels are aligned with input ids; adapters for frameworks that
pre-shift labels shift them back). Neutral tokens never fail; the verdict
records whether the framework trains them, and the framework's native label
convention. If the framework's tokens don't match its declared template's
render, the result is inconclusive (see t1.template_parity).
A fail is a known difference only when a zone-signature entry in the registry
explains every violation (parity.checks.known_differences).

Planted faults (all synthetic). These apply to every fixture with a prompt before an assistant turn:
- unmask_prompt: give loss to everything before the first supervised token, user turn included.
- zero_mask: no loss anywhere.
- shift_labels_left: labels shifted left by one, as if pre-shifted labels were handed over as aligned.
- late_start: every supervised span starts one token late.
- wrong_label_values: every supervised label points at the next token id; which tokens get loss is
  unchanged, so only the label-value rule can catch it.
And these where the fixture allows:
- train_only_last_turn: loss only on the last assistant turn (fixtures with two or more).
- span_starts_at_header: each assistant span also trains its role header, e.g. <|im_start|>assistant\n
  (when every span has a header in the declared template's render).
"""

from __future__ import annotations

from typing import Any

from parity.adapters.base import LABEL_IGNORE, Dropped, Encoded
from parity.checks.common import (
    CONSISTENT,
    Declare,
    Tier1Check,
    declared_reference,
    mask_report,
    reference_info,
    unaligned_verdict,
    zone_difference,
)
from parity.checks.known_differences import describe
from parity.engine.plants import Plant, encoded_plant, supervised_runs, with_labels
from parity.engine.verdict import Verdict
from parity.fixtures import Fixture
from parity.references.template import AssistantSpan, TemplateReference, Zone
from parity.release import LoadedRelease


class AssistantMask(Tier1Check):
    id = "t1.assistant_mask"

    def run(self, adapter: Any, release: LoadedRelease, fixture: Fixture) -> Verdict:
        declaration = adapter.declare(self.base_cfg)
        ref = declared_reference(release, fixture, declaration, self.mode(release))
        info = reference_info(release, ref)
        conventions: dict[str, Any] = {"label_convention": declaration.label_convention}
        out = adapter.encode(fixture, self.base_cfg)
        if isinstance(out, Dropped):
            return self.verdict(
                adapter,
                fixture,
                "fail",
                kind="bug",
                reference=info,
                conventions=conventions,
                declared=declaration,
                evidence={"dropped": out.reason, "note": "the conversation was dropped with no length limit"},
            )
        if out.input_ids != ref.input_ids:
            return unaligned_verdict(
                self,
                adapter,
                release,
                fixture,
                declaration,
                out.input_ids,
                ref.input_ids,
                reference=info,
                conventions=conventions,
            )

        # The label convention is recorded, not verified: a pre-shifted pattern
        # handed over as aligned is judged as a bug here (see shift_labels_left).
        report = mask_report(ref.zones, out)
        conventions.update(report.conventions())
        evidence = report.evidence(release.tokenizer, out, ref.zones)
        evidence["must_train_tokens"] = sum(z == Zone.MUST_TRAIN for z in ref.zones)
        evidence["supervised_tokens"] = sum(label != LABEL_IGNORE for label in out.labels)

        def verdict(result: str, **fields: Any) -> Verdict:
            return self.verdict(
                adapter,
                fixture,
                result,
                reference=info,
                conventions=conventions,
                declared=declaration,
                verified={"chat_template": CONSISTENT},
                evidence=evidence,
                **fields,
            )

        if not report.violated:
            return verdict("pass")

        def explains(known: TemplateReference) -> tuple[bool, dict[str, Any]]:
            rest = mask_report(known.zones, out)
            return not rest.violated, rest.evidence(release.tokenizer, out, known.zones)

        entry, leftovers = zone_difference(self, adapter, release, fixture, declaration, ref, explains)
        if entry is not None:
            evidence["known_difference"] = describe(entry)
            return verdict("fail", kind="difference", known_difference=entry.id)
        if leftovers:
            evidence["known_difference_residual"] = leftovers
        return verdict("fail", kind="bug")

    def plants(self, release: LoadedRelease, fixture: Fixture, declare: Declare) -> list[Plant]:
        if not fixture.assistant_indices or fixture.assistant_indices[0] == 0:
            return []
        faults = [
            ("unmask_prompt", "give loss to everything before the first supervised token", _unmask_prompt),
            ("zero_mask", "no loss anywhere", _zero_mask),
            ("shift_labels_left", "labels shifted left by one", _shift_labels_left),
            ("late_start", "every supervised span starts one token late", _late_start),
            (
                "wrong_label_values",
                "supervised labels point at a different token; which tokens get loss is unchanged",
                _wrong_label_values(len(release.tokenizer)),
            ),
        ]
        if len(fixture.assistant_indices) >= 2:
            faults.append(("train_only_last_turn", "loss only on the last assistant turn", _train_only_last_turn))
        spans = declared_reference(release, fixture, declare(self.base_cfg), self.mode(release)).spans
        if spans and all(s.token_message_start is not None and s.token_message_start < s.token_start for s in spans):
            faults.append(
                ("span_starts_at_header", "each assistant span also trains its role header", _starts_at_header(spans))
            )
        return [encoded_plant(name, about, mutate, origin="synthetic") for name, about, mutate in faults]


def _unmask_prompt(row: Encoded) -> Encoded:
    first = next((i for i, label in enumerate(row.labels) if label != LABEL_IGNORE), len(row.labels))
    return with_labels(row, row.input_ids[:first] + row.labels[first:])


def _zero_mask(row: Encoded) -> Encoded:
    return with_labels(row, [LABEL_IGNORE] * len(row.labels))


def _shift_labels_left(row: Encoded) -> Encoded:
    return with_labels(row, row.labels[1:] + (LABEL_IGNORE,))


def _late_start(row: Encoded) -> Encoded:
    labels = list(row.labels)
    for start, _ in supervised_runs(row):
        labels[start] = LABEL_IGNORE
    return with_labels(row, labels)


def _wrong_label_values(vocab_size: int):
    def mutate(row: Encoded) -> Encoded:
        return with_labels(
            row,
            [label if label == LABEL_IGNORE else (label + 1) % vocab_size for label in row.labels],
        )

    return mutate


def _train_only_last_turn(row: Encoded) -> Encoded:
    runs = supervised_runs(row)
    labels = list(row.labels)
    for start, end in runs[:-1]:
        labels[start:end] = [LABEL_IGNORE] * (end - start)
    return with_labels(row, labels)


def _starts_at_header(spans: tuple[AssistantSpan, ...]):
    def mutate(row: Encoded) -> Encoded:
        labels = list(row.labels)
        for span in spans:
            for i in range(span.token_message_start, min(span.token_start, len(labels))):
                labels[i] = row.input_ids[i]
        return with_labels(row, labels)

    return mutate
