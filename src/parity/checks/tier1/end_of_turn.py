"""t1.end_of_turn: does every assistant turn end with the model's stop token, trained?

Bug class: a framework masks the token that closes an assistant turn (the
GLM-5.3-Flash bug Plumbline found), drops it, or swaps it for a different
token, so the model never learns to stop and runs on at inference.

Reference: the template the framework declares it uses. For each assistant
turn, the closing token is the turn's last stop token in the reference (the
tokenizer's EOS plus the release's generation_config eos ids), at position
`token_end - 1` of its span.

Pass rule: exact, per turn. At each closing position the framework must have
the same token, with loss, and with its label equal to the token. A turn whose
closing position is where the framework's tokens first diverge from the
reference fails as missing or replaced. If the tokens diverge earlier inside
a turn, that turn and every later one can't be lined up: the result is
inconclusive unless an earlier turn already failed (see t1.template_parity).

Planted fault: mask_closing_tokens: take the loss off every supervised stop
token. Applies to every fixture with an assistant turn.
"""

from __future__ import annotations

from typing import Any

from parity.adapters.base import LABEL_IGNORE, Declaration, Dropped, Encoded
from parity.checks.common import (
    Tier1Check,
    declared_reference,
    is_contradicted,
    reference_info,
    template_status,
    unaligned_verdict,
)
from parity.engine.evidence import decode_each, first_divergence, token_evidence
from parity.engine.plants import Plant, encoded_plant, with_labels
from parity.engine.verdict import Verdict
from parity.fixtures import Fixture
from parity.release import LoadedRelease

FAILING = ("missing_or_replaced", "not_trained", "wrong_label")


class EndOfTurn(Tier1Check):
    id = "t1.end_of_turn"

    def run(self, adapter: Any, release: LoadedRelease, fixture: Fixture) -> Verdict:
        tokenizer = release.tokenizer
        declaration = adapter.declare(self.base_cfg)
        ref = declared_reference(release, fixture, declaration)
        info = reference_info(release, ref)
        out = adapter.encode(fixture, self.base_cfg)
        if isinstance(out, Dropped):
            return self.verdict(
                adapter,
                fixture,
                "fail",
                kind="bug",
                reference=info,
                declared=declaration,
                evidence={"dropped": out.reason, "note": "the conversation was dropped with no length limit"},
            )
        status = template_status(release, fixture, declaration, out.input_ids)
        if is_contradicted(status):
            return unaligned_verdict(
                self, adapter, release, fixture, declaration, out.input_ids, ref.input_ids, reference=info
            )

        def verdict(result: str, **fields: Any) -> Verdict:
            verified = {"chat_template": status}
            return self.verdict(adapter, fixture, result, reference=info, declared=declaration, verified=verified, **fields)

        d = first_divergence(ref.input_ids, out.input_ids)
        turns = []
        for span in ref.spans:
            c = span.token_end - 1
            closing = ref.input_ids[c]
            turn: dict[str, Any] = {
                "message_index": span.message_index,
                "position": c,
                "expected_token": decode_each(tokenizer, [closing])[0],
            }
            if d is not None and d < c:
                turn["status"] = "unchecked"
            elif d == c:
                turn["status"] = "missing_or_replaced"
                turn["got_token"] = decode_each(tokenizer, [out.input_ids[c]])[0] if c < len(out.input_ids) else None
            elif out.labels[c] == LABEL_IGNORE:
                turn["status"] = "not_trained"
            elif out.labels[c] != closing:
                turn["status"] = "wrong_label"
                turn["label"] = out.labels[c]
            else:
                turn["status"] = "trained"
            turns.append(turn)

        evidence: dict[str, Any] = {"turns": turns}
        if d is not None:
            evidence["divergence"] = token_evidence(tokenizer, ref.input_ids, out.input_ids)
        if any(t["status"] in FAILING for t in turns):
            return verdict("fail", kind="bug", evidence=evidence)
        unchecked = [t["message_index"] for t in turns if t["status"] == "unchecked"]
        if unchecked:
            reason = (
                f"the framework's tokens differ from its declared template's render before the closing token of "
                f"message(s) {unchecked}, so they can't be lined up; see t1.template_parity"
            )
            return verdict("inconclusive", reason=reason, evidence=evidence)
        return verdict("pass", evidence=evidence)

    def plants(self, release: LoadedRelease, fixture: Fixture, declaration: Declaration) -> list[Plant]:
        if not fixture.assistant_indices:
            return []
        stops = release.stop_token_ids

        def mask_closing_tokens(row: Encoded) -> Encoded:
            return with_labels(
                row,
                [
                    LABEL_IGNORE if label != LABEL_IGNORE and token in stops else label
                    for token, label in zip(row.input_ids, row.labels)
                ],
            )

        return [
            encoded_plant(
                "mask_closing_tokens",
                "take the loss off every supervised stop token",
                mask_closing_tokens,
                origin="real-world",
                source="Plumbline (arXiv 2611.13393): GLM-5.3-Flash's turn-closing token was masked out of training",
            )
        ]
