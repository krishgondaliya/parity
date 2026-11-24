"""t1.truncation: at the length limit, does the framework cut the way it says it does, and keep the loss right?

Bug class: a framework truncates differently from its own declared mode, cuts
one token too many (an off-by-one at max length that loses the final stop
token), keeps a cut answer but loses its loss (silently training on nothing),
or re-appends a stop token after cutting through an answer (teaching the
model to stop mid-answer).

Reference: the template the framework declares it uses, cut the way the
framework declares it truncates, read from its own config or defaults:
- right: keep the first max_len tokens (and their zones);
- left: keep the last max_len tokens;
- drop: skip any conversation longer than max_len.
The mode is recorded in the verdict.

Limits: `long_answer` uses its own max_len (128 tokens: the prompt fits, the
answer doesn't). Every other fixture runs at max_len equal to its exact
reference length, the boundary where nothing may be cut and an off-by-one
shows.

Pass rule: exact. First, with no limit, the framework's tokens must equal the
declared template's render; otherwise the result is inconclusive (see
t1.template_parity). Then, at the limit:
- drop mode: a conversation longer than the limit must be dropped, and one that fits must be kept;
- right or left: the row must equal the cut reference token for token, and
  its loss must follow the cut reference's zones (no must-train token without
  loss, no must-not-train token with loss, labels equal to their tokens;
  neutral tokens free).
A framework that doesn't match its declared mode fails, unless its output is
exactly what a different mode gives: then the declaration is contradicted,
an adapter error until shown otherwise, and the result is inconclusive.

Planted faults (decided from the fixture, the reference length and the declared mode):
- off_by_one: right/left: cut to max_len - 1 whenever a row reaches max_len (every fixture).
  drop: also drop a row of exactly max_len (fixtures that fit exactly).
- lose_answer_labels: when the cut falls inside an assistant turn, keep the
  cut row but take the loss off everything (in drop mode: keep a right-cut row
  with no loss instead of dropping it).
- reappend_stop: right mode, when a cut happens: after cutting, put the turn's
  stop token back as the last token, with loss.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from parity.adapters.base import TRUNCATION_MODES, LABEL_IGNORE, Declaration, Dropped, Encoded, TrainCfg
from parity.checks.common import (
    CONSISTENT,
    CONTRADICTED,
    MISMATCH,
    Tier1Check,
    adapter_error,
    closing_token_id,
    declared_reference,
    mask_report,
    reference_info,
    unaligned_verdict,
)
from parity.engine.evidence import token_evidence
from parity.engine.plants import Plant, PlantedAdapter
from parity.engine.verdict import Verdict
from parity.fixtures import Conversation, Fixture
from parity.references.template import TemplateReference, Zone
from parity.release import LoadedRelease


class Truncation(Tier1Check):
    id = "t1.truncation"

    def run(self, adapter: Any, release: LoadedRelease, fixture: Fixture) -> Verdict:
        tokenizer = release.tokenizer
        declaration = adapter.declare(self.base_cfg)
        mode = declaration.truncation_mode
        ref = declared_reference(release, fixture, declaration)
        info = reference_info(release, ref)
        max_len = limit_for(fixture, ref)
        conventions = {
            "truncation_mode": mode,
            "max_len": max_len,
            "reference_len": len(ref.input_ids),
            "cut": len(ref.input_ids) > max_len,
            "limit": "fixture" if fixture.max_len is not None else "boundary (exact reference length)",
        }

        full = adapter.encode(fixture, self.base_cfg)
        if isinstance(full, Dropped):
            return self.verdict(
                adapter,
                fixture,
                "inconclusive",
                reason=f"dropped even with no length limit ({full.reason})",
                reference=info,
                conventions=conventions,
                declared=declaration,
            )
        if full.input_ids != ref.input_ids:
            return unaligned_verdict(
                self,
                adapter,
                release,
                fixture,
                declaration,
                full.input_ids,
                ref.input_ids,
                reference=info,
                conventions=conventions,
            )

        def verdict(result: str, mode_status: str, **fields: Any) -> Verdict:
            verified = {"chat_template": CONSISTENT, "truncation_mode": mode_status}
            return self.verdict(
                adapter,
                fixture,
                result,
                reference=info,
                conventions=conventions,
                declared=declaration,
                verified=verified,
                **fields,
            )

        out = adapter.encode(fixture, replace(self.base_cfg, max_len=max_len))
        ok, evidence = judge(tokenizer, ref, max_len, mode, out)
        if ok:
            status = CONSISTENT if conventions["cut"] else "unverified: nothing is cut at this limit"
            return verdict("pass", status, evidence=evidence)

        # Not what the declared mode gives. If it is exactly what another mode gives,
        # the declaration is contradicted: an adapter error until shown otherwise.
        others = [m for m in TRUNCATION_MODES if m != mode and judge(tokenizer, ref, max_len, m, out)[0]]
        if others:
            status = CONTRADICTED + f"the framework's row is exactly what {others[0]!r} truncation gives"
            return verdict("inconclusive", status, reason=adapter_error("truncation_mode", status), evidence=evidence)
        return verdict("fail", MISMATCH + "the row matches no truncation mode", kind="bug", evidence=evidence)

    def plants(self, release: LoadedRelease, fixture: Fixture, declaration: Declaration) -> list[Plant]:
        mode = declaration.truncation_mode
        ref = declared_reference(release, fixture, declaration)
        max_len = limit_for(fixture, ref)
        cut = len(ref.input_ids) > max_len
        out = []

        if mode in ("right", "left"):
            out.append(_plant("off_by_one", f"cut to max_len - 1 ({mode})", _off_by_one(mode)))
        elif not cut:
            out.append(_plant("off_by_one", "also drop a row of exactly max_len", _drop_at_limit))

        if cut and _cut_is_inside_an_assistant_turn(ref, max_len, mode):
            out.append(
                _plant("lose_answer_labels", "keep the cut row but take the loss off everything", _lose_labels(ref))
            )
        if cut and mode == "right":
            reappend = _reappend(closing_token_id(ref))
            out.append(_plant("reappend_stop", "put the stop token back as the last token, with loss", reappend))
        return out


def judge(
    tokenizer: Any, ref: TemplateReference, max_len: int, mode: str, out: Encoded | Dropped
) -> tuple[bool, dict[str, Any]]:
    """Is `out` exactly what `mode` truncation gives at `max_len`? Returns (ok, evidence)."""
    expected = cut_reference(ref, max_len, mode)
    if expected is None:
        if isinstance(out, Dropped):
            return True, {"dropped": out.reason}
        rule = f"{mode}: the conversation should be dropped, but a row was kept"
        return False, {"rule": rule, "got_len": len(out.input_ids)}
    expected_ids, expected_zones = expected
    if isinstance(out, Dropped):
        rule = f"{mode}: a row should be kept, but the conversation was dropped"
        return False, {"rule": rule, "dropped": out.reason}
    if out.input_ids != expected_ids:
        evidence = token_evidence(tokenizer, expected_ids, out.input_ids)
        evidence["rule"] = f"{mode}: the row isn't the reference cut at {max_len}"
        return False, evidence
    report = mask_report(expected_zones, out)
    evidence = report.evidence(tokenizer, out, expected_zones)
    evidence["supervised_tokens"] = sum(label != LABEL_IGNORE for label in out.labels)
    if report.violated:
        evidence["rule"] = f"{mode}: the kept row's loss doesn't follow the reference zones"
        return False, evidence
    return True, evidence


def limit_for(fixture: Fixture, ref: TemplateReference) -> int:
    """The fixture's own max_len, or its exact reference length (the boundary)."""
    return fixture.max_len if fixture.max_len is not None else len(ref.input_ids)


def cut_reference(ref: TemplateReference, max_len: int, mode: str) -> tuple[tuple[int, ...], tuple[Zone, ...]] | None:
    """The reference after the declared truncation, or None if it must be dropped."""
    ids, zones = ref.input_ids, ref.zones
    if len(ids) <= max_len:
        return ids, zones
    if mode == "drop":
        return None
    if mode == "right":
        return ids[:max_len], zones[:max_len]
    return ids[-max_len:], zones[-max_len:]


def _cut_is_inside_an_assistant_turn(ref: TemplateReference, max_len: int, mode: str) -> bool:
    position = len(ref.input_ids) - max_len if mode == "left" else max_len
    return any(span.token_start < position < span.token_message_end for span in ref.spans)


def _plant(name: str, description: str, after_encode: Any) -> Plant:
    """All three truncation faults are synthetic: plausible classes, not tied to one known incident."""
    return Plant(
        name,
        description,
        lambda adapter: PlantedAdapter(adapter, name, after_encode=after_encode),
        origin="synthetic",
    )


def _at_limit(out: Encoded | Dropped, cfg: TrainCfg) -> bool:
    """Planted faults act only on the call with a length limit, and only on a row that reaches it."""
    return cfg.max_len is not None and isinstance(out, Encoded) and len(out.input_ids) >= cfg.max_len


def _off_by_one(mode: str):
    def after(out: Encoded | Dropped, conv: Conversation, cfg: TrainCfg) -> Encoded | Dropped:
        if not _at_limit(out, cfg):
            return out
        keep = slice(0, cfg.max_len - 1) if mode == "right" else slice(len(out.input_ids) - (cfg.max_len - 1), None)
        return Encoded(out.input_ids[keep], out.labels[keep], out.attention_mask[keep])

    return after


def _drop_at_limit(out: Encoded | Dropped, conv: Conversation, cfg: TrainCfg) -> Encoded | Dropped:
    return Dropped("planted: off-by-one length limit") if _at_limit(out, cfg) else out


def _lose_labels(ref: TemplateReference):
    def after(out: Encoded | Dropped, conv: Conversation, cfg: TrainCfg) -> Encoded | Dropped:
        if cfg.max_len is None:
            return out
        if isinstance(out, Dropped):  # drop mode: keep a right-cut row with no loss instead
            ids = ref.input_ids[: cfg.max_len]
            return Encoded(ids, (LABEL_IGNORE,) * len(ids), (1,) * len(ids))
        return replace(out, labels=(LABEL_IGNORE,) * len(out.labels))

    return after


def _reappend(closing: int):
    def after(out: Encoded | Dropped, conv: Conversation, cfg: TrainCfg) -> Encoded | Dropped:
        if not _at_limit(out, cfg):
            return out
        m = cfg.max_len
        return Encoded(
            out.input_ids[: m - 1] + (closing,),
            out.labels[: m - 1] + (closing,),
            out.attention_mask[:m],
        )

    return after
