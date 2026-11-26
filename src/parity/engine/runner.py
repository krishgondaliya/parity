"""Run one check on one fixture together with its planted faults, and combine them into a verdict.

Order of decisions:

1. Applicability, from the fixture, the model and the serving mode alone: the
   check's own rule, then the model-level rule (`parity.serving.mode_not_documented`:
   a model that documents no such mode, D3), then the serving-mode rule
   (`parity.serving.not_applicable_in_mode`). Not applicable means the check is
   not run at all.
2. The framework's declaration. ModeNotConfigurable makes the cell not_configurable;
   FrameworkRefused (one of the adapter's pinned refusals, D2) makes it not_supported.
   Nothing is encoded in either case.
3. The clean run. A reference that cannot be computed without guessing, or
   any error, makes it inconclusive, never pass.
4. Every planted fault that applies to this fixture, run the same way. A
   fault counts as caught only if its run ends in a `fail` of kind `bug`.
5. If any applicable fault was missed, or none applies, the verdict is
   inconclusive whatever the clean run said (hard rule 3). The clean result is
   kept in the evidence so nothing is lost.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from parity.adapters.base import FrameworkRefused, ModeNotConfigurable
from parity.engine.verdict import Verdict, make_verdict
from parity.fixtures import Fixture
from parity.references.properties import marker_disagreement_property_id
from parity.references.template import MarkerDisagreementError, TemplateReferenceError
from parity.serving import mode_not_documented, not_applicable_in_mode

BLOCKING_CHECK = "t1.template_parity"


def evaluate_suite(checks: Sequence[Any], adapter: Any, fixtures: Sequence[Fixture]) -> list[Verdict]:
    """Every check on every fixture. One root cause, one finding.

    When template parity fails (kind bug) on a fixture, the other checks'
    fails on that fixture are held back as "blocked by t1.template_parity":
    inconclusive, flagged, with their own result kept in the evidence. Their
    passes stay passes. A matched known difference blocks nothing.

    One serving mode per call: verdicts are matched to template parity by check id.
    """
    if len({getattr(check, "serving_mode", None) for check in checks}) > 1:
        raise ValueError("evaluate_suite judges one serving mode per call; call it once per mode")
    verdicts = []
    for fixture in fixtures:
        by_check = {check.id: evaluate(check, adapter, fixture) for check in checks}
        parity = by_check.get(BLOCKING_CHECK)
        blocking = parity is not None and parity.result == "fail" and parity.kind == "bug"
        for check in checks:
            v = by_check[check.id]
            if blocking and check.id != BLOCKING_CHECK and v.result == "fail":
                v = _blocked(v, parity)
            verdicts.append(v)
    return verdicts


def _blocked(v: Verdict, parity: Verdict) -> Verdict:
    at = parity.evidence.get("first_divergence")
    return replace(
        v,
        result="inconclusive",
        kind=None,
        known_difference=None,
        reason=(
            f"blocked by {BLOCKING_CHECK}: template parity already fails on this fixture (first divergence at "
            f"token {at}). One root cause, one finding; this check's own fail is kept in the evidence."
        ),
        flags=sorted(set(v.flags) | {"blocked_by_template_parity"}),
        evidence={**v.evidence, "blocked_result": v.result, "blocked_kind": v.kind},
    )


def as_not_configurable(verdict: Verdict, reason: str, decision: str) -> Verdict:
    """The verdict the runner now gives this saved cell, after a decision made its serving mode not configurable.

    Used only to re-render saved evidence (P-0006, 2026-11-25). No new framework run is needed: the runner decides
    not_configurable when `declare()` raises ModeNotConfigurable, before the framework is touched. The fields follow
    `_evaluate` and `evaluate`:
    - a not_applicable verdict stays as it is, since the runner decides that first, from the fixture and the mode;
    - an already re-derived verdict stays as it is.
    The earlier result is kept in the evidence, next to the decision that changed it.
    """
    if verdict.result in ("not_applicable", "not_configurable"):
        return verdict
    full = f"serving mode: {reason}"
    return replace(
        verdict,
        result="not_configurable",
        kind=None,
        known_difference=None,
        self_test=None,
        reason=full,
        evidence={"rederived": decision, "as_run_result": verdict.result, "as_run_kind": verdict.kind},
        conventions={},
        reference={},
        plants=[],
        declaration={"error": full},
        template_properties=[],
        flags=[],
        root_cause=None,
    )


def evaluate(check: Any, adapter: Any, fixture: Fixture) -> Verdict:
    """Every verdict this returns records the adapter's declaration (or why it couldn't be read)."""
    verdict = _evaluate(check, adapter, fixture)
    if verdict.serving_mode is None and hasattr(check, "mode"):
        verdict = replace(verdict, serving_mode=check.mode(adapter.release))
    if verdict.result in ("not_configurable", "not_supported"):
        return replace(verdict, declaration={"error": verdict.reason})
    if verdict.result != "not_applicable" and hasattr(check, "reference_properties"):
        try:
            extra = check.reference_properties(adapter.release, fixture, adapter.declare(check.base_cfg))
        except Exception:  # noqa: BLE001 - a property lookup must never turn a verdict into a pass or a crash
            extra = []
        if extra:
            verdict = replace(verdict, template_properties=sorted(set(verdict.template_properties) | set(extra)))
    if verdict.declaration:
        return verdict
    try:
        record = adapter.declare(check.base_cfg).record(cfg=check.base_cfg)
    except Exception as exc:  # noqa: BLE001 - recorded, never hidden
        record = {"error": f"{type(exc).__name__}: {exc}"}
    return replace(verdict, declaration=record)


def _evaluate(check: Any, adapter: Any, fixture: Fixture) -> Verdict:
    reason = check.not_applicable(fixture)
    if reason is None and hasattr(check, "mode"):
        reason = mode_not_documented(adapter.release, check.mode(adapter.release))
    if reason is None and hasattr(check, "mode"):
        reason = not_applicable_in_mode(fixture, check.mode(adapter.release))
    if reason is not None:
        return make_verdict(check.id, adapter, fixture.id, "not_applicable", reason=reason)

    release = adapter.release
    try:
        adapter.declare(check.base_cfg)
    except ModeNotConfigurable as exc:
        return make_verdict(check.id, adapter, fixture.id, "not_configurable", reason=f"serving mode: {exc}")
    except FrameworkRefused as exc:
        return make_verdict(check.id, adapter, fixture.id, "not_supported", reason=f"framework refuses: {exc}")
    clean = run_safely(check, adapter, fixture)
    try:
        plants = list(check.plants(release, fixture, adapter.declare))
    except Exception as exc:  # noqa: BLE001 - any setup error must end inconclusive, never pass
        return _inconclusive(clean, [], f"could not set up the planted faults: {type(exc).__name__}: {exc}")

    records = []
    for plant in plants:
        planted = run_safely(check, plant.apply(adapter), fixture)
        records.append(
            {
                "plant": plant.name,
                "description": plant.description,
                "origin": plant.origin,
                "source": plant.source,
                "caught": planted.result == "fail" and planted.kind == "bug",
                "result": planted.result,
                "kind": planted.kind,
                "reason": planted.reason,
            }
        )

    if plants and all(r["caught"] for r in records):
        return replace(clean, self_test="caught", plants=records)
    if not plants:
        why = "no planted fault applies to this fixture, so the check cannot prove itself here"
    else:
        why = "planted fault(s) not caught: " + ", ".join(r["plant"] for r in records if not r["caught"])
    return _inconclusive(clean, records, why)


def run_safely(check: Any, adapter: Any, fixture: Fixture) -> Verdict:
    """Run the comparison; anything that stops it becomes inconclusive with the reason recorded."""
    mode = check.mode(adapter.release) if hasattr(check, "mode") else None
    try:
        return check.run(adapter, adapter.release, fixture)
    except MarkerDisagreementError as exc:
        return make_verdict(
            check.id,
            adapter,
            fixture.id,
            "inconclusive",
            reason=f"reference: {exc}",
            evidence={"marker_disagreement": exc.positions[:16], "triage": "possible model-template finding"},
            flags=["marker_disagreement"],
            template_properties=[marker_disagreement_property_id(adapter.release, exc.template_sha256, mode)],
            serving_mode=mode,
        )
    except TemplateReferenceError as exc:
        return make_verdict(check.id, adapter, fixture.id, "inconclusive", reason=f"reference: {exc}", serving_mode=mode)
    except Exception as exc:  # noqa: BLE001 - a crash is never a pass
        return make_verdict(
            check.id, adapter, fixture.id, "inconclusive", reason=f"error: {type(exc).__name__}: {exc}"
        )


def _inconclusive(clean: Verdict, records: list[dict[str, Any]], why: str) -> Verdict:
    reason = why if clean.result != "inconclusive" else f"{clean.reason}; also {why}"
    evidence = {**clean.evidence, "clean_result": clean.result, "clean_kind": clean.kind}
    if clean.known_difference:
        evidence["clean_known_difference"] = clean.known_difference
    return replace(
        clean,
        result="inconclusive",
        kind=None,
        known_difference=None,
        self_test="missed",
        reason=reason,
        evidence=evidence,
        plants=records,
    )
