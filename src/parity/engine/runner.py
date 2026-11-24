"""Run one check on one fixture together with its planted faults, and combine them into a verdict.

Order of decisions:

1. Applicability, from the fixture alone. Not applicable means the check is
   not run at all.
2. The clean run. A reference that cannot be computed without guessing, or
   any error, makes it inconclusive, never pass.
3. Every planted fault that applies to this fixture, run the same way. A
   fault counts as caught only if its run ends in a `fail` of kind `bug`.
4. If any applicable fault was missed, or none applies, the verdict is
   inconclusive whatever the clean run said (hard rule 3). The clean result is
   kept in the evidence so nothing is lost.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from parity.engine.verdict import Verdict, make_verdict
from parity.fixtures import Fixture
from parity.references.template import MarkerDisagreementError, TemplateReferenceError


def evaluate(check: Any, adapter: Any, fixture: Fixture) -> Verdict:
    reason = check.not_applicable(fixture)
    if reason is not None:
        return make_verdict(check.id, adapter, fixture.id, "not_applicable", reason=reason)

    release = adapter.release
    clean = run_safely(check, adapter, fixture)
    try:
        declaration = adapter.declare(check.base_cfg)
        plants = list(check.plants(release, fixture, declaration))
    except Exception as exc:  # noqa: BLE001 - any setup error must end inconclusive, never pass
        return _inconclusive(clean, [], f"could not set up the planted faults: {type(exc).__name__}: {exc}")

    records = []
    for plant in plants:
        planted = run_safely(check, plant.apply(adapter), fixture)
        records.append(
            {
                "plant": plant.name,
                "description": plant.description,
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
        )
    except TemplateReferenceError as exc:
        return make_verdict(check.id, adapter, fixture.id, "inconclusive", reason=f"reference: {exc}")
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
