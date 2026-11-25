"""Calibration status (M4): is a framework's run on the calibration model green?

Green (the human's definition, 2026-11-25) means:
- no untriaged fail;
- no harness-caused inconclusive;
- every planted fault caught.
Fails linked to a triaged finding (P-0002, P-0005) or to a known difference are allowed, and listed.

Each inconclusive verdict is attributed to one side:
- **framework:** the framework's own output stops this check from judging. Either the cell is blocked by a
  t1.template_parity fail on the same fixture (flag `blocked_by_template_parity`), or the framework's rows don't
  line up with its declared template's render. Either way the template parity fail on that fixture explains it, so
  the inconclusive is allowed only if that fail is linked to a root cause.
- **harness:** anything else, and anything unrecognized, to be safe. That covers our reference refusing a fixture
  ("reference: ..."), a declared/actual mismatch (the adapter), a crash, and the packing case no Phase 0 check can
  judge. These block green.
A planted fault counts as missed wherever its record says not caught, whichever side the verdict is attributed to.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field

from parity.engine.verdict import Verdict, framework_label

UNALIGNED = "the framework's tokens differ from its own declared template's render"
PARITY = "t1.template_parity"


@dataclass(frozen=True)
class Cell:
    mode: str
    check: str
    fixture: str
    detail: str = ""

    def __str__(self) -> str:
        return f"{self.mode} {self.check}/{self.fixture}" + (f": {self.detail}" if self.detail else "")


@dataclass
class Status:
    framework: str
    results: dict[str, int] = field(default_factory=dict)
    linked_fails: dict[str, list[Cell]] = field(default_factory=dict)  # root cause -> cells
    untriaged_fails: list[Cell] = field(default_factory=list)
    framework_inconclusives: list[Cell] = field(default_factory=list)  # explained by a linked template parity fail
    harness_inconclusives: list[Cell] = field(default_factory=list)
    missed_plants: list[Cell] = field(default_factory=list)

    @property
    def green(self) -> bool:
        return not (self.untriaged_fails or self.harness_inconclusives or self.missed_plants)


def attribute(verdicts: Iterable[Verdict]) -> list[Status]:
    """One Status per framework (with its backend), in name order."""
    by_framework: dict[str, list[Verdict]] = {}
    for v in verdicts:
        by_framework.setdefault(framework_label(v), []).append(v)
    return [_status(name, vs) for name, vs in sorted(by_framework.items())]


def _status(name: str, verdicts: list[Verdict]) -> Status:
    status = Status(name, results=dict(sorted(Counter(v.result for v in verdicts).items())))
    parity = {(v.serving_mode, v.fixture): v for v in verdicts if v.check == PARITY}
    for v in verdicts:
        cell = Cell(v.serving_mode or "unrecorded", v.check, v.fixture)
        if v.result == "fail":
            if v.root_cause:
                status.linked_fails.setdefault(v.root_cause, []).append(cell)
            else:
                status.untriaged_fails.append(cell)
        elif v.result == "inconclusive":
            gate = parity.get((v.serving_mode, v.fixture))
            framework_side = "blocked_by_template_parity" in v.flags or (v.reason or "").startswith(UNALIGNED)
            if framework_side and gate is not None and gate.result == "fail" and gate.root_cause:
                status.framework_inconclusives.append(Cell(cell.mode, cell.check, cell.fixture, f"by {gate.root_cause}"))
            else:
                status.harness_inconclusives.append(Cell(cell.mode, cell.check, cell.fixture, (v.reason or "")[:160]))
        for plant in v.plants:
            if not plant.get("caught"):
                status.missed_plants.append(Cell(cell.mode, cell.check, cell.fixture, str(plant.get("plant"))))
    return status


def table_rows(verdicts: Iterable[Verdict]) -> list[str]:
    """Markdown rows, one per framework x mode x check: the clean result over the fixtures, and the planted faults."""
    groups: dict[tuple[str, str, str], list[Verdict]] = {}
    for v in verdicts:
        groups.setdefault((framework_label(v), v.serving_mode or "unrecorded", v.check), []).append(v)
    rows = []
    for (name, mode, check), vs in sorted(groups.items()):
        clean = ", ".join(f"{n} {r}" for r, n in sorted(Counter(_shown(v) for v in vs).items()))
        plants = [p for v in vs for p in v.plants]
        caught = sum(1 for p in plants if p.get("caught"))
        planted = f"{caught} of {len(plants)} caught" if plants else "none run"
        notes = sorted({f"root cause {v.root_cause}" for v in vs if v.root_cause})
        rows.append(f"| {name} | {mode} | {check} | {clean} | {planted} | {'; '.join(notes)} |")
    return rows


def _shown(v: Verdict) -> str:
    if v.result == "fail":
        return "difference" if v.kind == "difference" else "fail"
    return {"not_applicable": "n/a"}.get(v.result, v.result)
