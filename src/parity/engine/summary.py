"""Roll verdicts up into one cell per (model, framework, check).

Each cell reports how many fixtures the check actually ran on (everything that
was not `not_applicable`). A cell where the check ran on zero fixtures is
inconclusive: nothing was tested, so it can't be a pass.

Cell result, first rule that applies:
  ran on 0 fixtures            -> inconclusive
  any fail of kind bug         -> fail
  any inconclusive             -> inconclusive
  any matched known difference -> difference (listed by entry id)
  otherwise                    -> pass
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from parity.engine.verdict import Verdict


@dataclass(frozen=True)
class Cell:
    model: str
    framework: str
    check: str
    fixtures: int
    ran_on: int
    passed: int
    failed: int  # kind bug
    differences: int  # kind difference
    inconclusive: int
    not_applicable: int
    known_differences: tuple[str, ...]
    template_properties: tuple[str, ...]  # model-level properties the cell's verdicts inherit
    result: str


def summarize(verdicts: Iterable[Verdict]) -> list[Cell]:
    groups: dict[tuple[str, str, str], list[Verdict]] = defaultdict(list)
    for v in verdicts:
        groups[(v.model, v.framework, v.check)].append(v)

    cells = []
    for (model, framework, check), vs in sorted(groups.items()):
        passed = sum(v.result == "pass" for v in vs)
        failed = sum(v.result == "fail" and v.kind == "bug" for v in vs)
        differences = sum(v.result == "fail" and v.kind == "difference" for v in vs)
        inconclusive = sum(v.result == "inconclusive" for v in vs)
        not_applicable = sum(v.result == "not_applicable" for v in vs)
        ran_on = len(vs) - not_applicable
        if ran_on == 0:
            result = "inconclusive"
        elif failed:
            result = "fail"
        elif inconclusive:
            result = "inconclusive"
        elif differences:
            result = "difference"
        else:
            result = "pass"
        cells.append(
            Cell(
                model=model,
                framework=framework,
                check=check,
                fixtures=len(vs),
                ran_on=ran_on,
                passed=passed,
                failed=failed,
                differences=differences,
                inconclusive=inconclusive,
                not_applicable=not_applicable,
                known_differences=tuple(sorted({v.known_difference for v in vs if v.known_difference})),
                template_properties=tuple(sorted({p for v in vs for p in v.template_properties})),
                result=result,
            )
        )
    return cells
