"""Roll verdicts up into one cell per (model, framework, backend, check, serving mode).

A framework that records a backend (Unsloth: gpu or cpu) gets one cell per backend, named by
`framework_label` ("unsloth@2026.11.10 (gpu)"), so results from two backends are never merged.

Each cell reports how many fixtures the check actually ran on (everything that
was not `not_applicable`). A cell where the check ran on zero fixtures is
inconclusive: nothing was tested, so it can't be a pass.

Cell result, first rule that applies:
  not configurable everywhere  -> not_configurable (the framework documents no way to set the serving mode)
  ran on 0 fixtures            -> inconclusive
  any fail of kind bug         -> fail
  any inconclusive             -> inconclusive
  any matched known difference -> difference (listed by entry id)
  otherwise                    -> pass
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from parity.engine.verdict import Verdict, framework_label, require_self_test


@dataclass(frozen=True)
class Cell:
    model: str
    framework: str  # framework_label: "<name>@<version>", plus " (<backend>)" when one is recorded
    check: str
    serving_mode: str  # the reference's mode (parity.serving); "unrecorded" for verdicts written before modes existed
    fixtures: int
    ran_on: int
    passed: int
    failed: int  # kind bug
    differences: int  # kind difference
    inconclusive: int
    not_applicable: int
    known_differences: tuple[str, ...]
    template_properties: tuple[str, ...]  # model-level properties the cell's verdicts inherit
    flags: dict[str, int]  # how many of the cell's verdicts carry each flag
    result: str
    root_causes: tuple[str, ...] = ()  # the findings that explain this cell's fails (engine.root_causes)
    untriaged: int = 0  # fails in this cell with no root cause
    not_configurable: int = 0  # verdicts where the framework can't be set to this serving mode
    not_configurable_reasons: tuple[str, ...] = ()
    backend: str | None = None  # Verdict.backend, shared by every verdict in the cell


def summarize(verdicts: Iterable[Verdict]) -> list[Cell]:
    groups: dict[tuple[str, str, str, str], list[Verdict]] = defaultdict(list)
    for v in verdicts:
        groups[(v.model, framework_label(v), v.check, v.serving_mode or "unrecorded")].append(require_self_test(v))

    cells = []
    for (model, framework, check, mode), vs in sorted(groups.items()):
        passed = sum(v.result == "pass" for v in vs)
        failed = sum(v.result == "fail" and v.kind == "bug" for v in vs)
        differences = sum(v.result == "fail" and v.kind == "difference" for v in vs)
        inconclusive = sum(v.result == "inconclusive" for v in vs)
        not_applicable = sum(v.result == "not_applicable" for v in vs)
        not_configurable = sum(v.result == "not_configurable" for v in vs)
        ran_on = len(vs) - not_applicable - not_configurable
        if not_configurable and ran_on == 0:
            result = "not_configurable"
        elif ran_on == 0:
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
                serving_mode=mode,
                fixtures=len(vs),
                ran_on=ran_on,
                passed=passed,
                failed=failed,
                differences=differences,
                inconclusive=inconclusive,
                not_applicable=not_applicable,
                known_differences=tuple(sorted({v.known_difference for v in vs if v.known_difference})),
                template_properties=tuple(sorted({p for v in vs for p in v.template_properties})),
                flags=dict(sorted(Counter(f for v in vs for f in v.flags).items())),
                result=result,
                root_causes=tuple(sorted({v.root_cause for v in vs if v.result == "fail" and v.root_cause})),
                untriaged=sum(v.result == "fail" and not v.root_cause for v in vs),
                not_configurable=not_configurable,
                not_configurable_reasons=tuple(sorted({v.reason for v in vs if v.result == "not_configurable" and v.reason})),
                backend=vs[0].backend,
            )
        )
    return cells
