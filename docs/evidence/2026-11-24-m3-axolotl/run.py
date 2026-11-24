"""Run the six Tier 1 checks on the real Axolotl adapter (calibration model); write verdicts, a summary, and a
comparison with predictions.md (committed before the run).

Run from the repo root, in Axolotl's environment:
    envs/axolotl-0.21.1/.venv/bin/python docs/evidence/2026-11-24-m3-axolotl/run.py
"""

from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

OUT = Path(__file__).parent
CELL = {"pass": "pass", "fail": "fail", "n/a": "not_applicable"}


def predictions() -> dict[tuple[str, str], str]:
    lines = (OUT / "predictions.md").read_text(encoding="utf-8").splitlines()
    header = next(line for line in lines if line.startswith("| Check |"))
    fixtures = [c.strip() for c in header.strip("|").split("|")[1:]]
    out = {}
    for line in lines:
        if line.startswith("| t1."):
            cells = [c.strip().strip("*") for c in line.strip("|").split("|")]
            for fixture, value in zip(fixtures, cells[1:]):
                out[(cells[0], fixture)] = CELL[value]
    return out


def main() -> None:
    from parity.adapters.axolotl import AxolotlAdapter
    from parity.adapters.base import TrainCfg
    from parity.checks.common import official_template
    from parity.checks.tier1 import TIER1_CHECKS
    from parity.engine.report import render_summary
    from parity.engine.runner import evaluate_suite
    from parity.engine.verdict import write_verdict
    from parity.fixtures import load_all
    from parity.references.properties import template_properties_for
    from parity.release import load_release
    from parity.targets import CALIBRATION

    release = load_release(CALIBRATION)
    adapter = AxolotlAdapter(release)
    fixtures = load_all()
    verdicts = evaluate_suite(TIER1_CHECKS, adapter, fixtures)

    verdict_dir = OUT / "verdicts"
    shutil.rmtree(verdict_dir, ignore_errors=True)
    for v in verdicts:
        write_verdict(v, verdict_dir)

    declared = adapter.declare(TrainCfg())
    properties = template_properties_for(release, official_template(release), "model", fixtures)
    if declared.chat_template != official_template(release):
        properties += template_properties_for(release, declared.chat_template, declared.template_source, fixtures)
    (OUT / "summary.md").write_text(render_summary(verdicts, properties), encoding="utf-8")

    predicted = predictions()
    rows, misses = [], []
    for v in verdicts:
        actual = v.result if v.result != "fail" else ("fail" if v.kind == "bug" else "difference")
        expected = predicted[(v.check, v.fixture)]
        ok = actual == expected
        rows.append(f"| {v.check} | {v.fixture} | {expected} | {actual} | {'yes' if ok else '**no**'} | {v.self_test} |")
        if not ok:
            misses.append((v.check, v.fixture, expected, actual, v.reason))
    compare = [
        "# Axolotl on Qwen3: predicted vs actual",
        "",
        f"{len(verdicts) - len(misses)} of {len(verdicts)} cells as predicted.",
        "",
        "| Check | Fixture | Predicted | Actual | Match | Self-test |",
        "| --- | --- | --- | --- | --- | --- |",
        *rows,
        "",
    ]
    (OUT / "compare.md").write_text("\n".join(compare), encoding="utf-8")
    table = {f"{v.check} / {v.fixture}": (v.result, v.kind, v.known_difference, v.self_test, v.flags) for v in verdicts}
    (OUT / "results.json").write_text(json.dumps(table, indent=1) + "\n", encoding="utf-8")
    print(Counter((v.result, v.kind) for v in verdicts))
    print(f"{len(verdicts) - len(misses)} of {len(verdicts)} as predicted")
    for miss in misses:
        print("NOT AS PREDICTED:", miss)


if __name__ == "__main__":
    main()
