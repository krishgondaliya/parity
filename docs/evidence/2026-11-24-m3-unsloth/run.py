"""Run the six Tier 1 checks on the real Unsloth adapter (calibration model), primary (thinking) and secondary
(non-thinking) tables; write verdicts, a summary, and a comparison with predictions.md (committed before the run).

Run from the repo root, in Unsloth's environment:
    envs/unsloth-2026.11.10/.venv/bin/python docs/evidence/2026-11-24-m3-unsloth/run.py
Add --relink to re-link saved verdicts to findings/root_causes.json and re-render, without running the framework.
"""

from __future__ import annotations

import sys

if "--relink" not in sys.argv:
    from parity.adapters.unsloth import UnslothAdapter  # Unsloth is imported first, as its notebooks do

import json
import shutil
from collections import Counter
from pathlib import Path

OUT = Path(__file__).parent
MODES = ("thinking", "non_thinking")
CELL = {"pass": "pass", "fail": "fail", "n/a": "not_applicable", "inconclusive": "inconclusive"}


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
    from parity.adapters.base import TrainCfg
    from parity.checks.common import official_template
    from parity.checks.tier1 import tier1_checks
    from parity.engine.report import render_summary
    from parity.engine.root_causes import count_fails, link_root_causes, pin
    from parity.engine.runner import evaluate_suite
    from parity.engine.verdict import read_verdict, write_verdict
    from parity.fixtures import load_all
    from parity.references.properties import template_properties_for
    from parity.release import load_release
    from parity.targets import CALIBRATION

    release = load_release(CALIBRATION)
    fixtures = load_all()
    if "--relink" in sys.argv:
        verdicts = [read_verdict(p) for p in sorted((OUT / "verdicts").rglob("*.json"))]
        templates = json.loads((OUT / "templates.json").read_text(encoding="utf-8"))
    else:
        adapter = UnslothAdapter(release)
        verdicts = []
        for mode in MODES:
            verdicts += evaluate_suite(tier1_checks(mode), adapter, fixtures)
        declared = adapter.declare(TrainCfg())
        templates = {"model": official_template(release)}
        if declared.chat_template != templates["model"]:
            templates[declared.template_source] = declared.chat_template
        (OUT / "templates.json").write_text(json.dumps(templates, indent=1) + "\n", encoding="utf-8")

    verdicts = link_root_causes(verdicts)
    shutil.rmtree(OUT / "verdicts", ignore_errors=True)
    for v in verdicts:
        write_verdict(v, OUT / "verdicts" / v.serving_mode)
    properties = []
    for mode in MODES:
        for source, template in templates.items():
            properties += template_properties_for(release, template, source, fixtures, mode)
    backend = (
        "> **Backend: CPU** (Unsloth's CPU fallback; `envs/unsloth-2026.11.10/environment.json`). These are Unsloth's "
        "CPU-backend results. Once a GPU run exists, Unsloth's GPU route is its primary result and this becomes the "
        "labeled secondary \"CPU backend\" table (the human's decision, 2026-11-24).\n\n"
    )
    (OUT / "summary.md").write_text(backend + render_summary(verdicts, properties), encoding="utf-8")

    predicted = predictions()
    rows, misses = [], []
    for v in (v for v in verdicts if v.serving_mode == "thinking"):
        actual = v.result if v.result != "fail" else ("fail" if v.kind == "bug" else "difference")
        expected = predicted[(v.check, v.fixture)]
        ok = actual == expected
        rows.append(f"| {v.check} | {v.fixture} | {expected} | {actual} | {'yes' if ok else '**no**'} | {v.self_test} |")
        if not ok:
            misses.append((v.check, v.fixture, expected, actual, v.reason))
    n = len(rows)
    compare = ["# Unsloth on Qwen3 (primary, thinking mode): predicted vs actual", "",
               f"{n - len(misses)} of {n} cells as predicted. See predictions.md for which predictions rest on a probe.", "",
               "| Check | Fixture | Predicted | Actual | Match | Self-test |", "| --- | --- | --- | --- | --- | --- |", *rows, ""]
    (OUT / "compare.md").write_text("\n".join(compare), encoding="utf-8")

    table = {
        f"{v.serving_mode} / {v.check} / {v.fixture}": {
            "result": v.result, "kind": v.kind, "known_difference": v.known_difference, "root_cause": v.root_cause,
            "self_test": v.self_test, "flags": v.flags,
        }
        for v in verdicts
    }  # fmt: skip
    (OUT / "results.json").write_text(json.dumps(table, indent=1) + "\n", encoding="utf-8")
    untriaged = [{"verdict": f"{v.serving_mode}/{v.check}__{v.fixture}", "pin": pin(v), "evidence": v.evidence}
                 for v in verdicts if v.result == "fail" and v.kind == "bug" and not v.root_cause]  # fmt: skip
    (OUT / "untriaged.json").write_text(json.dumps(untriaged, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    for mode in MODES:
        print(mode, sorted(Counter((v.result, v.kind) for v in verdicts if v.serving_mode == mode).items(), key=str))
    print(f"{n - len(misses)} of {n} primary cells as predicted")
    for miss in misses:
        print("NOT AS PREDICTED:", miss[:4], (miss[4] or "")[:160])
    count = count_fails(verdicts)
    print(f"fail verdicts: {count.cells} ({count.bugs} bug, {count.differences} difference); untriaged {len(count.untriaged)}")


if __name__ == "__main__":
    main()
