"""Re-run the six Tier 1 checks on TRL and Axolotl (calibration model) after the 2026-11-24 decisions:
serving modes (primary thinking, secondary non-thinking), KD-0002/0003, root-cause linking, the token-level
false->False plant, the tool_arguments property, and added-token structural headers.

Run from the repo root, in the framework's own environment:
    envs/trl-1.18.2/.venv/bin/python docs/evidence/2026-11-24-m3-rerun/run.py trl
    envs/axolotl-0.21.1/.venv/bin/python docs/evidence/2026-11-24-m3-rerun/run.py axolotl
Add --relink to re-link saved verdicts to findings/root_causes.json and re-render, without running the framework.

Writes <framework>/verdicts/<mode>/*.json, <framework>/summary.md, <framework>/results.json, and
<framework>/untriaged.json (every fail with no root cause, with the pin triage would add).
"""

from __future__ import annotations

import json
import shutil
import sys
from collections import Counter
from pathlib import Path

OUT = Path(__file__).parent
MODES = ("thinking", "non_thinking")


def main() -> None:
    from parity.checks.common import official_template
    from parity.engine.report import render_summary
    from parity.engine.root_causes import count_fails, link_root_causes, pin
    from parity.engine.verdict import read_verdict, write_verdict
    from parity.fixtures import load_all
    from parity.references.properties import template_properties_for
    from parity.release import load_release
    from parity.targets import CALIBRATION

    framework, relink = sys.argv[1], "--relink" in sys.argv
    out = OUT / framework
    release = load_release(CALIBRATION)
    fixtures = load_all()

    if relink:
        verdicts = [read_verdict(p) for p in sorted((out / "verdicts").rglob("*.json"))]
        templates = json.loads((out / "templates.json").read_text(encoding="utf-8"))
    else:
        from parity.adapters.base import TrainCfg
        from parity.checks.tier1 import tier1_checks
        from parity.engine.runner import evaluate_suite

        if framework == "trl":
            from parity.adapters.trl import TRLAdapter as Adapter
        else:
            from parity.adapters.axolotl import AxolotlAdapter as Adapter
        adapter = Adapter(release)
        verdicts = []
        for mode in MODES:
            verdicts += evaluate_suite(tier1_checks(mode), adapter, fixtures)
        declared = adapter.declare(TrainCfg())
        templates = {"model": official_template(release)}
        if declared.chat_template != templates["model"]:
            templates[declared.template_source] = declared.chat_template
        out.mkdir(parents=True, exist_ok=True)
        (out / "templates.json").write_text(json.dumps(templates, indent=1) + "\n", encoding="utf-8")

    verdicts = link_root_causes(verdicts)
    shutil.rmtree(out / "verdicts", ignore_errors=True)
    for v in verdicts:
        write_verdict(v, out / "verdicts" / v.serving_mode)

    properties = []
    for mode in MODES:
        for source, template in templates.items():
            properties += template_properties_for(release, template, source, fixtures, mode)
    (out / "summary.md").write_text(render_summary(verdicts, properties), encoding="utf-8")

    table = {
        f"{v.serving_mode} / {v.check} / {v.fixture}": {
            "result": v.result, "kind": v.kind, "known_difference": v.known_difference, "root_cause": v.root_cause,
            "self_test": v.self_test, "flags": v.flags,
        }
        for v in verdicts
    }  # fmt: skip
    (out / "results.json").write_text(json.dumps(table, indent=1) + "\n", encoding="utf-8")
    untriaged = [
        {"verdict": f"{v.serving_mode}/{v.check}__{v.fixture}", "pin": pin(v), "evidence": v.evidence}
        for v in verdicts
        if v.result == "fail" and v.kind == "bug" and not v.root_cause
    ]
    (out / "untriaged.json").write_text(json.dumps(untriaged, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    for mode in MODES:
        print(mode, sorted(Counter((v.result, v.kind) for v in verdicts if v.serving_mode == mode).items(), key=str))
    count = count_fails(verdicts)
    print(f"fail verdicts: {count.cells}; root causes: {count.total} ({count.root_causes}, untriaged {len(count.untriaged)})")
    for item in count.untriaged:
        print("UNTRIAGED:", item)


if __name__ == "__main__":
    main()
