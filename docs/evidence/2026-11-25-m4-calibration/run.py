"""M4 calibration: the six Tier 1 checks and every planted fault on the calibration model, both serving modes, under
the current checks (including t1.special_tokens' split padding rules, 2026-11-25, and the serving-mode applicability
rule: a fixture whose assistant turn carries reasoning is not applicable in non-thinking mode, 2026-11-25).

Run from the repo root, in the framework's own environment:
    envs/trl-1.18.2/.venv/bin/python docs/evidence/2026-11-25-m4-calibration/run.py trl
    envs/axolotl-0.21.1/.venv/bin/python docs/evidence/2026-11-25-m4-calibration/run.py axolotl
Add --relink to re-link saved verdicts to findings/root_causes.json without re-running the framework.
Unsloth's calibration runs on GPU (the Colab kit); its zip is merged by `report.py`.

Writes <framework>/verdicts/<mode>/*.json, templates.json, summary.md, results.json and untriaged.json (every fail
with no root cause, with the pin /triage would add). `report.py` writes runs/calibration-<date>.md from them.
"""

from __future__ import annotations

import json
import shutil
import sys
from collections import Counter
from pathlib import Path

OUT = Path(__file__).parent
MODES = ("thinking", "non_thinking")
ADAPTERS = {"trl": ("parity.adapters.trl", "TRLAdapter"), "axolotl": ("parity.adapters.axolotl", "AxolotlAdapter")}


def main() -> None:
    import importlib

    from parity.checks.common import official_template
    from parity.engine.report import render_summary
    from parity.engine.root_causes import count_fails, link_root_causes, pin
    from parity.engine.verdict import read_verdict, write_verdict
    from parity.fixtures import load_all
    from parity.references.properties import template_properties_for
    from parity.release import load_release
    from parity.serving import not_applicable_in_mode
    from parity.targets import CALIBRATION

    framework, relink = sys.argv[1], "--relink" in sys.argv
    if framework not in ADAPTERS:
        raise SystemExit(f"framework must be one of {sorted(ADAPTERS)} (Unsloth calibrates on GPU)")
    out = OUT / framework
    release, fixtures = load_release(CALIBRATION), load_all()

    if relink:
        verdicts = [read_verdict(p) for p in sorted((out / "verdicts").rglob("*.json"))]
        templates = json.loads((out / "templates.json").read_text(encoding="utf-8"))
    else:
        from parity.adapters.base import TrainCfg
        from parity.checks.tier1 import tier1_checks
        from parity.engine.runner import evaluate_suite

        module, name = ADAPTERS[framework]
        adapter = getattr(importlib.import_module(module), name)(release)
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
        applicable = [f for f in fixtures if not_applicable_in_mode(f, mode) is None]  # what the checks judged
        for source, template in templates.items():
            properties += template_properties_for(release, template, source, applicable, mode)
    (out / "summary.md").write_text(render_summary(verdicts, properties), encoding="utf-8")
    table = {
        f"{v.serving_mode} / {v.check} / {v.fixture}": {
            "result": v.result, "kind": v.kind, "known_difference": v.known_difference, "root_cause": v.root_cause,
            "self_test": v.self_test, "flags": v.flags, "reason": v.reason,
        }
        for v in verdicts
    }  # fmt: skip
    (out / "results.json").write_text(json.dumps(table, indent=1) + "\n", encoding="utf-8")
    untriaged = [{"verdict": f"{v.serving_mode}/{v.check}__{v.fixture}", "pin": pin(v), "evidence": v.evidence}
                 for v in verdicts if v.result == "fail" and v.kind == "bug" and not v.root_cause]  # fmt: skip
    (out / "untriaged.json").write_text(json.dumps(untriaged, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    for mode in MODES:
        print(mode, sorted(Counter((v.result, v.kind) for v in verdicts if v.serving_mode == mode).items(), key=str))
    count = count_fails(verdicts)
    print(f"fail verdicts: {count.cells}; root causes: {count.root_causes}; untriaged {len(count.untriaged)}")
    for item in count.untriaged:
        print("UNTRIAGED:", item)


if __name__ == "__main__":
    main()
