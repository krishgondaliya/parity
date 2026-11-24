"""Run the six Tier 1 checks on the real TRL adapter (calibration model), and write verdicts and a summary.

Run from the repo root, in TRL's environment:
    envs/trl-1.18.2/.venv/bin/python docs/evidence/2026-11-24-m3-trl/run.py
"""

from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

from parity.adapters.base import TrainCfg
from parity.adapters.trl import TRLAdapter
from parity.checks.common import official_template
from parity.checks.tier1 import TIER1_CHECKS
from parity.engine.report import render_summary
from parity.engine.runner import evaluate_suite
from parity.engine.verdict import write_verdict
from parity.fixtures import load_all
from parity.references.properties import template_properties_for
from parity.release import load_release
from parity.targets import CALIBRATION

OUT = Path(__file__).parent


def main() -> None:
    release = load_release(CALIBRATION)
    adapter = TRLAdapter(release)
    fixtures = load_all()
    verdicts = evaluate_suite(TIER1_CHECKS, adapter, fixtures)

    verdict_dir = OUT / "verdicts"
    shutil.rmtree(verdict_dir, ignore_errors=True)
    for v in verdicts:
        write_verdict(v, verdict_dir)

    declared = adapter.declare(TrainCfg())
    properties = template_properties_for(release, official_template(release), "model", fixtures)
    properties += template_properties_for(release, declared.chat_template, declared.template_source, fixtures)
    (OUT / "summary.md").write_text(render_summary(verdicts, properties), encoding="utf-8")

    table = {f"{v.check} / {v.fixture}": (v.result, v.kind, v.known_difference, v.self_test, v.flags) for v in verdicts}
    (OUT / "results.json").write_text(json.dumps(table, indent=1) + "\n", encoding="utf-8")
    print(Counter((v.result, v.kind) for v in verdicts))
    for v in verdicts:
        if v.result not in ("pass", "not_applicable") and v.known_difference != "KD-0001":
            print("NOT AS EXPECTED:", v.check, v.fixture, v.result, v.kind, v.reason)


if __name__ == "__main__":
    main()
