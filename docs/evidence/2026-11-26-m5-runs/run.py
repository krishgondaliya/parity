"""M5: the six Tier 1 checks on one Phase 0 target through one framework, both serving modes, predictions first.

Run from the repo root, in the framework's own environment (the adapter checks it is the right one):
    envs/trl-1.18.2/vlm/.venv/bin/python docs/evidence/2026-11-26-m5-runs/run.py trl thinkingmachines/Inkling-2-Mini
    envs/trl-1.18.2/.venv/bin/python     docs/evidence/2026-11-26-m5-runs/run.py trl deepseek-ai/DeepSeek-V4.1-Flash-1017
    envs/axolotl-0.21.1/.venv/bin/python docs/evidence/2026-11-26-m5-runs/run.py axolotl deepseek-ai/DeepSeek-V4.1-Flash-1017
Add --relink to re-link saved verdicts to findings/root_causes.json without re-running the framework.
Unsloth runs on GPU, through the Colab kit (docs/gpu-run/), which applies the same guard to its archive.

Before anything else it finds the pair's predictions (parity.engine.predictions): a predictions.md committed in HEAD,
unchanged in the working tree, whose title names the framework, its version and the release's exact revision. No
such file, no run. Then it writes into that file's folder:
    verdicts/<mode>/*.json   every verdict, root causes linked
    templates.json           the model's template, and the declared one if it differs
    summary.md               the cells per check and mode, and the template properties
    results.json             one line per verdict
    untriaged.json           every fail with no root cause, with the pin /triage would add
    compare.md               predicted vs actual, cell by cell ("N of M cells as predicted")
    run.json                 the framework, environment, lockfile, model build and timing
"""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
MODES = ("thinking", "non_thinking")
ADAPTERS = {
    "trl": ("parity.adapters.trl", "TRLAdapter", "TRL"),
    "axolotl": ("parity.adapters.axolotl", "AxolotlAdapter", "Axolotl"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("framework", choices=sorted(ADAPTERS))
    parser.add_argument("target", help="org/name of a Phase 0 target")
    parser.add_argument("--relink", action="store_true")
    args = parser.parse_args()

    from parity.engine.predictions import committed_predictions
    from parity.targets import PHASE0_TARGETS

    module_name, class_name, display = ADAPTERS[args.framework]
    release_pin = next((t for t in PHASE0_TARGETS if t.repo_id == args.target), None)
    if release_pin is None:
        sys.exit(f"{args.target} is not a Phase 0 target")
    module = importlib.import_module(module_name)  # the framework is imported here, before any check runs
    predictions = committed_predictions(REPO, display, module.VERSION, release_pin.spec)  # raises if missing
    out = REPO / predictions.folder
    print(f"predictions: {predictions.path}")

    from parity.checks.common import official_template
    from parity.engine.predictions import actual, parse_cells
    from parity.engine.report import render_summary
    from parity.engine.root_causes import count_fails, link_root_causes, pin
    from parity.engine.verdict import read_verdict, write_verdict
    from parity.fixtures import load_all
    from parity.references.properties import template_properties_for
    from parity.release import load_release
    from parity.serving import mode_not_documented, not_applicable_in_mode

    release, fixtures = load_release(release_pin), load_all()
    started = time.time()
    run_facts: dict = {}
    if args.relink:
        verdicts = [read_verdict(p) for p in sorted((out / "verdicts").rglob("*.json"))]
        templates = json.loads((out / "templates.json").read_text(encoding="utf-8"))
    else:
        from parity.adapters.base import TrainCfg
        from parity.checks.tier1 import tier1_checks
        from parity.engine.runner import evaluate_suite

        adapter = getattr(module, class_name)(release)
        verdicts = []
        for mode in MODES:
            verdicts += evaluate_suite(tier1_checks(mode), adapter, fixtures)
        templates = {"model": official_template(release)}
        try:
            declared = adapter.declare(TrainCfg())
            if declared.chat_template != templates["model"]:
                templates[declared.template_source] = declared.chat_template
            run_facts["model_build"] = declared.model_build
        except Exception as exc:  # noqa: BLE001 - e.g. a framework refusal; recorded, the verdicts say the rest
            run_facts["declare_error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
            run_facts["model_build"] = getattr(adapter, "_model_build", None)
        out.mkdir(parents=True, exist_ok=True)
        (out / "templates.json").write_text(json.dumps(templates, indent=1) + "\n", encoding="utf-8")
        run_facts.update(
            framework=f"{adapter.name}@{adapter.version}",
            env_lock=adapter.env_lock,
            env_fingerprint=adapter.env_fingerprint,
            model=release_pin.spec,
            predictions=predictions.path,
            seconds=round(time.time() - started, 1),
        )
        (out / "run.json").write_text(json.dumps(run_facts, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    verdicts = link_root_causes(verdicts)
    shutil.rmtree(out / "verdicts", ignore_errors=True)
    for v in verdicts:
        write_verdict(v, out / "verdicts" / v.serving_mode)

    properties = []
    for mode in MODES:
        if mode_not_documented(release, mode):
            continue
        applicable = [f for f in fixtures if not_applicable_in_mode(f, mode) is None]
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
    (out / "results.json").write_text(json.dumps(table, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    untriaged = [{"verdict": f"{v.serving_mode}/{v.check}__{v.fixture}", "pin": pin(v), "evidence": v.evidence}
                 for v in verdicts if v.result == "fail" and v.kind == "bug" and not v.root_cause]  # fmt: skip
    (out / "untriaged.json").write_text(json.dumps(untriaged, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    predicted = parse_cells(predictions.text)
    rows, hits = [], Counter()
    for v in sorted(verdicts, key=lambda v: (MODES.index(v.serving_mode), v.check, v.fixture)):
        expected, got = predicted[(v.serving_mode, v.check, v.fixture)], actual(v)
        match = expected == got
        hits[v.serving_mode] += match
        reason = (v.reason or "").replace("|", "/").replace("\n", " ")[:140] if not match else ""
        rows.append(f"| {v.serving_mode} | {v.check} | {v.fixture} | {expected} | {got} | "
                    f"{'yes' if match else '**no**'} | {reason} |")  # fmt: skip
    totals = Counter(v.serving_mode for v in verdicts)
    compare = [
        f"# {display} {module.VERSION} on {release_pin.spec}: predicted vs actual",
        "",
        f"{sum(hits.values())} of {len(verdicts)} cells as predicted: thinking (primary) {hits['thinking']} of "
        f"{totals['thinking']}, non-thinking (secondary) {hits['non_thinking']} of {totals['non_thinking']}.",
        "",
        f"Predictions: [`predictions.md`](predictions.md), committed before this run. Verdicts: `verdicts/`.",
        "",
        "| Mode | Check | Fixture | Predicted | Actual | Match | Reason, where it differs |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        *rows,
        "",
    ]
    (out / "compare.md").write_text("\n".join(compare), encoding="utf-8")

    for mode in MODES:
        print(mode, sorted(Counter((v.result, v.kind) for v in verdicts if v.serving_mode == mode).items(), key=str))
    print(f"{sum(hits.values())} of {len(verdicts)} cells as predicted")
    count = count_fails(verdicts)
    print(f"fail verdicts: {count.cells}; root causes: {count.root_causes}; untriaged {len(count.untriaged)}")


if __name__ == "__main__":
    main()
