"""Run the six Tier 1 checks on the real Unsloth adapter, both serving modes, for each model given.

Runs inside Unsloth's environment, started by `colab_kit.py` (the GPU kit) from the root of the unpacked git archive:
    <env>/.venv/bin/python docs/gpu-run/run_checks.py --models org/name@<sha>[,...] --out <dir> --env-lock <lock>

For each model it writes, under <out>/<org>__<name>/:
    verdicts/<mode>/*.json   every verdict, as the engine produced it (no root-cause linking: that happens on ingest,
                             against the repo's findings/root_causes.json at that time)
    templates.json           the model's official template, and the declared one if it differs
    declaration.json         what the adapter declared for the default config, with its sources
    run.json                 backend, timings, counts, or the error if the model couldn't run
It refuses to run unless Unsloth took its GPU route; `--allow-cpu` exists only to test the kit on a machine without
a GPU, and those verdicts are recorded with backend "cpu".
"""

from __future__ import annotations

import argparse
import sys

from parity.adapters.unsloth import BACKEND, DEVICE_TYPE, UnslothAdapter  # Unsloth first, as its notebooks do

import json  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from collections import Counter  # noqa: E402
from dataclasses import asdict  # noqa: E402
from pathlib import Path  # noqa: E402

from parity.adapters.base import TrainCfg  # noqa: E402
from parity.checks.common import official_template  # noqa: E402
from parity.checks.tier1 import tier1_checks  # noqa: E402
from parity.engine.runner import evaluate_suite  # noqa: E402
from parity.engine.verdict import write_verdict  # noqa: E402
from parity.fixtures import load_all  # noqa: E402
from parity.release import Release, load_release  # noqa: E402
from parity.serving import SERVING_MODES  # noqa: E402


def model_folder(spec: str) -> str:
    return Release.parse(spec).repo_id.replace("/", "__")


def run_model(spec: str, out: Path, env_lock: str) -> dict:
    started = time.time()
    release = load_release(Release.parse(spec))
    adapter = UnslothAdapter(release, env_lock=env_lock)
    fixtures = load_all()
    verdicts = []
    for mode in SERVING_MODES:
        verdicts += evaluate_suite(tier1_checks(mode), adapter, fixtures)
    for v in verdicts:
        write_verdict(v, out / "verdicts" / (v.serving_mode or "unrecorded"))

    declared = adapter.declare(TrainCfg())
    templates = {"model": official_template(release)}
    if declared.chat_template != templates["model"]:
        templates[declared.template_source] = declared.chat_template
    (out / "templates.json").write_text(json.dumps(templates, indent=1) + "\n", encoding="utf-8")
    (out / "declaration.json").write_text(json.dumps(asdict(declared), indent=1) + "\n", encoding="utf-8")

    counts = {
        mode: dict(sorted(Counter(v.result if v.result != "fail" else f"fail:{v.kind}" for v in verdicts if v.serving_mode == mode).items()))
        for mode in SERVING_MODES
    }  # fmt: skip
    return {
        "model": spec,
        "framework": f"{adapter.name}@{adapter.version}",
        "backend": adapter.backend,
        "env_lock": adapter.env_lock,
        "env_fingerprint": adapter.env_fingerprint,
        "verdicts": len(verdicts),
        "counts": counts,
        "seconds": round(time.time() - started, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", required=True, help="comma-separated org/name@<40-char revision>")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--env-lock", required=True, help="the lockfile this environment was synced from")
    parser.add_argument("--allow-cpu", action="store_true", help="kit testing only: run on Unsloth's CPU fallback")
    args = parser.parse_args()

    specs = [s.strip() for s in args.models.split(",") if s.strip()]
    for spec in specs:
        Release.parse(spec)  # refuses anything but org/name@<40-char sha> (rule 7) before any work starts
    print(f"Unsloth backend: {BACKEND} (unsloth.DEVICE_TYPE={DEVICE_TYPE!r})", flush=True)
    if BACKEND != "gpu" and not args.allow_cpu:
        print("REFUSED: Unsloth took its CPU fallback, so this would not be a GPU run. Check the GPU runtime.")
        return 2

    failed = 0
    for spec in specs:
        out = args.out / model_folder(spec)
        out.mkdir(parents=True, exist_ok=True)
        print(f"== {spec}", flush=True)
        try:
            record = run_model(spec, out, args.env_lock)
            print(f"   done in {record['seconds']}s: {record['counts']}", flush=True)
        except Exception as exc:  # noqa: BLE001 - one model's failure is recorded, and the others still run
            failed += 1
            record = {"model": spec, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
            print(f"   FAILED: {record['error']}", flush=True)
        (out / "run.json").write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
