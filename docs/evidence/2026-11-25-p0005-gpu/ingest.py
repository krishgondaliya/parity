"""Ingest the P-0005-only GPU run (the Colab kit with RUN = "p0005"): verify the zip, keep it, write what it showed.

Run from the repo root, in the core environment:
    uv run python docs/evidence/2026-11-25-p0005-gpu/ingest.py dist/gpu-results/<results zip>

Verified with `parity.gpu_results.verify_zip` (mode p0005-only: manifests, `git archive` of the commit rebuilt here,
every step, Unsloth's CUDA route, the lockfile), plus: the P-0005 folder holds every run the kit's P0005_RUNS lists at
that commit, and each one completed (printed its result line).

Writes here: raw/<zip> (byte for byte), provenance.json, environment.json, status.json, runs.json (P-0005/run.json),
logs/<run>.log (each run's full output), summary.md.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

from parity.gpu_results import Refused, safe_extract, verify_zip

OUT = Path(__file__).parent
REPO = OUT.parents[2]
EXPECTED_RUNS = ("padded", "padding_free_unset", "padded_train_step", "adapter_check")


def ingest(zip_path: Path) -> dict:
    with tempfile.TemporaryDirectory(prefix="parity-ingest-") as tmp:
        root = safe_extract(zip_path, Path(tmp))
        facts = verify_zip(zip_path, root, REPO, mode="p0005-only")
        runs = json.loads((root / "P-0005" / "run.json").read_text(encoding="utf-8"))
        if tuple(runs["runs"]) != EXPECTED_RUNS:
            raise Refused(f"expected the runs {EXPECTED_RUNS}, got {tuple(runs['runs'])}")
        if runs.get("error") or not all(r["completed"] for r in runs["runs"].values()):
            raise Refused("a P-0005 run didn't complete")
        facts["runs_verified"] = len(runs["runs"])
        shutil.rmtree(OUT / "raw", ignore_errors=True)
        shutil.rmtree(OUT / "logs", ignore_errors=True)
        (OUT / "raw").mkdir()
        (OUT / "logs").mkdir()
        shutil.copyfile(zip_path, OUT / "raw" / zip_path.name)
        for name in ("environment.json", "status.json"):
            shutil.copyfile(root / name, OUT / name)
        shutil.copyfile(root / "P-0005" / "run.json", OUT / "runs.json")
        for name in EXPECTED_RUNS:
            shutil.copyfile(root / "P-0005" / f"{name}.log", OUT / "logs" / f"{name}.log")
    (OUT / "provenance.json").write_text(json.dumps(facts, indent=1) + "\n", encoding="utf-8")
    lines = [
        "# P-0005 on Unsloth's GPU route",
        "",
        f"Run `{facts['zip']}`: commit `{facts['commit'][:12]}`, {facts['gpu']} (driver {facts['driver']}), "
        f"torch {facts['torch']}, Unsloth on `{facts['unsloth_device_type']}`, `{facts['lockfile']}`. Verified: "
        f"{facts['files_verified']} files against their manifests, `git archive` of the commit rebuilt here with the "
        "recorded sha256, every step passed, every installed version as locked (`provenance.json`).",
        "",
    ]
    for name, run in runs["runs"].items():
        lines += [f"## {name} (exit {run['exit_code']})", "", run["meaning"], "", "```", *run["lines"], "```", ""]
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return facts


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    try:
        facts = ingest(Path(sys.argv[1]).resolve())
    except Refused as exc:
        print(f"REFUSED: {exc}")
        return 1
    print("verified:", facts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
