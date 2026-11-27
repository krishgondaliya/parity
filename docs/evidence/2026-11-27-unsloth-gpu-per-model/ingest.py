"""Ingest the GPU kit's per-model zips (one model each, written the moment it finished), each on its own.

Run from the repo root, in the core environment:
    uv run python docs/evidence/2026-11-27-unsloth-gpu-per-model/ingest.py dist/gpu-results/<per-model zip> [...]

Kit 4b4de273e093's run (2026-11-26) came back in parts: the combined results zip didn't download (Chrome blocks a
page's repeated downloads), so only per-model zips exist. Each is verified before anything from it is kept:
  - `parity.gpu_results.verify_zip` (unpacked into a fresh temporary folder; the manifest; `git archive` of the
    commit rebuilt here; every step; Unsloth's CUDA route; the lockfile and the installed versions);
  - it is a per-model zip (status.json `partial`), its model finished (`model_ok`), and the zip's name names that
    model and that commit;
  - every verdict records unsloth@2026.11.10, backend gpu, the run's lockfile, and one fingerprint.
Then it keeps the zip byte for byte (raw/) with its provenance (provenance/<model folder>.json).

The calibration model (Qwen3) is compared with M4: every verdict, field by field, against the M4 GPU run's verdicts
(the kept zip in docs/evidence/2026-11-25-m4-calibration/unsloth/raw/), with the M4 re-calibration's comparer. As
there, the one allowed difference is `declaration.model_build`, which verdicts record since 2026-11-26. One more
change is known and checked by name: M4's GPU run was at b73da05, before 8a00284 made `reasoning` not applicable in
non-thinking mode (M4 option a), a rule the runner applies before the framework's declaration. So the 6
non-thinking `reasoning` cells may move from not_configurable to not_applicable, in their result, reason and
declaration.error only. Any other difference is reported. The run is also checked against the calibration green
rule (`parity.engine.calibration`), with root causes linked. Writes qwen3-vs-m4.json and summary.md. A target
model's zip is verified and kept; its M5 ingest is a separate step.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path

from parity.engine.calibration import attribute
from parity.engine.root_causes import link_root_causes
from parity.engine.verdict import read_verdict
from parity.gpu_results import Refused, safe_extract, verify_zip
from parity.targets import CALIBRATION

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[2]
M4_RAW = REPO / "docs/evidence/2026-11-25-m4-calibration/unsloth/raw/parity-gpu-results-b73da059b614-20261125T221035Z.zip"
FRAMEWORK = "unsloth@2026.11.10"


def comparer():
    path = REPO / "docs/evidence/2026-11-26-m4-recalibration/recalibrate.py"
    spec = importlib.util.spec_from_file_location("recalibrate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def folder_of(spec: str) -> str:
    return spec.split("@", 1)[0].replace("/", "__")


def verify(zip_path: Path, root: Path) -> dict:
    facts = verify_zip(zip_path, root, REPO, "full")
    partial = facts.get("partial")
    if not partial:
        raise Refused("not a per-model zip (status.json has no `partial`)")
    if not partial.get("model_ok"):
        raise Refused(f"{partial['model']} didn't finish (model_ok is false)")
    folder = folder_of(partial["model"])
    if not zip_path.name.startswith(f"parity-gpu-model-{facts['commit'][:12]}-{folder}-"):
        raise Refused(f"the zip's name doesn't name {folder} and commit {facts['commit'][:12]}")
    verdicts = [read_verdict(p) for p in sorted((root / "results" / folder / "verdicts").rglob("*.json"))]
    records = {(v.framework, v.backend, v.env_lock, v.env_fingerprint) for v in verdicts}
    if len(records) != 1 or next(iter(records))[:3] != (FRAMEWORK, "gpu", facts["lockfile"]):
        raise Refused(f"verdicts don't all record {FRAMEWORK}, backend gpu and the run's lockfile: {records}")
    facts.update(model=partial["model"], models_in_run=partial["models"], verdicts_verified=len(verdicts))
    return facts


def reasoning_now_not_applicable(cell: dict) -> bool:
    """8a00284's rule (after M4's kit commit): a non-thinking `reasoning` cell, not_configurable -> not_applicable."""
    diff = cell.get("unexpected_differences") or {}
    return (
        cell["verdict"].startswith("non_thinking/")
        and cell["verdict"].endswith("__reasoning.json")
        and set(diff) <= {"result", "reason", "declaration.error"}
        and diff.get("result") == {"before": "not_configurable", "after": "not_applicable"}
    )


def compare_with_m4(root: Path, folder: str) -> dict:
    with tempfile.TemporaryDirectory(prefix="parity-m4-") as tmp:
        m4 = safe_extract(M4_RAW, Path(tmp))
        result = comparer().compare_dirs(m4 / "results" / folder / "verdicts", root / "results" / folder / "verdicts")
    result["before"], result["after"] = f"M4 GPU run ({M4_RAW.name})", "this per-model zip"
    return result


def ingest(zip_path: Path) -> dict:
    with tempfile.TemporaryDirectory(prefix="parity-gpu-model-") as tmp:
        root = safe_extract(zip_path, Path(tmp))
        facts = verify(zip_path, root)
        folder = folder_of(facts["model"])
        (OUT / "raw").mkdir(exist_ok=True)
        shutil.copyfile(zip_path, OUT / "raw" / zip_path.name)
        (OUT / "provenance").mkdir(exist_ok=True)
        (OUT / "provenance" / f"{folder}.json").write_text(json.dumps(facts, indent=1) + "\n", encoding="utf-8")
        print(f"verified and kept {zip_path.name}: {facts['verdicts_verified']} verdicts, commit {facts['commit'][:12]}")
        if facts["model"] != CALIBRATION.spec:
            print(f"  {facts['model']} is a target: verified and kept; its M5 ingest is a separate step")
            return facts
        result = compare_with_m4(root, folder)
        verdicts = link_root_causes([read_verdict(p) for p in sorted((root / "results" / folder / "verdicts").rglob("*.json"))])
    [status] = attribute(verdicts)
    green = not (status.untriaged_fails or status.harness_inconclusives or status.missed_plants or status.not_supported)
    result["calibration"] = {
        "green": green,
        "results": status.results,
        "linked_fails": {k: len(v) for k, v in status.linked_fails.items()},
        "untriaged_fails": len(status.untriaged_fails),
        "harness_inconclusives": len(status.harness_inconclusives),
        "missed_plants": len(status.missed_plants),
        "not_supported": len(status.not_supported),
    }
    (OUT / "qwen3-vs-m4.json").write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    different = [c for c in result["cells"] if "missing_in" in c or c["unexpected_differences"]]
    explained = [c for c in different if reasoning_now_not_applicable(c)]
    unexplained = [c for c in different if c not in explained]
    cal = result["calibration"]
    lines = [
        "# Kit 4b4de273e093's per-model zips (2026-11-26 Colab run)",
        "",
        "The combined results zip didn't download, so each per-model zip is verified and kept on its own (`ingest.py`).",
        "",
        f"- **Qwen3 (calibration):** `{zip_path.name}`, {facts['verdicts_verified']} verdicts, commit "
        f"`{facts['commit'][:12]}`, {facts['gpu']} (driver {facts['driver']}), torch {facts['torch']}, Unsloth's CUDA route.",
        f"  - Against M4's GPU run, field by field: {result['identical_apart_from_allowed']} of {result['verdicts']} "
        "verdicts identical apart from `declaration.model_build` (recorded since 2026-11-26).",
        f"  - {len(explained)} differ only as 8a00284 says (after M4's kit commit b73da05): non-thinking `reasoning`, "
        "not_configurable -> not_applicable, in result, reason and declaration.error. The CPU backend's re-calibration "
        "already shows them not_applicable (`docs/evidence/2026-11-26-m4-recalibration/`).",
        f"  - Unexplained differences: {', '.join(c['verdict'] for c in unexplained) if unexplained else 'none'}.",
        f"  - Calibration green rule: {'green' if cal['green'] else 'NOT green'}. Results {cal['results']}; fails "
        f"linked {cal['linked_fails']}, untriaged {cal['untriaged_fails']}; harness inconclusives "
        f"{cal['harness_inconclusives']}; missed plants {cal['missed_plants']}; not_supported {cal['not_supported']}.",
        f"- **Models in the run:** {', '.join(facts['models_in_run'])}. No zip for the others has come back.",
    ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    if unexplained or not cal["green"]:
        raise Refused("Qwen3 doesn't match M4 (see summary.md)")
    return facts


def main() -> int:
    zips = [Path(a) for a in sys.argv[1:]]
    if not zips:
        print(__doc__)
        return 2
    for zip_path in zips:
        try:
            ingest(zip_path)
        except Refused as exc:
            print(f"REFUSED {zip_path.name}: {exc}")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
