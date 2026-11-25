"""Ingest the Unsloth GPU run's results zip (the Colab kit, docs/gpu-run/): verify it, keep it, and render Unsloth's
results with its GPU route as the primary table and the CPU run (docs/evidence/2026-11-24-m3-unsloth/) as the labeled
secondary "CPU backend" table.

Run from the repo root, in the core environment:
    uv run python docs/evidence/2026-11-25-m3-unsloth-gpu/ingest.py dist/gpu-results/<results zip>
    uv run python docs/evidence/2026-11-25-m3-unsloth-gpu/ingest.py --relink   (re-link and re-render, no zip needed)

Verified before anything is kept (the zip is untrusted data: it is unpacked into a fresh temporary folder, and entry
names that are absolute, climb out with "..", or are links are refused):
- every file against the zip's MANIFEST.json (and P-0005/MANIFEST.json, when the optional repro ran);
- the commit exists here, and `git archive --format=zip <commit>` rebuilt here has the sha256 Colab recorded for the
  uploaded code, so the code that ran is exactly that commit;
- the run finished, Unsloth took its CUDA route (the smoke step), the lockfile's sha256 is the committed one at that
  commit, and every installed version was as locked;
- every verdict: framework unsloth@2026.11.10, backend gpu, that lockfile and its fingerprint.

Writes here: raw/<zip> (byte for byte), provenance.json (what was verified), environment.json, status.json,
run.json, declaration.json, templates.json, verdicts/<mode>/*.json (linked to findings/root_causes.json), summary.md,
compare.md (against predictions.md, both tables), results.json, untriaged.json.

The verdicts as run are always read from the raw zip.
- compare.md judges the predictions against them.
- The non-thinking cells are then re-derived under the human's 2026-11-25 decision: Unsloth's non-default serving
  modes are not configurable (`parity.adapters.unsloth_docs`; P-0006). `as_not_configurable` gives exactly what the
  runner now gives, which a test checks against the real adapter.
- verdicts/, summary.md, results.json and untriaged.json show the re-derived cells. Each keeps its as-run result in
  its evidence.
"""

from __future__ import annotations

import hashlib
import json
import posixpath
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

OUT = Path(__file__).parent
REPO = OUT.parents[2]
CPU_RUN = REPO / "docs" / "evidence" / "2026-11-24-m3-unsloth"
MODES = ("thinking", "non_thinking")
FRAMEWORK = "unsloth@2026.11.10"
MODEL_FOLDER = "Qwen__Qwen3-0.6B"
KEPT = ("environment.json", "status.json")
KEPT_PER_MODEL = ("run.json", "declaration.json", "templates.json")
CELL = {"pass": "pass", "fail": "fail", "n/a": "not_applicable", "inconclusive": "inconclusive"}


class Refused(RuntimeError):
    pass


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git(*args: str) -> bytes:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, check=True).stdout


def safe_extract(zip_path: Path, dest: Path) -> Path:
    with zipfile.ZipFile(zip_path) as zf:
        tops = set()
        for info in zf.infolist():
            name = info.filename
            if name.startswith("/") or "\\" in name or posixpath.normpath(name).startswith(".."):
                raise Refused(f"unsafe entry name {name!r}")
            if stat.S_ISLNK(info.external_attr >> 16):
                raise Refused(f"link entry {name!r}")
            tops.add(name.split("/", 1)[0])
        if len(tops) != 1:
            raise Refused(f"expected one top folder, got {sorted(tops)}")
        zf.extractall(dest)
    return dest / tops.pop()


def verify(zip_path: Path, root: Path) -> dict:
    """Every check from the module docstring. Returns what was verified; raises Refused on the first failure."""
    facts: dict = {"zip": zip_path.name, "zip_sha256": sha256(zip_path.read_bytes())}
    manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
    listed = dict(manifest["files"])
    if (root / "P-0005" / "MANIFEST.json").is_file():
        extra = json.loads((root / "P-0005" / "MANIFEST.json").read_text(encoding="utf-8"))["files"]
        listed.update({f"P-0005/{k}": v for k, v in extra.items()})
    present = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()} - {"MANIFEST.json", "P-0005/MANIFEST.json"}
    if present != set(listed):
        raise Refused(f"files and manifests differ: {sorted(present ^ set(listed))}")
    bad = [name for name, digest in listed.items() if sha256((root / name).read_bytes()) != digest]
    if bad:
        raise Refused(f"sha256 mismatch: {bad}")
    facts["files_verified"] = len(listed)

    commit = manifest["commit"]
    environment = json.loads((root / "environment.json").read_text(encoding="utf-8"))
    if environment["archive"]["commit"] != commit:
        raise Refused("MANIFEST.json and environment.json name different commits")
    git("cat-file", "-e", f"{commit}^{{commit}}")
    rebuilt = git("archive", "--format=zip", commit)
    if not sha256(rebuilt) == manifest["archive_sha256"] == environment["archive"]["sha256"]:
        raise Refused("git archive of the commit doesn't match the sha256 recorded for the uploaded code")
    facts.update(commit=commit, archive_sha256=manifest["archive_sha256"], archive_rebuilt_here=True)

    status = json.loads((root / "status.json").read_text(encoding="utf-8"))
    if not status["ok"]:
        raise Refused("the run didn't finish (status.json ok is false)")
    smoke = environment["platform"].get("torch", {})
    if environment["backend"] != "gpu" or smoke.get("unsloth_device_type") != "cuda":
        raise Refused("the smoke step didn't confirm Unsloth's CUDA route")
    lock = environment["lockfile"]
    if sha256(git("show", f"{commit}:{lock}")) != environment["lockfile_sha256"]:
        raise Refused(f"{lock} at {commit[:12]} isn't the lockfile the run recorded")
    if environment["lock_mismatches"]:
        raise Refused(f"installed versions differ from the lockfile: {environment['lock_mismatches']}")
    facts.update(lockfile=lock, lockfile_sha256=environment["lockfile_sha256"], gpu=environment["platform"]["gpu"],
                 driver=environment["platform"]["driver"], torch=smoke["torch"], unsloth_device_type="cuda")  # fmt: skip

    from parity.engine.verdict import read_verdict

    verdicts = [read_verdict(p) for p in sorted((root / "results" / MODEL_FOLDER / "verdicts").rglob("*.json"))]
    records = {(v.framework, v.backend, v.env_lock, v.env_fingerprint) for v in verdicts}
    if records != {(FRAMEWORK, "gpu", lock, environment["lockfile_sha256"])}:
        raise Refused(f"verdicts don't all record {FRAMEWORK}, backend gpu and the run's lockfile: {records}")
    facts["verdicts_verified"] = len(verdicts)
    return facts


def ingest(zip_path: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="parity-ingest-") as tmp:
        root = safe_extract(zip_path, Path(tmp))
        facts = verify(zip_path, root)
        raw = OUT / "raw"
        shutil.rmtree(raw, ignore_errors=True)
        raw.mkdir()
        shutil.copyfile(zip_path, raw / zip_path.name)
        for name in KEPT:
            shutil.copyfile(root / name, OUT / name)
        for name in KEPT_PER_MODEL:
            shutil.copyfile(root / "results" / MODEL_FOLDER / name, OUT / name)
        if (root / "P-0005" / "run.json").is_file():
            facts["p0005_optional_run"] = json.loads((root / "P-0005" / "run.json").read_text(encoding="utf-8"))
    (OUT / "provenance.json").write_text(json.dumps(facts, indent=1) + "\n", encoding="utf-8")
    print("verified:", {k: v for k, v in facts.items() if k != "p0005_optional_run"})


def predictions() -> dict[tuple[str, str, str], str]:
    """(mode, check, fixture) -> predicted result, from both tables of predictions.md."""
    out, mode, fixtures = {}, None, []
    for line in (OUT / "predictions.md").read_text(encoding="utf-8").splitlines():
        if line.startswith("## Predicted cells: "):
            mode = line.removeprefix("## Predicted cells: ").split()[0]
        elif line.startswith("| Check |"):
            fixtures = [c.strip() for c in line.strip("|").split("|")[1:]]
        elif line.startswith("| t1.") and mode:
            cells = [c.strip().strip("*") for c in line.strip("|").split("|")]
            for fixture, value in zip(fixtures, cells[1:]):
                out[(mode, cells[0], fixture)] = CELL[value]
    return out


def as_run_verdicts() -> list:
    """The GPU verdicts exactly as the run wrote them, from the kept zip."""
    from parity.engine.verdict import read_verdict

    zip_path = next((OUT / "raw").glob("parity-gpu-results-*.zip"))
    with tempfile.TemporaryDirectory(prefix="parity-ingest-") as tmp:
        root = safe_extract(zip_path, Path(tmp))
        return [read_verdict(p) for p in sorted((root / "results" / MODEL_FOLDER / "verdicts").rglob("*.json"))]


def under_mode_decision(verdicts: list) -> list:
    """Unsloth's non-thinking cells are not_configurable (the human's decision, 2026-11-25; P-0006)."""
    from parity.adapters.unsloth_docs import DECISION, NOT_CONFIGURABLE
    from parity.engine.runner import as_not_configurable

    return [as_not_configurable(v, NOT_CONFIGURABLE, DECISION) if v.serving_mode == "non_thinking" else v
            for v in verdicts]  # fmt: skip


def render() -> None:
    from parity.engine.report import render_summary
    from parity.engine.root_causes import count_fails, link_root_causes, pin
    from parity.engine.verdict import read_verdict, write_verdict
    from parity.fixtures import load_all
    from parity.references.properties import template_properties_for
    from parity.release import load_release
    from parity.targets import CALIBRATION

    as_run = link_root_causes(as_run_verdicts())
    gpu = link_root_causes(under_mode_decision(as_run))
    cpu = link_root_causes(under_mode_decision([read_verdict(p) for p in sorted((CPU_RUN / "verdicts").rglob("*.json"))]))
    if {v.backend for v in gpu} != {"gpu"} or {v.backend for v in cpu} != {"cpu"}:
        raise Refused("the GPU folder must hold only gpu verdicts, and the CPU run only cpu ones")
    shutil.rmtree(OUT / "verdicts", ignore_errors=True)
    for v in gpu:
        write_verdict(v, OUT / "verdicts" / v.serving_mode)

    release, fixtures = load_release(CALIBRATION), load_all()
    templates = json.loads((OUT / "templates.json").read_text(encoding="utf-8"))
    templates.update(json.loads((CPU_RUN / "templates.json").read_text(encoding="utf-8")))
    properties = []
    for mode in MODES:
        for source, template in templates.items():
            properties += template_properties_for(release, template, source, fixtures, mode)
    header = (
        "> **Unsloth's primary result is its GPU route** (Tesla T4, `envs/unsloth-2026.11.10/uv.lock`; verified in "
        "`provenance.json`). Its CPU fallback (`docs/evidence/2026-11-24-m3-unsloth/`) is the labeled secondary "
        "\"CPU backend\" table under each mode. **Non-thinking mode: not configurable.** Unsloth documents "
        "non-thinking training via the /no_think soft switch; hard-switch training is not configurable (the human's "
        "decision, 2026-11-25; P-0006). Those cells were run, and re-derived; their as-run results are in "
        "`compare.md` and the raw zip. `t1.special_tokens` is "
        "inconclusive everywhere because its padding part saw 537 tokens, not pad_to=545. That is consistent with "
        "P-0005, but not attributed to it until the adapter's pad_to mapping is checked on GPU (P-0005's record).\n\n"
    )
    (OUT / "summary.md").write_text(header + render_summary(gpu + cpu, properties), encoding="utf-8")

    predicted = predictions()
    rows, misses = [], []
    for v in as_run:
        actual = v.result if v.result != "fail" else ("fail" if v.kind == "bug" else "difference")
        expected = predicted[(v.serving_mode, v.check, v.fixture)]
        rows.append(f"| {v.serving_mode} | {v.check} | {v.fixture} | {expected} | {actual} | "
                    f"{'yes' if actual == expected else '**no**'} | {v.self_test} |")  # fmt: skip
        if actual != expected:
            misses.append(v)
    score = {mode: sum(1 for v in as_run if v.serving_mode == mode and v not in misses) for mode in MODES}
    totals = {mode: sum(1 for v in as_run if v.serving_mode == mode) for mode in MODES}
    compare = [
        "# Unsloth on Qwen3, GPU route: predicted (docs alone) vs actual",
        "",
        f"{sum(score.values())} of {sum(totals.values())} cells as predicted: thinking (primary) "
        f"{score['thinking']} of {totals['thinking']}, non-thinking (secondary, exploratory) "
        f"{score['non_thinking']} of {totals['non_thinking']}.",
        "",
        "Every miss is a `t1.special_tokens` cell: one of the cells predictions.md named in advance, where the docs "
        "predict pass and P-0005's code reading predicts an inconclusive padding part. See P-0005's record.",
        "",
        "These are the verdicts as run. Since the human's 2026-11-25 decision (P-0006), the summary shows the "
        "non-thinking cells as not_configurable.",
        "",
        "| Mode | Check | Fixture | Predicted | Actual | Match | Self-test |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        *rows,
        "",
    ]
    (OUT / "compare.md").write_text("\n".join(compare), encoding="utf-8")

    table = {
        f"{v.serving_mode} / {v.check} / {v.fixture}": {
            "result": v.result, "kind": v.kind, "root_cause": v.root_cause, "self_test": v.self_test,
            "flags": v.flags, "reason": v.reason,
        }
        for v in gpu
    }  # fmt: skip
    (OUT / "results.json").write_text(json.dumps(table, indent=1) + "\n", encoding="utf-8")
    untriaged = [{"verdict": f"{v.serving_mode}/{v.check}__{v.fixture}", "pin": pin(v), "evidence": v.evidence}
                 for v in gpu if v.result == "fail" and v.kind == "bug" and not v.root_cause]  # fmt: skip
    (OUT / "untriaged.json").write_text(json.dumps(untriaged, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    for mode in MODES:
        print(mode, sorted(Counter((v.result, v.kind) for v in gpu if v.serving_mode == mode).items(), key=str))
    print(f"{sum(score.values())} of {sum(totals.values())} cells as predicted; misses: "
          f"{sorted(Counter((v.serving_mode, v.check) for v in misses).items())}")  # fmt: skip
    count = count_fails(gpu)
    print(f"GPU fail verdicts: {count.cells}; root causes {count.root_causes}; untriaged {len(count.untriaged)}")
    print("GPU non-thinking:", dict(Counter(v.result for v in gpu if v.serving_mode == "non_thinking")))


def main() -> int:
    try:
        if "--relink" not in sys.argv:
            if len(sys.argv) != 2:
                print(__doc__)
                return 2
            ingest(Path(sys.argv[1]).resolve())
        render()
    except Refused as exc:
        print(f"REFUSED: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
