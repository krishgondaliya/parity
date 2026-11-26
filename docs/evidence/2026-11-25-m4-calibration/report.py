"""Write runs/calibration-2026-11-25.md from the M4 calibration runs, and merge Unsloth's GPU run into them.

Run from the repo root, in the core environment:
    uv run python docs/evidence/2026-11-25-m4-calibration/report.py
    uv run python docs/evidence/2026-11-25-m4-calibration/report.py --unsloth-zip dist/gpu-results/<zip>

TRL and Axolotl run locally (run.py, in their own environments). Unsloth runs on GPU, with the Colab kit and
RUN = "full". `--unsloth-zip` verifies that run's zip (`parity.gpu_results.verify_zip`, plus every verdict: framework
unsloth@2026.11.10, backend gpu, the run's lockfile). It then keeps the zip byte for byte under unsloth/raw/, writes
unsloth/verdicts (linked to findings/root_causes.json), environment.json, status.json, templates.json,
provenance.json and summary.md, and renders the report with all three.

Green (`parity.engine.calibration`): no untriaged fail, no harness-caused inconclusive, every planted fault caught.
Fails linked to a triaged finding or a known difference are allowed and listed.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from parity.gpu_results import Refused, safe_extract, verify_zip

OUT = Path(__file__).parent
REPO = OUT.parents[2]
REPORT = REPO / "runs" / "calibration-2026-11-25.md"
FRAMEWORKS = ("trl", "axolotl", "unsloth")
MODES = ("thinking", "non_thinking")


def merge_unsloth(zip_path: Path) -> dict:
    from parity.engine.verdict import read_verdict

    out = OUT / "unsloth"
    with tempfile.TemporaryDirectory(prefix="parity-ingest-") as tmp:
        root = safe_extract(zip_path, Path(tmp))
        facts = verify_zip(zip_path, root, REPO, mode="full")
        results = root / "results" / "Qwen__Qwen3-0.6B" / "verdicts"
        verdicts = [read_verdict(p) for p in sorted(results.rglob("*.json"))]
        records = {(v.framework, v.backend, v.env_lock, v.env_fingerprint) for v in verdicts}
        if records != {("unsloth@2026.11.10", "gpu", facts["lockfile"], facts["lockfile_sha256"])}:
            raise Refused(f"verdicts don't all record unsloth@2026.11.10, backend gpu and the run's lockfile: {records}")
        facts["verdicts_verified"] = len(verdicts)
        shutil.rmtree(out, ignore_errors=True)
        (out / "raw").mkdir(parents=True)
        shutil.copyfile(zip_path, out / "raw" / zip_path.name)
        shutil.copytree(results, out / "verdicts")
        for name in ("environment.json", "status.json"):
            shutil.copyfile(root / name, out / name)
        shutil.copyfile(root / "results" / "Qwen__Qwen3-0.6B" / "templates.json", out / "templates.json")
    (out / "provenance.json").write_text(json.dumps(facts, indent=1) + "\n", encoding="utf-8")
    return facts


def load(framework: str) -> list:
    from parity.engine.root_causes import link_root_causes
    from parity.engine.verdict import read_verdict, write_verdict

    folder = OUT / framework / "verdicts"
    if not folder.is_dir():
        return []
    verdicts = link_root_causes(read_verdict(p) for p in sorted(folder.rglob("*.json")))
    if framework == "unsloth":  # the run.py of the other two writes linked verdicts and a summary itself
        shutil.rmtree(folder)
        for v in verdicts:
            write_verdict(v, folder / v.serving_mode)
        write_unsloth_summary(verdicts)
    return verdicts


def write_unsloth_summary(verdicts: list) -> None:
    """unsloth/summary.md, as run.py writes the other two: properties judged on the fixtures each mode applies to."""
    from parity.engine.report import render_summary
    from parity.fixtures import load_all
    from parity.references.properties import template_properties_for
    from parity.release import load_release
    from parity.serving import not_applicable_in_mode
    from parity.targets import CALIBRATION

    release, fixtures = load_release(CALIBRATION), load_all()
    templates = json.loads((OUT / "unsloth" / "templates.json").read_text(encoding="utf-8"))
    properties = []
    for mode in MODES:
        applicable = [f for f in fixtures if not_applicable_in_mode(f, mode) is None]
        for source, template in templates.items():
            properties += template_properties_for(release, template, source, applicable, mode)
    facts = json.loads((OUT / "unsloth" / "provenance.json").read_text(encoding="utf-8"))
    note = f"> {unsloth_note(facts['commit'])}\n\n" if unsloth_note(facts["commit"]) else ""
    (OUT / "unsloth" / "summary.md").write_text(note + render_summary(verdicts, properties), encoding="utf-8")


BEFORE_APPLICABILITY_RULE = "b73da059b6141e3f7d1c8c5bd77728f10bd73b04"  # the kit commit of the 2026-11-25 run


def unsloth_note(commit: str) -> str:
    """What a reader needs about a GPU run whose kit predates the serving-mode applicability rule; else nothing."""
    if commit != BEFORE_APPLICABILITY_RULE:
        return ""
    return (
        "**Unsloth's calibration run is its GPU route** (Colab kit, RUN = \"full\"; verified in `provenance.json`). "
        "Non-thinking mode is not configurable for Unsloth (P-0006, the human's decision). The kit was built at "
        "commit `b73da05`, before the serving-mode applicability rule (2026-11-25), and that rule is the only code "
        "change since. So the six `reasoning` × non-thinking cells record `not_configurable`, where today's runner "
        "would record `not_applicable`. Neither is judged, and neither counts toward green. Every judged cell is "
        "unaffected."
    )


def render(statuses: dict, rows: list[str], unsloth_facts: dict | None) -> str:
    from parity.targets import CALIBRATION

    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short=12", "HEAD"], capture_output=True,
                            text=True, check=True).stdout.strip()  # fmt: skip
    done = [f for f in FRAMEWORKS if f in statuses]
    green = len(done) == len(FRAMEWORKS) and all(statuses[f].green for f in done)
    lines = [
        "# Calibration, 2026-11-25 (M4)",
        "",
        f"- **Model:** `{CALIBRATION.spec}`, all 8 fixtures, both serving modes, every planted fault.",
        f"- **Checks:** as of commit `{commit}`, including `t1.special_tokens`' split padding rules and the serving-mode "
        "applicability rule (a fixture whose assistant turn carries reasoning is not applicable in non-thinking mode).",
        "- **Green means:** no untriaged fail, no harness-caused inconclusive, every planted fault caught "
        "(`parity.engine.calibration`). Fails linked to triaged findings or known differences are allowed, and listed.",
        f"- **Overall: {'GREEN' if green else 'NOT GREEN'}.**",
        "",
    ]
    for framework in FRAMEWORKS:
        if framework not in statuses:
            lines += [f"## {framework}: pending", "", "Runs on GPU with the Colab kit (RUN = \"full\"); not merged yet.", ""]
            continue
        s = statuses[framework]
        lines += [f"## {s.framework}: {'green' if s.green else 'NOT GREEN'}", ""]
        if framework == "unsloth" and unsloth_facts:
            lines += [f"GPU run `{unsloth_facts['zip']}` (commit `{unsloth_facts['commit'][:12]}`, {unsloth_facts['gpu']}), "
                      "verified (`unsloth/provenance.json`).", ""]  # fmt: skip
            if unsloth_note(unsloth_facts["commit"]):
                lines += [unsloth_note(unsloth_facts["commit"]), ""]
        lines.append("- Results: " + ", ".join(f"{n} {r}" for r, n in s.results.items()))
        for cause, cells in sorted(s.linked_fails.items()):
            lines.append(f"- Linked fails, {cause}: {len(cells)} ({'; '.join(map(str, cells))})")
        for label, cells in (
            ("Untriaged fails", s.untriaged_fails),
            ("Inconclusive, the framework's (a linked template parity fail explains it)", s.framework_inconclusives),
            ("**Inconclusive, harness-caused**", s.harness_inconclusives),
            ("**Planted faults missed**", s.missed_plants),
        ):
            lines.append(f"- {label}: {len(cells)}")
            lines += [f"  - {c}" for c in cells]
        if s.not_supported:  # D2: blocks green; listed only when present, so earlier reports render unchanged
            lines.append(f"- **Not supported (the framework refuses)**: {len(s.not_supported)}")
            lines += [f"  - {c}" for c in s.not_supported]
        lines.append("")
    lines += [
        "## Framework x mode x check",
        "",
        "| Framework | Mode | Check | Clean result (over 8 fixtures) | Planted faults | Notes |",
        "| --- | --- | --- | --- | --- | --- |",
        *rows,
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    from parity.engine.calibration import attribute, table_rows

    unsloth_facts = None
    try:
        if "--unsloth-zip" in sys.argv:
            unsloth_facts = merge_unsloth(Path(sys.argv[sys.argv.index("--unsloth-zip") + 1]).resolve())
            print("verified:", unsloth_facts)
        elif (OUT / "unsloth" / "provenance.json").is_file():
            unsloth_facts = json.loads((OUT / "unsloth" / "provenance.json").read_text(encoding="utf-8"))
    except Refused as exc:
        print(f"REFUSED: {exc}")
        return 1
    statuses, rows = {}, []
    for framework in FRAMEWORKS:
        verdicts = load(framework)
        if verdicts:
            [statuses[framework]] = attribute(verdicts)
            rows += table_rows(verdicts)
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(render(statuses, rows, unsloth_facts), encoding="utf-8")
    for framework, s in statuses.items():
        print(f"{s.framework}: {'green' if s.green else 'NOT GREEN'}; untriaged {len(s.untriaged_fails)}, "
              f"harness inconclusive {len(s.harness_inconclusives)}, plants missed {len(s.missed_plants)}")  # fmt: skip
    print(f"wrote {REPORT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
