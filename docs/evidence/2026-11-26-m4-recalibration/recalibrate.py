"""After parity.tiny and the D2/D3 engine rules: does M4 calibration on Qwen3 come out the same, cell for cell?

Step 1, run (in each framework's own environment; verdicts go to runs/, which is gitignored):
    envs/trl-1.18.2/.venv/bin/python     docs/evidence/2026-11-26-m4-recalibration/recalibrate.py run trl
    envs/axolotl-0.21.1/.venv/bin/python docs/evidence/2026-11-26-m4-recalibration/recalibrate.py run axolotl
  Unsloth's M4 calibration ran on GPU (Colab), which can't be re-run here. Its CPU backend is compared instead,
  before and after the adapter change, with the GPU kit's own checks step (`docs/gpu-run/run_checks.py --allow-cpu`):
  the "before" run at commit f13a5a8 (the factory not yet in the adapter), the "after" run at the working tree.

Step 2, compare (core environment; writes results.json and summary.md next to this file):
    uv run python docs/evidence/2026-11-26-m4-recalibration/recalibrate.py compare \\
        --unsloth-before <before dir> --unsloth-after <after dir>

The comparison is field by field over every verdict. One difference is expected and allowed:
`declaration.model_build`, the new record of the tiny model the adapter built (parity.tiny). Any other difference,
in any field, is reported.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
M4 = REPO / "docs/evidence/2026-11-25-m4-calibration"
RUNS = REPO / "runs/recalibration-2026-11-26"
MODES = ("thinking", "non_thinking")
ADAPTERS = {"trl": ("parity.adapters.trl", "TRLAdapter"), "axolotl": ("parity.adapters.axolotl", "AxolotlAdapter")}
ALLOWED = ("declaration.model_build", "declaration.other_calls.<n>.model_build")


def run(framework: str) -> None:
    import importlib

    from parity.checks.tier1 import tier1_checks
    from parity.engine.root_causes import link_root_causes
    from parity.engine.runner import evaluate_suite
    from parity.engine.verdict import write_verdict
    from parity.fixtures import load_all
    from parity.release import load_release
    from parity.targets import CALIBRATION

    module, name = ADAPTERS[framework]
    adapter = getattr(importlib.import_module(module), name)(load_release(CALIBRATION))
    fixtures = load_all()
    verdicts = []
    for mode in MODES:
        verdicts += evaluate_suite(tier1_checks(mode), adapter, fixtures)
    out = RUNS / framework
    for v in link_root_causes(verdicts):  # as M4's run.py does
        write_verdict(v, out / "verdicts" / v.serving_mode)
    print(f"{framework}: {len(verdicts)} verdicts in {out.relative_to(REPO)}")


def flatten(value, prefix: str = "") -> dict[str, object]:
    """Leaf paths, into lists too, so a declaration nested in a list (declaration.other_calls.0...) is compared field
    by field like the top-level one."""
    if isinstance(value, dict):
        out: dict[str, object] = {}
        for key, item in value.items():
            out.update(flatten(item, f"{prefix}.{key}" if prefix else key))
        return out
    if isinstance(value, list) and any(isinstance(item, (dict, list)) for item in value):
        out = {f"{prefix}.#len": len(value)}
        for i, item in enumerate(value):
            out.update(flatten(item, f"{prefix}.{i}"))
        return out
    return {prefix: value}


def allowed(key: str) -> bool:
    """`model_build` in the verdict's declaration, or in a declaration nested inside it (other_calls)."""
    parts = key.split(".")
    return parts[0] == "declaration" and "model_build" in parts


def compare_dirs(before: Path, after: Path) -> dict:
    """Every verdict file in `before`, matched by relative path in `after`."""
    old = {p.relative_to(before).as_posix(): p for p in sorted(before.rglob("*.json"))}
    new = {p.relative_to(after).as_posix(): p for p in sorted(after.rglob("*.json"))}
    cells = []
    for rel in sorted(set(old) | set(new)):
        if rel not in old or rel not in new:
            cells.append({"verdict": rel, "missing_in": "before" if rel not in old else "after"})
            continue
        a = flatten(json.loads(old[rel].read_text(encoding="utf-8")))
        b = flatten(json.loads(new[rel].read_text(encoding="utf-8")))
        differing = sorted(k for k in set(a) | set(b) if a.get(k, "<absent>") != b.get(k, "<absent>"))
        unexpected = [k for k in differing if not allowed(k)]
        cells.append({
            "verdict": rel,
            "result": b.get("result"),
            "kind": b.get("kind"),
            "self_test": b.get("self_test"),
            "unexpected_differences": {k: {"before": a.get(k, "<absent>"), "after": b.get(k, "<absent>")} for k in unexpected},
            "allowed_differences": [k for k in differing if k not in unexpected],
        })  # fmt: skip
    same = [c for c in cells if "missing_in" not in c and not c["unexpected_differences"]]
    return {"before": str(before.relative_to(REPO)) if before.is_relative_to(REPO) else before.name,
            "after": str(after.relative_to(REPO)) if after.is_relative_to(REPO) else after.name,
            "verdicts": len(cells), "identical_apart_from_allowed": len(same), "cells": cells}  # fmt: skip


def compare(unsloth_before: Path | None, unsloth_after: Path | None) -> None:
    results = {"allowed": list(ALLOWED), "frameworks": {}}
    for framework in ADAPTERS:
        results["frameworks"][framework] = compare_dirs(M4 / framework / "verdicts", RUNS / framework / "verdicts")
    if unsloth_before and unsloth_after:
        folder = "Qwen__Qwen3-0.6B"
        r = compare_dirs(unsloth_before / folder / "verdicts", unsloth_after / folder / "verdicts")
        r["before"] = "run_checks.py --allow-cpu at f13a5a8 (before the adapter change)"
        r["after"] = "run_checks.py --allow-cpu at the working tree (after)"
        results["frameworks"]["unsloth (cpu backend)"] = r
    (HERE / "results.json").write_text(json.dumps(results, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# M4 re-calibration after parity.tiny and the D2/D3 rules: cell for cell",
        "",
        "Generated by `recalibrate.py compare`. Every verdict compared field by field with the earlier run; allowed: "
        + ", ".join(f"`{a}`" for a in ALLOWED) + " (the new record of the tiny model).",
        "",
        "| framework | compared with | verdicts | identical apart from the allowed field | other differences |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, r in results["frameworks"].items():
        other = r["verdicts"] - r["identical_apart_from_allowed"]
        lines.append(f"| {name} | `{r['before']}` | {r['verdicts']} | {r['identical_apart_from_allowed']} | {other} |")
    lines.append("")
    for name, r in results["frameworks"].items():
        for c in r["cells"]:
            if "missing_in" in c:
                lines.append(f"- {name} `{c['verdict']}`: missing in the {c['missing_in']} run")
            for key, diff in c.get("unexpected_differences", {}).items():
                lines.append(f"- {name} `{c['verdict']}` `{key}`: {str(diff['before'])[:120]!r} -> {str(diff['after'])[:120]!r}")
    lines.append("")
    (HERE / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[4:]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["run", "compare"])
    parser.add_argument("framework", nargs="?", choices=sorted(ADAPTERS))
    parser.add_argument("--unsloth-before", type=Path)
    parser.add_argument("--unsloth-after", type=Path)
    args = parser.parse_args()
    if args.step == "run":
        if not args.framework:
            sys.exit("run needs a framework")
        run(args.framework)
    else:
        compare(args.unsloth_before, args.unsloth_after)
