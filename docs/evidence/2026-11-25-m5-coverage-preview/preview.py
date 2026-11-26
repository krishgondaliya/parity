"""M5 prep, item 1: what our own side can judge on each Phase 0 target, before any framework runs.

Run from the repo root, in the core environment (no framework is imported):
    uv run python docs/evidence/2026-11-25-m5-coverage-preview/preview.py

Two tables per target, in both serving modes:

1. Reference level (fixture x mode): does the Tier 1 reference compute on the model's own template? If it does, which
   method delimited each assistant turn (prefix diff or structural), whether a pre-fill is missing from history, and
   whether the template's generation markers were cross-checked. If it refuses, the reference's exact reason.
2. Cell level (check x fixture x mode): the six checks, with their planted faults, run against a known-good stand-in
   encoder: the model's own template, labels from our reference (tests/stubs.py, TemplateStub, made mode-aware here).
   Nothing a framework does can reach these cells, so any cell that is not a clean pass here is stopped on our side
   and will be the same for every framework. This is the most any framework can get on that cell.

These are not verdicts and are never published as such: the stand-in is circular on purpose (a correct encoder by
construction). It only shows where the reference refuses, where no planted fault applies, and where a planted fault
goes uncaught.

D3 (the human's decision, 2026-11-25): a model with no documented non-thinking mode gets not_applicable for its whole
non-thinking table, before any framework output. The engine rule isn't built yet; this preview labels those cells
"n/a (D3)" from the model's mode spec, and also shows what today's code does there.

Writes results.json and summary.md next to this file.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "tests"))  # the known-good stand-in encoder lives with the tests

import tokenizers  # noqa: E402
import transformers  # noqa: E402
from stubs import TemplateStub  # noqa: E402

from parity.adapters.base import LABEL_IGNORE, Dropped, Encoded, TrainCfg  # noqa: E402
from parity.checks.common import official_reference, official_template, reference_on  # noqa: E402
from parity.checks.tier1 import tier1_checks  # noqa: E402
from parity.engine.runner import evaluate_suite  # noqa: E402
from parity.fixtures import FIXTURE_IDS, load_fixture  # noqa: E402
from parity.references.template import TemplateReferenceError, Zone, template_token_ids  # noqa: E402
from parity.release import load_release  # noqa: E402
from parity.serving import SERVING_MODES, mode_spec, not_applicable_in_mode, template_kwargs  # noqa: E402
from parity.targets import PHASE0_TARGETS  # noqa: E402

D3 = "n/a (D3): the model documents no non-thinking mode"


class ModeAwareStub(TemplateStub):
    """TemplateStub that renders with the serving mode's chat-template arguments, so it is a correct encoder in that
    mode too (the test stand-in renders with the template's defaults only). Labels follow our reference in the same
    mode. Truncation as TemplateStub's."""

    def __init__(self, release, mode: str):
        super().__init__(release, name="preview-stand-in", labels_mode=mode)
        self.mode = mode

    def encode(self, conv, cfg: TrainCfg) -> Encoded | Dropped:
        tokenizer = self.release.tokenizer
        ids = template_token_ids(tokenizer, conv, self.template, **template_kwargs(self.release, self.mode))
        zones = reference_on(self.release, conv, self.template, self.template_source, self.mode).zones
        labels = [t if z == Zone.MUST_TRAIN else LABEL_IGNORE for t, z in zip(ids, zones)]
        if cfg.max_len is not None and len(ids) > cfg.max_len:
            ids, labels = ids[: cfg.max_len], labels[: cfg.max_len]  # right truncation, as declared
        return Encoded(tuple(ids), tuple(labels), (1,) * len(ids))


def reference_row(release, fixture, mode: str) -> dict:
    """One fixture in one mode: computes (with how), or refuses (with the reference's exact reason)."""
    row: dict = {"fixture": fixture.id, "mode": mode}
    if mode_spec(release).kwargs.get(mode) is None:
        row["status"] = D3
        return row
    reason = not_applicable_in_mode(fixture, mode)
    if reason is not None:
        row["status"] = "n/a (mode rule)"
        row["reason"] = reason
        return row
    try:
        ref = official_reference(release, fixture, mode)
    except TemplateReferenceError as exc:
        row["status"] = "refuses"
        row["reason"] = f"{type(exc).__name__}: {exc}"
        return row
    row["status"] = "computes"
    row["tokens"] = len(ref.input_ids)
    if mode == mode_spec(release).default:
        # In the default mode a framework is told nothing (parity.serving.framework_mode), so it renders with the
        # template's own defaults, while the reference renders with the mode's arguments. They must agree.
        bare = template_token_ids(release.tokenizer, fixture, official_template(release))
        row["default_render_equals_mode_render"] = tuple(bare) == ref.input_ids
    row["marker_check"] = ref.marker_check
    row["spans"] = [
        {"message": s.message_index, "method": s.method, "prefill_missing": s.prefill_missing} for s in ref.spans
    ]
    row["prefix_breaks"] = list(ref.prefix_breaks)
    return row


def classify(v) -> str:
    """What stopped this cell on our side, if anything."""
    if v.result == "not_applicable":
        return "n/a"
    if v.result == "pass" and v.self_test == "caught":
        return "judgeable"
    if v.result == "inconclusive":
        reason = v.reason or ""
        if reason.startswith("reference:"):
            return "inconclusive: reference refuses"
        if reason.startswith("error:"):
            return "inconclusive: harness error"
        if "no planted fault applies" in reason:
            return "inconclusive: no planted fault applies"
        if "not caught" in reason:
            return "inconclusive: planted fault missed"
        return "inconclusive: other"
    return f"UNEXPECTED on a correct encoder: {v.result}/{v.kind}"


def cell_rows(release, mode: str) -> list[dict]:
    fixtures = [load_fixture(fid) for fid in FIXTURE_IDS]
    verdicts = evaluate_suite(tier1_checks(mode), ModeAwareStub(release, mode), fixtures)
    rows = []
    for v in verdicts:
        rows.append(
            {
                "check": v.check,
                "fixture": v.fixture,
                "mode": mode,
                "class": classify(v),
                "result": v.result,
                "kind": v.kind,
                "self_test": v.self_test,
                "reason": v.reason,
                "plants_missed": [p["plant"] for p in v.plants if not p["caught"]],
                "plants_run": len(v.plants),
            }
        )
    return rows


def main() -> None:
    results: dict = {
        "generated_by": "docs/evidence/2026-11-25-m5-coverage-preview/preview.py",
        "versions": {"transformers": transformers.__version__, "tokenizers": tokenizers.__version__},
        "targets": {},
    }
    for target in PHASE0_TARGETS:
        release = load_release(target)
        spec = mode_spec(release)
        entry: dict = {
            "spec": target.spec,
            "default_mode": spec.default,
            "citation_level": spec.citation_level,
            "reference": [],
            "cells": [],
        }
        for mode in SERVING_MODES:
            entry["reference"] += [reference_row(release, load_fixture(fid), mode) for fid in FIXTURE_IDS]
            rows = cell_rows(release, mode)
            if spec.kwargs.get(mode) is None:  # D3: label the whole table, keep what today's code did
                rows = [{**r, "today": r["class"], "class": D3} for r in rows]
            entry["cells"] += rows
        results["targets"][target.repo_id] = entry
        print(target.repo_id, sorted(Counter((c["mode"], c["class"]) for c in entry["cells"]).items()))
    (HERE / "results.json").write_text(json.dumps(results, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (HERE / "summary.md").write_text(render(results), encoding="utf-8")


SHORT = {
    "judgeable": "ok",
    "n/a": "n/a",
    D3: "n/a (D3)",
    "inconclusive: reference refuses": "REF",
    "inconclusive: no planted fault applies": "NOPLANT",
    "inconclusive: planted fault missed": "MISS",
    "inconclusive: harness error": "ERROR",
    "inconclusive: other": "OTHER",
}


def render(results: dict) -> str:
    checks = [c.id for c in tier1_checks()]
    out = [
        "# M5 coverage preview: what our side can judge on each target",
        "",
        "Generated by `preview.py` (core environment, no framework). transformers "
        f"{results['versions']['transformers']}, tokenizers {results['versions']['tokenizers']}.",
        "",
        "Cell level: the six checks and their planted faults on a known-good stand-in encoder (the model's own "
        "template, labels from our reference). Any cell that is not `ok` is stopped on our side, so it will be the "
        "same for every framework. Legend: `ok` = judgeable (stand-in passes, every planted fault caught); `n/a` = "
        "the check's or the mode's applicability rule; `n/a (D3)` = no documented non-thinking mode; `REF` = the "
        "reference refuses; `NOPLANT` = no planted fault applies; `MISS` = a planted fault goes uncaught; `ERROR` = "
        "a harness error. These are not verdicts.",
        "",
    ]
    for label, primary in (("Primary", True), ("Secondary", False)):
        cells = [
            c
            for e in results["targets"].values()
            for c in e["cells"]
            if (c["mode"] == e["default_mode"]) == primary
        ]
        counts = Counter(SHORT.get(c["class"], c["class"]) for c in cells)
        out.append(
            f"- **{label}, all three targets, per framework:** {counts['ok']} of {len(cells)} cells judgeable; "
            + ", ".join(f"{n} {k}" for k, n in sorted(counts.items()) if k != "ok")
            + "."
        )
    out.append("")
    for repo_id, entry in results["targets"].items():
        out += [f"## {entry['spec']}", "", f"Default mode: {entry['default_mode']} ({entry['citation_level']}).", ""]
        for mode in SERVING_MODES:
            label = "primary" if mode == entry["default_mode"] else "secondary"
            cells = [c for c in entry["cells"] if c["mode"] == mode]
            counts = Counter(SHORT.get(c["class"], c["class"]) for c in cells)
            out += [f"### {mode} ({label}): " + ", ".join(f"{n} {k}" for k, n in sorted(counts.items())), ""]
            out += ["| check | " + " | ".join(FIXTURE_IDS) + " |", "| --- |" + " --- |" * len(FIXTURE_IDS)]
            for check in checks:
                row = {c["fixture"]: SHORT.get(c["class"], c["class"]) for c in cells if c["check"] == check}
                out.append(f"| `{check}` | " + " | ".join(row[f] for f in FIXTURE_IDS) + " |")
            out.append("")
            stopped = [c for c in cells if c["class"] not in ("judgeable", "n/a", D3)]
            for c in stopped:
                missed = f"; missed: {', '.join(c['plants_missed'])}" if c["plants_missed"] else ""
                out.append(f"- `{c['check']}` / `{c['fixture']}`: {c['reason']}{missed}")
            if any("today" in c for c in cells):
                today = Counter(SHORT.get(c["today"], c["today"]) for c in cells)
                out.append(f"- Today's code, before the D3 rule exists: {dict(sorted(today.items()))}.")
            if stopped or any("today" in c for c in cells):
                out.append("")
        out += ["### Reference, per fixture", "", "| fixture | mode | status | how / why |", "| --- | --- | --- | --- |"]
        for r in entry["reference"]:
            if r["status"] == "computes":
                how = ", ".join(
                    f"msg {s['message']}: {s['method']}" + (" (pre-fill missing)" if s["prefill_missing"] else "")
                    for s in r["spans"]
                )
                how += f"; markers: {r['marker_check']}"
                if "default_render_equals_mode_render" in r:
                    same = r["default_render_equals_mode_render"]
                    how += "; template defaults render the same tokens" if same else "; TEMPLATE DEFAULTS RENDER DIFFERENTLY"
            else:
                how = r.get("reason", "")
            out.append(f"| `{r['fixture']}` | {r['mode']} | {r['status']} | {how.replace('|', '/')} |")
        out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    main()
