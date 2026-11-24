"""A Markdown summary of verdicts, grouped by model.

Phase 0's `runs/phase0/summary.md` and the Phase 1 report card build on this.
Each model gets one section, in this order:

1. Template properties: facts about the chat templates its frameworks use,
   judged before any framework (for example Qwen3's prefix breaks). Shown once
   per model, however many verdicts depend on them.
2. Results: one row per (framework, check) cell, with how many fixtures the
   check actually ran on. A cell whose verdicts inherit a template property
   says so and links to it, so a pass that inherits prefix breaks never reads
   as "prefix preserved".
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable

from parity.engine.summary import summarize
from parity.engine.verdict import Verdict
from parity.references.properties import TemplateProperty

INHERITED_NOTE = {
    "prefix_breaks": "prefix not preserved; the breaks are the template's",
    "marker_disagreement": "the template's generation markers disagree with the reference",
}
FLAG_NOTE = {
    "zero_trainable_tokens": "kept a row with zero trainable tokens",
    "declared_actual_mismatch": "declared/actual mismatch (route to /triage)",
    "marker_disagreement": "template markers disagree",
    "blocked_by_template_parity": "blocked by t1.template_parity",
}


def render_summary(verdicts: Iterable[Verdict], properties: Iterable[TemplateProperty]) -> str:
    verdicts = list(verdicts)
    by_id = {p.id: p for p in properties}
    referenced = {p for v in verdicts for p in v.template_properties}
    missing = sorted(referenced - by_id.keys())
    if missing:
        raise ValueError(f"verdicts reference template properties that weren't provided: {missing}")

    cells = summarize(verdicts)
    lines = ["# Tier 1 summary", ""]
    for model in sorted({c.model for c in cells}):
        lines += [f"## {model}", "", "### Template properties", ""]
        model_properties = sorted((p for p in by_id.values() if p.model == model), key=lambda p: p.id)
        if not model_properties:
            lines += ["None found for the templates checked.", ""]
        for p in model_properties:
            lines += [
                f'<a id="{anchor(p.id)}"></a>',
                f"**{p.id}** ({p.kind}; template `{p.template_source}`, sha256 `{p.template_sha256[:12]}`)",
                "",
                p.summary,
                "",
            ]
            lines += _property_details(p)
            lines.append("")

        lines += [
            "### Results",
            "",
            "| Framework | Check | Result | Ran on | Notes |",
            "| --- | --- | --- | --- | --- |",
        ]
        for c in (c for c in cells if c.model == model):
            notes = []
            if c.known_differences:
                notes.append("known differences: " + ", ".join(c.known_differences))
            for prop in c.template_properties:
                note = INHERITED_NOTE.get(by_id[prop].kind, "inherits a template property")
                notes.append(f"{note} (see [{prop}](#{anchor(prop)}))")
            for flag, n in c.flags.items():
                notes.append(f"{FLAG_NOTE.get(flag, flag)} on {n} fixture(s)")
            if c.not_applicable:
                notes.append(f"not applicable on {c.not_applicable} fixture(s)")
            lines.append(
                f"| {c.framework} | {c.check} | {c.result} | {c.ran_on} of {c.fixtures} | {'; '.join(notes)} |"
            )
        lines.append("")
    return "\n".join(lines)


def anchor(property_id: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", property_id.lower()).strip("-")


def _shown(tokens: list[str]) -> str:
    return "nothing" if not tokens else f"`{json.dumps(''.join(tokens), ensure_ascii=False)}`"


def _property_details(p: TemplateProperty) -> list[str]:
    lines = []
    for detail in p.details:
        if p.kind == "prefix_breaks":
            for b in detail["breaks"]:
                edit = b["edit"]
                lines.append(
                    f"- `{detail['fixture']}`, boundary {b['boundary']}: removes {_shown(edit['removed'])}, "
                    f"inserts {_shown(edit['inserted'])}"
                )
        elif p.kind == "marker_disagreement":
            first = detail["positions"][0]
            lines.append(
                f"- `{detail['fixture']}`: {detail['count']} token(s) disagree; first at {first['index']} "
                f"({json.dumps(first['token'], ensure_ascii=False)}, zone {first['zone']}, marked {first['marked']})"
            )
        else:
            lines.append(f"- `{detail.get('fixture', '?')}`: {json.dumps(detail, ensure_ascii=False)}")
    return lines
