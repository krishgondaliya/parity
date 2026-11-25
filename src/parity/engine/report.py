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

Results come in one table per serving mode (parity.serving). The primary mode
comes first; any other mode is a clearly labeled secondary result: each
framework encoded again with its documented setting for that mode (or
not_configurable), judged against a reference rendered in that mode.

Backends: a framework that records one (Unsloth) is named with it, e.g.
"unsloth@2026.11.10 (gpu)". When a framework has GPU results, its CPU results
go in a separate table under each mode, labeled "CPU backend (secondary)": the
GPU route is what its users run (the human's decision, 2026-11-24).

3. Framework properties: behavior a framework declares that no check grades
   pass/fail but users need to know (tool arguments parsed as JSON or passed
   through), with whether the framework's behavior confirmed it.
4. Fails by root cause: fails are counted by root cause, not by cell
   (parity.engine.root_causes). Each root cause lists the cells it explains;
   each untriaged fail is listed on its own and counts as its own root cause.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable

from parity.engine.root_causes import card_labels, count_fails, harness_error_ids
from parity.engine.summary import Cell, summarize
from parity.engine.verdict import Verdict, framework_label
from parity.references.properties import TemplateProperty

INHERITED_NOTE = {
    "prefix_breaks": "prefix not preserved; the breaks are the template's",
    "marker_disagreement": "the template's generation markers disagree with the reference",
    "prefill_missing_from_history": "spans anchored past a pre-fill missing from history",
}
FLAG_NOTE = {
    "zero_trainable_tokens": "kept a row with zero trainable tokens",
    "declared_actual_mismatch": "declared/actual mismatch (route to /triage)",
    "marker_disagreement": "template markers disagree",
    "blocked_by_template_parity": "blocked by t1.template_parity",
}


PRIMARY_MODE = "thinking"  # the kill test's primary results (docs/DECISIONS.md, 2026-11-24, from P-0001)


def render_summary(
    verdicts: Iterable[Verdict],
    properties: Iterable[TemplateProperty],
    primary_mode: str = PRIMARY_MODE,
    harness_errors: Iterable[str] | None = None,
    labels: dict[str, str] | None = None,
) -> str:
    verdicts = list(verdicts)
    by_id = {p.id: p for p in properties}
    referenced = {p for v in verdicts for p in v.template_properties}
    missing = sorted(referenced - by_id.keys())
    if missing:
        raise ValueError(f"verdicts reference template properties that weren't provided: {missing}")

    ours = frozenset(harness_errors) if harness_errors is not None else harness_error_ids()
    labels = labels if labels is not None else card_labels()
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
                f"**{p.id}** ({p.kind}; template `{p.template_source}`, sha256 `{p.template_sha256[:12]}`, "
                f"{p.serving_mode} mode)",
                "",
                p.summary,
                "",
            ]
            lines += _property_details(p)
            lines.append("")

        model_cells = [c for c in cells if c.model == model]
        modes = sorted({c.serving_mode for c in model_cells}, key=lambda m: (m != primary_mode, m))
        for mode in modes:
            if mode == primary_mode:
                lines += [f"### Results: {mode} mode (primary)", ""]
            else:
                lines += [
                    f"### Secondary results: {mode} mode",
                    "",
                    f"Secondary: each framework encoded again with its documented setting for {mode} mode (or "
                    f"not_configurable when it documents none), judged against a reference rendered in {mode} mode.",
                    "",
                ]
            mode_cells = [c for c in model_cells if c.serving_mode == mode]
            main, cpu_secondary = _split_backends(mode_cells)
            lines += _results_table(main, by_id)
            if cpu_secondary:
                lines += [
                    f"#### CPU backend (secondary): {mode} mode",
                    "",
                    "The same framework on its CPU fallback. Its GPU results above are its primary result.",
                    "",
                ]
                lines += _results_table(cpu_secondary, by_id)
        lines += _framework_properties([v for v in verdicts if v.model == model])
        lines += _root_cause_section([v for v in verdicts if v.model == model], primary_mode, ours, labels)
    return "\n".join(lines)


PROPERTY_LABELS = {
    "tool_arguments": (
        "tool-call arguments given as a string",
        {"parsed_as_json": "parsed as JSON, then rendered like a dict", "passed_through": "rendered verbatim"},
    ),
}


def _split_backends(cells: list[Cell]) -> tuple[list[Cell], list[Cell]]:
    """Main cells, and CPU-backend cells of frameworks that also have GPU cells (shown as a secondary table)."""
    with_gpu = {c.framework.removesuffix(" (gpu)") for c in cells if c.backend == "gpu"}
    secondary = [c for c in cells if c.backend == "cpu" and c.framework.removesuffix(" (cpu)") in with_gpu]
    return [c for c in cells if c not in secondary], secondary


def _framework_properties(verdicts: list[Verdict]) -> list[str]:
    rows = []
    for framework in sorted({framework_label(v) for v in verdicts}):
        mine = [v for v in verdicts if framework_label(v) == framework and v.declaration.get("verified")]
        for field, (label, values) in PROPERTY_LABELS.items():
            declared = {v.declaration.get(field) for v in mine} - {None}
            if not declared:
                continue
            checks = sorted(
                {
                    f"{v.declaration['verified'][field]} ({v.check} on {v.fixture})"
                    for v in mine
                    if v.declaration["verified"].get(field, "unverified") != "unverified"
                }
            )
            shown = ", ".join(values.get(d, d) for d in sorted(declared))
            rows.append(f"| {framework} | {label} | {shown} | {'; '.join(checks) or 'unverified'} |")
    if not rows:
        return []
    return [
        "### Framework properties",
        "",
        "| Framework | Property | Declared | Verified by behavior |",
        "| --- | --- | --- | --- |",
        *rows,
        "",
    ]


def _root_cause_section(
    verdicts: list[Verdict], primary_mode: str, harness_errors: frozenset[str], labels: dict[str, str]
) -> list[str]:
    count = count_fails(verdicts, harness_errors)
    lines = ["### Fails by root cause", ""]
    if not count.cells:
        return lines + ["No fails.", ""]
    by_mode = {
        mode: sum(v.result == "fail" and (v.serving_mode or "unrecorded") == mode for v in verdicts)
        for mode in sorted({v.serving_mode or "unrecorded" for v in verdicts}, key=lambda m: (m != primary_mode, m))
    }
    split = ", ".join(f"{n} {mode}{' (primary)' if mode == primary_mode else ''}" for mode, n in by_mode.items())
    lines += [
        f"{count.cells} fail verdict(s) ({count.bugs} of kind bug, {count.differences} known differences; {split}) "
        f"come from {count.total} framework root cause(s): {len(count.root_causes)} linked, "
        f"{len(count.untriaged)} untriaged (each untriaged fail counts as its own)."
        + (
            f" {sum(count.harness_errors.values())} more come from harness errors (ours), never counted against "
            "the framework."
            if count.harness_errors
            else ""
        ),
        "",
        "| Root cause | Fail verdicts | Where |",
        "| --- | --- | --- |",
    ]
    rows = [(cause, f"{cause} ({labels[cause]})" if cause in labels else cause, n) for cause, n in count.root_causes.items()]
    rows += [(cause, f"{cause} (harness error, ours)", n) for cause, n in count.harness_errors.items()]
    for cause, label, n in rows:
        where = sorted(
            {
                f"{framework_label(v)} {v.check}/{v.fixture}"
                + ("" if v.serving_mode == primary_mode else f" ({v.serving_mode})")
                for v in verdicts
                if v.result == "fail" and v.root_cause == cause
            }
        )
        lines.append(f"| {label} | {n} | {'; '.join(where)} |")
    for item in count.untriaged:
        lines.append(f"| untriaged | 1 | {item} |")
    lines.append("")
    return lines


def _results_table(cells: list[Cell], by_id: dict[str, TemplateProperty]) -> list[str]:
    lines = [
        "| Framework | Check | Result | Ran on | Notes |",
        "| --- | --- | --- | --- | --- |",
    ]
    for c in cells:
        notes = []
        if c.known_differences:
            notes.append("known differences: " + ", ".join(c.known_differences))
        if c.root_causes:
            notes.append("root cause: " + ", ".join(c.root_causes))
        if c.untriaged:
            notes.append(f"{c.untriaged} untriaged fail(s)")
        for prop in c.template_properties:
            note = INHERITED_NOTE.get(by_id[prop].kind, "inherits a template property")
            notes.append(f"{note} (see [{prop}](#{anchor(prop)}))")
        for flag, n in c.flags.items():
            notes.append(f"{FLAG_NOTE.get(flag, flag)} on {n} fixture(s)")
        if c.not_applicable:
            notes.append(f"not applicable on {c.not_applicable} fixture(s)")
        if c.not_configurable:
            notes.append(f"not configurable on {c.not_configurable} fixture(s): " + "; ".join(c.not_configurable_reasons))
        lines.append(f"| {c.framework} | {c.check} | {c.result} | {c.ran_on} of {c.fixtures} | {'; '.join(notes)} |")
    lines.append("")
    return lines


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
        elif p.kind == "prefill_missing_from_history":
            turns = ", ".join(f"message {t['message_index']}" for t in detail["turns"])
            lines.append(f"- `{detail['fixture']}`: {turns} render without the pre-fill")
        else:
            lines.append(f"- `{detail.get('fixture', '?')}`: {json.dumps(detail, ensure_ascii=False)}")
    return lines
