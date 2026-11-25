"""Fails link to their root cause, and the report and Gate A count root causes, not cells."""

import json

import pytest

from parity.engine.report import render_summary
from parity.engine.root_causes import (
    Pin,
    RootCause,
    count_fails,
    evidence_sha256,
    link_root_causes,
    load_root_causes,
    pin,
)
from parity.engine.verdict import Verdict

MODEL = "Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca"
AXOLOTL = "axolotl@0.21.1"


def fail(check, fixture="empty_assistant", mode="thinking", evidence=None, framework=AXOLOTL, **kw):
    return Verdict(
        check=check, model=MODEL, framework=framework, fixture=fixture, result="fail",
        kind=kw.pop("kind", "bug"), self_test="caught", serving_mode=mode, plants=[{"plant": "p", "caught": True}],
        evidence=evidence if evidence is not None else {"must_train_unsupervised": [{"index": 14}]}, **kw,
    )  # fmt: skip


def p0002(*verdicts):
    return RootCause(
        id="P-0002", record="findings/P-0002/record.md", status="candidate", framework=AXOLOTL, model=MODEL,
        summary="empty turn's end token", pins=tuple(Pin(**pin(v)) for v in verdicts),
    )  # fmt: skip


def test_one_root_cause_across_checks_and_modes_counts_once():
    cells = [fail(c, mode=m) for c in ("t1.end_of_turn", "t1.special_tokens", "t1.assistant_mask") for m in ("thinking", "non_thinking")]
    linked = link_root_causes(cells, [p0002(*cells)])
    assert {v.root_cause for v in linked} == {"P-0002"}
    count = count_fails(linked)
    assert (count.cells, count.total, count.root_causes, count.untriaged) == (6, 1, {"P-0002": 6}, ())


def test_a_changed_failure_is_untriaged_again():
    """Same cell, different evidence: the pin no longer matches, so it isn't silently absorbed."""
    triaged = fail("t1.end_of_turn")
    changed = fail("t1.end_of_turn", evidence={"must_train_unsupervised": [{"index": 14}, {"index": 37}]})
    [v] = link_root_causes([changed], [p0002(triaged)])
    assert v.root_cause is None
    count = count_fails([v])
    assert (count.total, count.untriaged) == (1, (f"{AXOLOTL} t1.end_of_turn/empty_assistant (thinking)",))


@pytest.mark.parametrize("field,value", [("framework", "axolotl@0.21.2"), ("serving_mode", "non_thinking"), ("fixture", "single_turn")])
def test_pins_are_exact(field, value):
    triaged = fail("t1.end_of_turn")
    other = fail(
        "t1.end_of_turn",
        framework=value if field == "framework" else AXOLOTL,
        mode=value if field == "serving_mode" else "thinking",
        fixture=value if field == "fixture" else "empty_assistant",
    )
    [v] = link_root_causes([other], [p0002(triaged)])
    assert v.root_cause is None


def test_untriaged_fails_count_one_each():
    cells = [fail("t1.end_of_turn"), fail("t1.special_tokens")]
    count = count_fails(link_root_causes(cells, []))
    assert (count.cells, count.total, len(count.untriaged)) == (2, 2, 2)


def test_differences_link_to_the_finding_their_entry_came_from():
    axolotl = fail("t1.assistant_mask", fixture="single_turn", kind="difference", known_difference="KD-0002")
    trl = fail("t1.template_parity", fixture="single_turn", framework="trl@1.18.2", kind="difference", known_difference="KD-0001")
    a, t = link_root_causes([axolotl, trl], [])
    assert (a.root_cause, t.root_cause) == ("P-0001", "KD-0001")


def test_only_bug_fails_are_pinned_and_only_fails_carry_a_root_cause():
    with pytest.raises(ValueError):
        pin(fail("t1.assistant_mask", kind="difference", known_difference="KD-0002"))
    with pytest.raises(ValueError):
        Verdict(check="t1.end_of_turn", model=MODEL, framework=AXOLOTL, fixture="x", result="pass", self_test="caught", root_cause="P-0002")


def test_the_registry_file_needs_full_evidence_hashes(tmp_path):
    path = tmp_path / "root_causes.json"
    entry = {"id": "P-0002", "record": "r", "status": "candidate", "framework": AXOLOTL, "model": MODEL, "summary": "s",
             "pins": [{"check": "t1.end_of_turn", "fixture": "empty_assistant", "serving_mode": "thinking", "evidence_sha256": "abc"}]}  # fmt: skip
    path.write_text(json.dumps({"schema": "parity.root_causes/v0", "root_causes": [entry]}))
    with pytest.raises(ValueError):
        load_root_causes(path)
    entry["pins"][0]["evidence_sha256"] = evidence_sha256(fail("t1.end_of_turn"))
    path.write_text(json.dumps({"schema": "parity.root_causes/v0", "root_causes": [entry]}))
    [cause] = load_root_causes(path)
    assert cause.id == "P-0002"


def test_the_repository_registry_loads():
    for cause in load_root_causes():
        assert cause.record.startswith("findings/") and cause.pins


def test_the_report_counts_root_causes_not_cells():
    cells = [fail(c) for c in ("t1.end_of_turn", "t1.special_tokens")] + [fail("t1.truncation", fixture="tool_call")]
    linked = link_root_causes(cells, [p0002(*cells[:2])])
    text = render_summary(linked, [], harness_errors=())
    assert "3 fail verdict(s) (3 of kind bug, 0 known differences; 3 thinking (primary)) come from 2 framework root cause(s): 1 linked, 1 untriaged" in text
    assert "| P-0002 | 2 | axolotl@0.21.1 t1.end_of_turn/empty_assistant; axolotl@0.21.1 t1.special_tokens/empty_assistant |" in text
    assert "| untriaged | 1 | axolotl@0.21.1 t1.truncation/tool_call (thinking) |" in text


def test_harness_errors_are_shown_but_never_counted_against_the_framework():
    cells = [fail("t1.assistant_mask", mode="non_thinking", framework="trl@1.18.2")]
    ours = RootCause(
        id="P-0003", record="findings/P-0003/record.md", status="harness error", framework="trl@1.18.2", model=MODEL,
        summary="secondary rows never encoded in that mode", pins=tuple(Pin(**pin(v)) for v in cells),
    )  # fmt: skip
    linked = link_root_causes(cells, [ours])
    count = count_fails(linked, {"P-0003"})
    assert (count.total, count.root_causes, count.harness_errors) == (0, {}, {"P-0003": 1})
    text = render_summary(linked, [], harness_errors={"P-0003"})
    assert "come from 0 framework root cause(s)" in text and "1 more come from harness errors (ours)" in text
    assert "| P-0003 (harness error, ours) | 1 |" in text


def test_the_ledger_counts_every_harness_error_in_the_registry():
    """findings/LEDGER.md keeps the running count; every registry entry marked "harness error" is in it."""
    from pathlib import Path

    from parity.engine.root_causes import harness_error_ids

    ledger = (Path(__file__).resolve().parents[1] / "findings" / "LEDGER.md").read_text(encoding="utf-8")
    for cause in harness_error_ids():
        assert cause in ledger


def test_a_card_label_is_shown_next_to_its_root_cause():
    cells = [fail("t1.template_parity", fixture="single_turn")]
    cause = p0002(*cells)
    linked = link_root_causes(cells, [cause])
    text = render_summary(linked, [], harness_errors=(), labels={"P-0002": "CPU backend only; low impact"})
    assert "| P-0002 (CPU backend only; low impact) | 1 |" in text


def test_p_0004_carries_its_card_label():
    from parity.engine.root_causes import card_labels

    assert card_labels()["P-0004"] == "Unsloth CPU backend only; low impact"


# Backends (Unsloth records gpu or cpu): a root cause pinned on one backend never explains a verdict from another.

UNSLOTH = "unsloth@2026.11.10"


def p0004(*verdicts, backend="cpu"):
    return RootCause(
        id="P-0004", record="findings/P-0004/record.md", status="candidate (CPU backend only)", framework=UNSLOTH,
        model=MODEL, summary="CPU backend appends a second <|im_end|>", pins=tuple(Pin(**pin(v)) for v in verdicts),
        card_label="Unsloth CPU backend only; low impact", backend=backend,
    )  # fmt: skip


def test_a_cpu_root_cause_never_explains_the_same_fail_on_gpu():
    evidence = {"first_divergence": 31, "expected": None, "got": "<|im_end|>"}
    on_cpu = fail("t1.template_parity", fixture="single_turn", framework=UNSLOTH, backend="cpu", evidence=evidence)
    on_gpu = fail("t1.template_parity", fixture="single_turn", framework=UNSLOTH, backend="gpu", evidence=evidence)
    cpu, gpu = link_root_causes([on_cpu, on_gpu], [p0004(on_cpu)])
    assert (cpu.root_cause, gpu.root_cause) == ("P-0004", None)
    assert count_fails([gpu]).untriaged == (f"{UNSLOTH} (gpu) t1.template_parity/single_turn (thinking)",)


def test_a_root_cause_with_no_backend_explains_only_verdicts_with_none():
    triaged = fail("t1.end_of_turn")
    on_gpu = fail("t1.end_of_turn", backend="gpu")
    [v] = link_root_causes([on_gpu], [p0002(triaged)])
    assert v.root_cause is None


def test_the_registry_pins_p0004_to_the_cpu_backend():
    [p0004_entry] = [r for r in load_root_causes() if r.id == "P-0004"]
    assert p0004_entry.backend == "cpu"
    assert all(r.backend is None for r in load_root_causes() if not r.framework.startswith("unsloth@"))


def test_a_verdict_backend_must_be_gpu_or_cpu():
    with pytest.raises(ValueError, match="backend"):
        fail("t1.end_of_turn", backend="cuda")


def test_the_report_puts_cpu_results_in_a_secondary_table_once_gpu_results_exist():
    def passing(fixture, backend):
        return Verdict(
            check="t1.end_of_turn", model=MODEL, framework=UNSLOTH, fixture=fixture, result="pass",
            self_test="caught", serving_mode="thinking", plants=[{"plant": "p", "caught": True}], backend=backend,
        )  # fmt: skip

    both = render_summary([passing("single_turn", "gpu"), passing("single_turn", "cpu")], [])
    main, secondary = both.split("#### CPU backend (secondary): thinking mode")
    assert f"| {UNSLOTH} (gpu) | t1.end_of_turn | pass |" in main and "(cpu)" not in main
    assert f"| {UNSLOTH} (cpu) | t1.end_of_turn | pass |" in secondary.split("### Fails by root cause")[0]

    cpu_only = render_summary([passing("single_turn", "cpu")], [])
    assert "CPU backend (secondary)" not in cpu_only and f"| {UNSLOTH} (cpu) | t1.end_of_turn | pass |" in cpu_only
