"""The calibration status (M4): green means no untriaged fail, no harness-caused inconclusive, every plant caught."""

from parity.engine.calibration import attribute, table_rows
from parity.engine.verdict import Verdict

MODEL = "Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca"
CAUGHT = {"self_test": "caught", "plants": [{"plant": "p", "caught": True}]}


def v(check, fixture, result, mode="thinking", framework="axolotl@0.21.1", **kw):
    fields = dict(CAUGHT) if result in ("pass", "fail") else {}
    fields.update(kw)
    return Verdict(check=check, model=MODEL, framework=framework, fixture=fixture, result=result, serving_mode=mode,
                   **fields)  # fmt: skip


def test_passes_and_linked_fails_are_green():
    [s] = attribute([
        v("t1.end_of_turn", "single_turn", "pass"),
        v("t1.end_of_turn", "empty_assistant", "fail", kind="bug", root_cause="P-0002"),
        v("t1.assistant_mask", "single_turn", "fail", kind="difference", known_difference="KD-0002", root_cause="P-0001"),
        v("t1.multiturn_prefix", "single_turn", "not_applicable", reason="one assistant turn"),
    ])  # fmt: skip
    assert s.green and set(s.linked_fails) == {"P-0002", "P-0001"}


def test_an_untriaged_fail_is_not_green():
    [s] = attribute([v("t1.end_of_turn", "empty_assistant", "fail", kind="bug")])
    assert not s.green and [str(c) for c in s.untriaged_fails] == ["thinking t1.end_of_turn/empty_assistant"]


def test_a_reference_refusal_is_harness_caused_and_not_green():
    refused = v("t1.assistant_mask", "reasoning", "inconclusive", mode="non_thinking",
                reason="reference: message 1: generation prompt render is not a prefix of the full render",
                self_test="missed", plants=[{"plant": "zero_mask", "caught": False}])  # fmt: skip
    [s] = attribute([refused])
    assert not s.green and len(s.harness_inconclusives) == 1 and len(s.missed_plants) == 1


def test_a_cell_blocked_by_a_linked_template_parity_fail_is_the_frameworks():
    parity = v("t1.template_parity", "single_turn", "fail", kind="bug", root_cause="P-0004", framework="unsloth@2026.11.10",
               backend="cpu")  # fmt: skip
    blocked = v("t1.special_tokens", "single_turn", "inconclusive", framework="unsloth@2026.11.10", backend="cpu",
                reason="blocked by t1.template_parity: ...", flags=["blocked_by_template_parity"], **CAUGHT)  # fmt: skip
    [s] = attribute([parity, blocked])
    assert s.green and [str(c) for c in s.framework_inconclusives] == ["thinking t1.special_tokens/single_turn: by P-0004"]


def test_a_cell_blocked_by_an_untriaged_template_parity_fail_is_not_explained():
    parity = v("t1.template_parity", "single_turn", "fail", kind="bug")
    blocked = v("t1.special_tokens", "single_turn", "inconclusive", reason="blocked by t1.template_parity: ...",
                flags=["blocked_by_template_parity"], **CAUGHT)  # fmt: skip
    [s] = attribute([parity, blocked])
    assert not s.green and len(s.harness_inconclusives) == 1 and len(s.untriaged_fails) == 1


def test_backends_are_kept_apart_and_the_table_counts_plants():
    gpu = v("t1.end_of_turn", "single_turn", "pass", framework="unsloth@2026.11.10", backend="gpu")
    cpu = v("t1.end_of_turn", "single_turn", "pass", framework="unsloth@2026.11.10", backend="cpu")
    assert [s.framework for s in attribute([gpu, cpu])] == ["unsloth@2026.11.10 (cpu)", "unsloth@2026.11.10 (gpu)"]
    [row] = table_rows([gpu])
    assert row == "| unsloth@2026.11.10 (gpu) | thinking | t1.end_of_turn | 1 pass | 1 of 1 caught |  |"
