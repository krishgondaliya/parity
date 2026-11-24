"""t1.multiturn_prefix on the calibration model, against the test stand-ins.

Qwen3's own template is not prefix-preserving after assistant turns (it gives
only the last assistant turn a think block); TRL's training template is.
"""

from dataclasses import replace

import pytest
from stubs import plant_named, run_on_all_fixtures, with_a_plant_that_does_nothing

from parity.adapters.base import Dropped, TrainCfg
from parity.checks.common import official_reference, official_template
from parity.checks.tier1.multiturn_prefix import MultiturnPrefix
from parity.engine.plants import PlantedAdapter
from parity.engine.runner import evaluate
from parity.engine.summary import summarize
from parity.fixtures import FIXTURE_IDS, load_fixture
from parity.references.properties import boundaries, break_signature, prefix_break_property_id
from parity.references.template import template_sha256

CHECK = MultiturnPrefix()
APPLICABLE = {"multi_turn": [2, 4], "tool_call": [2], "reasoning": [2], "empty_assistant": [2]}


def test_applicability_is_decided_from_the_fixture_alone():
    assert {fid: boundaries(load_fixture(fid)) for fid in FIXTURE_IDS if boundaries(load_fixture(fid))} == APPLICABLE
    for fid in FIXTURE_IDS:
        assert (CHECK.not_applicable(load_fixture(fid)) is None) == (fid in APPLICABLE)


def test_official_template_breaks_are_inherited_not_blamed(official):
    verdicts = run_on_all_fixtures(CHECK, official)
    for fid, v in verdicts.items():
        if fid not in APPLICABLE:
            assert (v.result, v.self_test, v.plants) == ("not_applicable", None, [])
            continue
        assert (v.result, v.self_test) == ("pass", "caught"), (fid, v.evidence, v.plants)
        assert v.conventions["template_breaks_at"] == APPLICABLE[fid]
        assert v.conventions["framework_breaks_at"] == APPLICABLE[fid]
        assert all(b["same_break_as_template"] for b in v.evidence["boundaries"])
    [cell] = summarize(verdicts.values())
    assert (cell.ran_on, cell.not_applicable, cell.result) == (4, 4, "pass")


def test_template_breaks_match_the_references_prefix_breaks(qwen3):
    """Token-level breaks here agree with the text-level prefix_breaks in the reference."""
    template = official_template(qwen3)
    for fid, expected in APPLICABLE.items():
        fx = load_fixture(fid)
        ref = official_reference(qwen3, fx)
        assert [j for j in boundaries(fx) if j in ref.prefix_breaks] == expected
        full = ref.input_ids
        for j in boundaries(fx):
            prefix_ids = qwen3.tokenizer.apply_chat_template(fx.prefix(j).messages, tools=fx.tools, chat_template=template)
            assert bool(break_signature(prefix_ids, full)) == (j in ref.prefix_breaks)


def test_the_qwen3_break_is_the_dropped_empty_think_block(official):
    """At boundary 2 of multi_turn, history drops the empty think block before "Paris.<|im_end|>\\n"."""
    v = evaluate(CHECK, official, load_fixture("multi_turn"))
    first, second = v.evidence["boundaries"]
    think = ["<think>", "\n\n", "</think>", "\n\n"]
    assert first["template_edits"] == {"tokens_from_end": 8, "removed": think, "inserted": []}
    assert first["framework_edits"] == first["template_edits"]
    assert second["template_edits"]["removed"] == think and second["template_edits"]["inserted"] == []


def test_passes_on_trls_prefix_preserving_template(trl_markers):
    for fid, v in run_on_all_fixtures(CHECK, trl_markers).items():
        if fid in APPLICABLE:
            assert (v.result, v.self_test) == ("pass", "caught"), (fid, v.evidence, v.plants)
            assert v.conventions["template_breaks_at"] == [] and v.conventions["framework_breaks_at"] == []


def test_plants_on_each_fixture(qwen3, official):
    for fid in APPLICABLE:
        names = {p.name for p in CHECK.plants(qwen3, load_fixture(fid), official.declare(TrainCfg()))}
        expected = {"history_edit", "history_false_as_python"} if fid == "tool_call" else {"history_edit"}
        assert names == expected, fid


@pytest.mark.parametrize("fid", sorted(APPLICABLE))
def test_a_plant_that_does_nothing_makes_the_verdict_inconclusive(official, fid):
    v = evaluate(with_a_plant_that_does_nothing(CHECK), official, load_fixture(fid))
    assert (v.result, v.self_test) == ("inconclusive", "missed")


@pytest.mark.parametrize("stub", ["official", "trl_markers"])
def test_false_rerendered_as_python_is_caught(request, qwen3, stub):
    adapter = request.getfixturevalue(stub)
    fx = load_fixture("tool_call")
    plant = plant_named(CHECK, qwen3, fx, adapter.declare(TrainCfg()), "history_false_as_python")
    v = CHECK.run(plant.apply(adapter), qwen3, fx)
    assert (v.result, v.kind) == ("fail", "bug")
    [boundary] = v.evidence["boundaries"]
    assert "False" in "".join(boundary["framework_edits"]["inserted"])
    assert "false" in "".join(boundary["framework_edits"]["removed"])


def test_breaking_where_the_template_does_not_fails(qwen3, trl_markers):
    """TRL's template preserves every prefix, so any history re-render is the framework's own break."""
    fx = load_fixture("reasoning")
    plant = plant_named(CHECK, qwen3, fx, trl_markers.declare(TrainCfg()), "history_edit")
    v = CHECK.run(plant.apply(trl_markers), qwen3, fx)
    assert (v.result, v.kind) == ("fail", "bug")
    assert v.conventions["template_breaks_at"] == [] and v.conventions["framework_breaks_at"] == [2]
    assert v.template_properties == []


def test_a_contradicted_template_declaration_is_an_adapter_error(qwen3, official, trl_qwen3_template):
    """Renders exactly Qwen3's own template but declares TRL's."""
    lying = PlantedAdapter(official, "x", on_declare=lambda d: replace(d, chat_template=trl_qwen3_template))
    v = CHECK.run(lying, qwen3, load_fixture("reasoning"))
    assert v.result == "inconclusive" and v.reason.startswith("adapter error")


def test_an_inherited_break_passes_but_never_reads_as_prefix_preserved(qwen3, official):
    v = evaluate(CHECK, official, load_fixture("reasoning"))
    assert v.result == "pass"
    assert v.conventions["prefix_preserved"] is False
    assert v.template_properties == [prefix_break_property_id(qwen3, template_sha256(official_template(qwen3)))]
    assert "NOT preserved" in v.evidence["inherited_from_template"]


def test_preserving_the_prefix_where_the_template_breaks_passes(qwen3, trl_markers):
    """Renders with TRL's prefix-preserving template but declares Qwen3's: never breaking is allowed."""
    lying = PlantedAdapter(trl_markers, "x", on_declare=lambda d: replace(d, chat_template=official_template(qwen3)))
    v = CHECK.run(lying, qwen3, load_fixture("multi_turn"))
    assert v.result == "pass"
    assert v.conventions["template_breaks_at"] == [2, 4] and v.conventions["framework_breaks_at"] == []


def test_a_dropped_prefix_fails(qwen3, official):
    dropper = PlantedAdapter(
        official, "x", after_encode=lambda out, conv, cfg: Dropped("too short") if len(conv.messages) == 2 else out
    )
    v = CHECK.run(dropper, qwen3, load_fixture("reasoning"))
    assert (v.result, v.kind) == ("fail", "bug")


@pytest.mark.parametrize(
    "prefix, full, expected",
    [
        ([1, 2, 3], [1, 2, 3, 4, 5], None),  # a true prefix: no break
        ([1, 9, 2, 3], [1, 2, 3, 4], (3, (9,), ())),  # history drops a token
        ([1, 2, 3], [1, 7, 3, 4], (2, (2,), (7,))),  # history changes a token
        ([1, 2, 3], [1, 5, 2, 3, 4], (2, (), (5,))),  # history gains a token
        ([1, 2, 3, 4], [1, 2, 3, 9, 2, 3, 4], (1, (), (9, 2, 3))),  # the longest matching tail wins
    ],
)
def test_break_signature(prefix, full, expected):
    assert break_signature(prefix, full) == expected
