"""Model-level template properties and the Markdown summary."""

import pytest
from stubs import run_on_all_fixtures

from parity.checks.common import official_template
from parity.checks.tier1 import TIER1_CHECKS
from parity.engine.report import anchor, render_summary
from parity.engine.runner import evaluate
from parity.fixtures import FIXTURE_IDS, load_all, load_fixture
from parity.references.properties import prefix_break_property
from parity.targets import CALIBRATION

QWEN3_BREAKS = {"multi_turn": [2, 4], "tool_call": [2], "reasoning": [2], "empty_assistant": [2]}


@pytest.fixture(scope="module")
def qwen3_property(qwen3):
    return prefix_break_property(qwen3, official_template(qwen3), "model", load_all())


def test_qwen3s_own_template_has_the_prefix_break_property(qwen3_property):
    assert qwen3_property.kind == "prefix_breaks"
    assert qwen3_property.model == CALIBRATION.spec
    assert {d["fixture"]: [b["boundary"] for b in d["breaks"]] for d in qwen3_property.details} == QWEN3_BREAKS
    for d in qwen3_property.details:
        for b in d["breaks"]:
            removed = "".join(b["edit"]["removed"])
            assert b["edit"]["inserted"] == []
            if d["fixture"] == "reasoning":  # history also drops the first turn's reasoning text
                assert removed == "<think>\n" + load_fixture("reasoning").messages[1]["reasoning_content"] + "\n</think>\n\n"
            else:
                assert removed == "<think>\n\n</think>\n\n"


def test_trls_template_preserves_every_prefix(qwen3, trl_qwen3_template):
    assert prefix_break_property(qwen3, trl_qwen3_template, "trl", load_all()) is None


def test_inheriting_verdicts_reference_the_property(official, qwen3_property):
    verdicts = run_on_all_fixtures(TIER1_CHECKS[-1], official)
    assert {fid for fid, v in verdicts.items() if v.template_properties} == set(QWEN3_BREAKS)
    assert {p for v in verdicts.values() for p in v.template_properties} == {qwen3_property.id}


def test_report_shows_each_property_once_per_model_and_links_it(official, qwen3_property):
    verdicts = [evaluate(c, official, load_fixture(fid)) for c in TIER1_CHECKS for fid in FIXTURE_IDS]
    text = render_summary(verdicts, [qwen3_property])
    assert text.count("### Template properties") == 1
    assert text.count(f'<a id="{anchor(qwen3_property.id)}"></a>') == 1
    prefix_row = next(line for line in text.splitlines() if "| t1.multiturn_prefix |" in line)
    assert "| pass | 4 of 8 |" in prefix_row
    assert "prefix not preserved" in prefix_row and f"(#{anchor(qwen3_property.id)})" in prefix_row
    assert "not applicable on 4 fixture(s)" in prefix_row
    assert '- `multi_turn`, boundary 4: removes `"<think>\\n\\n</think>\\n\\n"`, inserts nothing' in text


def test_report_refuses_a_verdict_that_references_a_missing_property(official):
    verdicts = [evaluate(TIER1_CHECKS[-1], official, load_fixture("reasoning"))]
    with pytest.raises(ValueError, match="weren't provided"):
        render_summary(verdicts, [])
