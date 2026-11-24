"""The declared, behavior-verified tool_arguments property, and the token-level history_false_as_python plant."""

import pytest
from stubs import official_stub, trl_template_stub

from parity.checks.tier1 import MultiturnPrefix
from parity.engine.runner import evaluate
from parity.fixtures import load_fixture

CHECK = MultiturnPrefix()


def plant_record(v, name):
    return next(p for p in v.plants if p["plant"] == name)


@pytest.mark.parametrize("kind", ["official", "trl_markers"])
def test_the_token_level_plant_is_caught(request, kind):
    adapter = request.getfixturevalue(kind)
    v = evaluate(CHECK, adapter, load_fixture("tool_call"))
    record = plant_record(v, "history_false_as_python_tokens")
    assert record["caught"], record
    assert v.self_test == "caught"


def test_a_framework_that_passes_arguments_through_is_consistent(qwen3):
    stub = official_stub(qwen3, tool_arguments="passed_through")
    v = evaluate(CHECK, stub, load_fixture("tool_call"))
    assert v.declaration["verified"]["tool_arguments"] == "consistent"
    assert v.evidence["tool_arguments_probe"]["behaves_as"] == "passed_through"
    assert (v.result, v.self_test) == ("pass", "caught")


def test_a_framework_that_parses_arguments_skips_the_input_plant_and_keeps_the_token_plant(qwen3):
    stub = official_stub(qwen3, tool_arguments="parsed_as_json", parses_tool_arguments=True, name="stub-parses")
    fx = load_fixture("tool_call")
    assert {p.name for p in CHECK.plants(qwen3, fx, stub.declare)} == {"history_edit", "history_false_as_python_tokens"}
    v = evaluate(CHECK, stub, fx)
    assert v.declaration["verified"]["tool_arguments"] == "consistent"
    assert v.evidence["tool_arguments_probe"]["behaves_as"] == "parsed_as_json"
    assert (v.result, v.self_test) == ("pass", "caught")


@pytest.mark.parametrize("declared,parses", [("parsed_as_json", False), ("passed_through", True)])
def test_a_contradicted_declaration_is_inconclusive_and_flagged(qwen3, declared, parses):
    stub = official_stub(qwen3, tool_arguments=declared, parses_tool_arguments=parses, name=f"stub-{declared}-{parses}")
    v = CHECK.run(stub, qwen3, load_fixture("tool_call"))
    assert v.declaration["verified"]["tool_arguments"].startswith("contradicted: ")
    assert v.result == "inconclusive" and "declared_actual_mismatch" in v.flags
    assert "tool_arguments" in v.reason


def test_undeclared_means_unverified_and_no_probe(official):
    v = evaluate(CHECK, official, load_fixture("tool_call"))
    assert v.declaration["verified"]["tool_arguments"] == "unverified"
    assert "tool_arguments_probe" not in v.evidence


def test_fixtures_without_tool_arguments_have_no_probe(qwen3):
    stub = official_stub(qwen3, tool_arguments="passed_through")
    v = evaluate(CHECK, stub, load_fixture("multi_turn"))
    assert v.declaration["verified"]["tool_arguments"] == "unverified"
    assert "history_false_as_python_tokens" not in {p["plant"] for p in v.plants}


def test_trl_templates_stub_is_consistent_with_passed_through(qwen3, trl_qwen3_template):
    stub = trl_template_stub(qwen3, trl_qwen3_template, tool_arguments="passed_through")
    v = evaluate(CHECK, stub, load_fixture("tool_call"))
    assert v.declaration["verified"]["tool_arguments"] == "consistent"


def test_the_report_card_shows_the_property(qwen3):
    from parity.checks.common import official_template
    from parity.engine.report import render_summary
    from parity.references.properties import template_properties_for

    stub = official_stub(qwen3, tool_arguments="parsed_as_json", parses_tool_arguments=True, name="stub-parses")
    fixtures = [load_fixture(f) for f in ("tool_call", "multi_turn")]
    verdicts = [evaluate(CHECK, stub, fx) for fx in fixtures]
    text = render_summary(verdicts, template_properties_for(qwen3, official_template(qwen3), "model", fixtures))
    assert "### Framework properties" in text
    assert (
        "| stub-parses@0 | tool-call arguments given as a string | parsed as JSON, then rendered like a dict "
        "| consistent (t1.multiturn_prefix on tool_call) |"
    ) in text
