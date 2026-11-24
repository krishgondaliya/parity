import pytest

from parity.fixtures import FIXTURE_IDS, fixture_path, load_all, load_fixture

ROLES = {"system", "user", "assistant", "tool"}


def test_there_are_exactly_the_eight_blueprint_fixtures():
    assert FIXTURE_IDS == (
        "single_turn",
        "multi_turn",
        "system_prompt",
        "tool_call",
        "reasoning",
        "empty_assistant",
        "long_answer",
        "non_ascii",
    )
    assert [f.id for f in load_all()] == list(FIXTURE_IDS)


@pytest.mark.parametrize("fid", FIXTURE_IDS)
def test_fixture_file_has_no_carriage_returns(fid):
    assert b"\r" not in fixture_path(fid).read_bytes()


@pytest.mark.parametrize("fid", FIXTURE_IDS)
def test_fixture_is_a_well_formed_conversation(fid):
    fx = load_fixture(fid)
    roles = [m["role"] for m in fx.messages]
    assert set(roles) <= ROLES
    assert roles[0] in ("system", "user")
    assert "system" not in roles[1:], "a system message may only come first"
    assert fx.assistant_indices, "every fixture needs at least one assistant turn"
    assert all(isinstance(m["content"], str) for m in fx.messages)


def test_unknown_fixture_is_rejected():
    with pytest.raises(KeyError):
        load_fixture("not_a_fixture")


def test_each_load_returns_a_fresh_copy():
    first = load_fixture("single_turn")
    first.messages[0]["content"] = "changed"
    assert load_fixture("single_turn").messages[0]["content"] != "changed"


def test_multi_turn_has_at_least_three_assistant_turns():
    assert len(load_fixture("multi_turn").assistant_indices) >= 3


def test_system_prompt_comes_first():
    assert load_fixture("system_prompt").messages[0]["role"] == "system"


def test_tool_call_has_tools_a_boolean_false_argument_and_a_tool_result():
    fx = load_fixture("tool_call")
    assert fx.tools
    calls = [call for m in fx.messages for call in m.get("tool_calls", [])]
    assert any(value is False for call in calls for value in call["function"]["arguments"].values())
    assert any(m["role"] == "tool" for m in fx.messages)


def test_reasoning_has_two_exchanges_with_reasoning_in_each_assistant_turn():
    fx = load_fixture("reasoning")
    assert len(fx.assistant_indices) == 2
    assert all(fx.messages[i].get("reasoning_content") for i in fx.assistant_indices)


def test_empty_assistant_has_an_empty_turn_that_is_not_the_last_message():
    fx = load_fixture("empty_assistant")
    first = fx.assistant_indices[0]
    assert fx.messages[first]["content"] == ""
    assert first != len(fx.messages) - 1


def test_only_long_answer_sets_max_len():
    assert {f.id for f in load_all() if f.max_len is not None} == {"long_answer"}


def test_non_ascii_covers_accents_cjk_and_astral_emoji():
    text = " ".join(m["content"] for m in load_fixture("non_ascii").messages)
    assert any("À" <= ch <= "ɏ" for ch in text), "Latin accents"
    assert any("一" <= ch <= "鿿" for ch in text), "CJK ideographs"
    assert any(ord(ch) > 0xFFFF for ch in text), "emoji outside the Basic Multilingual Plane"
    assert "‍" in text, "a zero-width-joiner emoji sequence"
