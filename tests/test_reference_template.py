"""Tier 1 reference on the calibration model: Qwen3-0.6B with its own template.

Three kinds of evidence:
1. The tokens are exactly what apply_chat_template returns.
2. Golden spans, written by hand from reading Qwen3's template.
3. A cross-check against TRL's Qwen3 training template, whose
   {% generation %} markers give an independent assistant mask.
"""

import pytest
from stubs import TRL_QWEN3_TRAINING_SHA256, TRL_SOURCE, markers_moved_onto_the_header

from parity.fixtures import FIXTURE_IDS, load_fixture
from parity.references.template import (
    MarkerDisagreementError,
    TemplateReferenceError,
    Zone,
    _token_boundary,
    assistant_mask_from_markers,
    assistant_spans_by_prefix_diff,
    has_generation_markers,
    prefix_breaks,
    render_reference,
)

THINK_EMPTY = "<think>\n\n</think>\n\n"


@pytest.fixture(scope="module")
def own(qwen3):
    return {fid: render_reference(qwen3, load_fixture(fid)) for fid in FIXTURE_IDS}


def _decode(tokenizer, ids):
    return tokenizer.decode(list(ids), skip_special_tokens=False, clean_up_tokenization_spaces=False)


def _span_texts(qwen3, ref):
    return [_decode(qwen3.tokenizer, ref.input_ids[s.token_start : s.token_end]) for s in ref.spans]


def _mask_through_message_end(length, spans):
    mask = [0] * length
    for s in spans:
        mask[s.token_start : s.token_message_end] = [1] * (s.token_message_end - s.token_start)
    return mask


# 1. Tokens


@pytest.mark.parametrize("fid", FIXTURE_IDS)
def test_tokens_are_exactly_apply_chat_template(qwen3, own, fid):
    fx = load_fixture(fid)
    tok = qwen3.tokenizer
    direct = tok.apply_chat_template(fx.messages, tools=fx.tools, tokenize=True, return_dict=True)
    assert list(own[fid].input_ids) == list(direct["input_ids"])
    assert own[fid].text == tok.apply_chat_template(fx.messages, tools=fx.tools, tokenize=False)


def test_qwen3_template_has_no_markers_so_nothing_is_cross_checked(qwen3, own):
    assert not has_generation_markers(qwen3.tokenizer.chat_template)
    assert {r.mask_method for r in own.values()} == {"prefix_diff"}
    assert {r.marker_check for r in own.values()} == {"no_markers"}


@pytest.mark.parametrize("fid", FIXTURE_IDS)
def test_zones_match_the_spans(qwen3, own, fid):
    """must-train is the supervised span; neutral is only the newline after each <|im_end|>."""
    ref = own[fid]
    assert len(ref.zones) == len(ref.input_ids)
    assert [i for i, z in enumerate(ref.zones) if z == Zone.MUST_TRAIN] == [
        i for i, m in enumerate(ref.assistant_mask) if m
    ]
    neutral = [i for i, z in enumerate(ref.zones) if z == Zone.NEUTRAL]
    assert neutral == [i for s in ref.spans for i in range(s.token_end, s.token_message_end)]
    assert [_decode(qwen3.tokenizer, [ref.input_ids[i]]) for i in neutral] == ["\n"] * len(ref.spans)


@pytest.mark.parametrize("fid", FIXTURE_IDS)
def test_mask_shape_and_one_span_per_assistant_turn(own, fid):
    ref = own[fid]
    assert len(ref.assistant_mask) == len(ref.input_ids)
    assert sum(ref.assistant_mask) > 0
    assert [s.message_index for s in ref.spans] == load_fixture(fid).assistant_indices
    assert sum(ref.assistant_mask) == sum(s.token_end - s.token_start for s in ref.spans)


@pytest.mark.parametrize("fid", FIXTURE_IDS)
def test_every_span_ends_on_the_turn_closing_token(qwen3, own, fid):
    ref = own[fid]
    im_end = qwen3.tokenizer.convert_tokens_to_ids("<|im_end|>")
    for s in ref.spans:
        assert ref.input_ids[s.token_end - 1] == im_end
        # The newline after <|im_end|> belongs to the message but is never generated.
        assert _decode(qwen3.tokenizer, ref.input_ids[s.token_end : s.token_message_end]) == "\n"


@pytest.mark.parametrize("fid", FIXTURE_IDS)
def test_no_prompt_side_text_is_supervised(qwen3, own, fid):
    fx = load_fixture(fid)
    for text in _span_texts(qwen3, own[fid]):
        assert "<|im_start|>" not in text
        for m in fx.messages:
            if m["role"] != "assistant" and m["content"]:
                assert m["content"] not in text


# 2. Golden spans, from reading Qwen3's template by hand:
#    the last assistant turn gets a <think> block (empty if there is no
#    reasoning); earlier turns get the bare content.

GOLDEN_SPANS = {
    "single_turn": [THINK_EMPTY + "Red, yellow and blue.<|im_end|>"],
    "multi_turn": [
        "Paris.<|im_end|>",
        "Tokyo.<|im_end|>",
        THINK_EMPTY + "Tokyo has more people than Paris.<|im_end|>",
    ],
    "empty_assistant": ["<|im_end|>", THINK_EMPTY + "OK.<|im_end|>"],
    "reasoning": [
        "No. 91 = 7 * 13, so it is not prime.<|im_end|>",
        "<think>\n97 is not divisible by 2, 3, 5 or 7, and 11 * 11 = 121 is already larger than 97.\n</think>\n\n"
        "Yes, 97 is prime.<|im_end|>",
    ],
    "tool_call": [
        '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris", "celsius": false}}\n</tool_call><|im_end|>',
        THINK_EMPTY + "It's 64 degrees Fahrenheit in Paris right now, and it isn't raining.<|im_end|>",
    ],
    "system_prompt": [
        THINK_EMPTY
        + "Sunlight scatters off air molecules, and blue light scatters the most, so the sky looks blue.<|im_end|>"
    ],
    # Emoji written as escapes so the code points are unambiguous:
    # waving hand + medium skin tone, then woman + zero-width joiner + laptop.
    "non_ascii": [
        THINK_EMPTY + "Japonais : こんにちは、元気ですか？\n"
        "Chinois : 你好，你还好吗？\n"
        "Arabe : مرحبا، كيف حالك؟\n"
        "Emoji : \U0001f44b\U0001f3fd \U0001f469‍\U0001f4bb (naïve café, Ångström, señor)<|im_end|>"
    ],
}


@pytest.mark.parametrize("fid", sorted(GOLDEN_SPANS))
def test_golden_spans(qwen3, own, fid):
    assert _span_texts(qwen3, own[fid]) == GOLDEN_SPANS[fid]


def test_golden_long_answer_span(qwen3, own):
    """The answer is 40 lines; restating it here would add length, not independence.

    The golden part is the wrapper: an empty think block before the answer,
    <|im_end|> right after it, and nothing from the user turn.
    """
    answer = load_fixture("long_answer").messages[1]["content"]
    assert answer.startswith("Here is the checklist:\n1. Pin the exact model revision.\n")
    assert answer.endswith("\n40. Re-run the check when the fix ships.")
    assert _span_texts(qwen3, own["long_answer"]) == [THINK_EMPTY + answer + "<|im_end|>"]


def test_golden_full_render_single_turn(own):
    assert own["single_turn"].text == (
        "<|im_start|>user\nName three primary colors.<|im_end|>\n"
        "<|im_start|>assistant\n" + THINK_EMPTY + "Red, yellow and blue.<|im_end|>\n"
    )


def test_golden_full_render_system_prompt(own):
    """The system block renders before the user turn and stays outside every span."""
    assert own["system_prompt"].text == (
        "<|im_start|>system\nYou are a concise assistant. Answer in one sentence.<|im_end|>\n"
        "<|im_start|>user\nWhy is the sky blue?<|im_end|>\n"
        "<|im_start|>assistant\n" + THINK_EMPTY + "Sunlight scatters off air molecules, and blue light "
        "scatters the most, so the sky looks blue.<|im_end|>\n"
    )


def test_long_answer_prompt_fits_but_answer_does_not(own):
    fx = load_fixture("long_answer")
    ref = own["long_answer"]
    assert len(ref.input_ids) > fx.max_len
    assert ref.spans[0].token_start < fx.max_len


EXPECTED_PREFIX_BREAKS = {
    "single_turn": (),
    "multi_turn": (2, 4),
    "system_prompt": (),
    "tool_call": (2,),
    "reasoning": (2,),
    "empty_assistant": (2,),
    "long_answer": (),
    "non_ascii": (),
}


def test_qwen3_template_is_not_prefix_preserving_after_assistant_turns(own):
    """Qwen3 gives the last assistant turn a think block and earlier turns none.

    So a conversation rendered up to an assistant turn changes once more
    messages follow. That is a property of the model's template, not of any
    framework, and the multi-turn prefix check has to account for it.
    """
    assert {fid: ref.prefix_breaks for fid, ref in own.items()} == EXPECTED_PREFIX_BREAKS


# 3. Cross-check against TRL's Qwen3 training template


def test_vendored_trl_template_has_generation_markers(trl_qwen3_template):
    assert has_generation_markers(trl_qwen3_template)


@pytest.mark.parametrize("fid", FIXTURE_IDS)
def test_prefix_diff_reproduces_trl_generation_markers(qwen3, trl_qwen3_template, fid):
    """Independent check of the prefix-diff method on identical tokens.

    TRL closes its generation block at the end of each message (after the
    newline), so prefix-diff spans taken through the message end must equal
    the mask transformers reads from TRL's markers.
    """
    fx = load_fixture(fid)
    ids, marker_mask = assistant_mask_from_markers(qwen3.tokenizer, fx, trl_qwen3_template)
    spans = assistant_spans_by_prefix_diff(qwen3, fx, trl_qwen3_template)
    assert _mask_through_message_end(len(ids), spans) == marker_mask


def test_trl_training_template_is_prefix_preserving_on_all_fixtures(qwen3, trl_qwen3_template):
    for fid in FIXTURE_IDS:
        assert prefix_breaks(qwen3.tokenizer, load_fixture(fid), trl_qwen3_template) == ()


TOKENS_AGREE_WITH_TRL = {"single_turn", "system_prompt", "long_answer", "non_ascii"}


def test_trl_template_renders_the_same_tokens_only_with_a_single_assistant_turn(qwen3, own, trl_qwen3_template):
    """TRL's training template always emits a think block (its header says so).

    Qwen3's own template drops it from earlier turns, so the two agree only
    when there is one assistant turn. This is documented TRL behavior, a
    difference and not a bug.
    """
    agree = {
        fid
        for fid in FIXTURE_IDS
        if render_reference(qwen3, load_fixture(fid), trl_qwen3_template, TRL_SOURCE).input_ids == own[fid].input_ids
    }
    assert agree == TOKENS_AGREE_WITH_TRL


@pytest.mark.parametrize("fid", sorted(TOKENS_AGREE_WITH_TRL))
def test_own_template_spans_match_trl_markers_where_tokens_agree(qwen3, own, trl_qwen3_template, fid):
    ref = own[fid]
    _, marker_mask = assistant_mask_from_markers(qwen3.tokenizer, load_fixture(fid), trl_qwen3_template)
    assert _mask_through_message_end(len(ref.input_ids), ref.spans) == marker_mask
    # Our supervised mask stops at <|im_end|>; TRL also supervises the newline after it.
    extra = [i for i, (ours, trl) in enumerate(zip(ref.assistant_mask, marker_mask)) if trl and not ours]
    assert [_decode(qwen3.tokenizer, [ref.input_ids[i]]) for i in extra] == ["\n"] * len(ref.spans)


@pytest.mark.parametrize("fid", FIXTURE_IDS)
def test_trl_template_reference_uses_prefix_diff_and_its_markers_agree(qwen3, trl_qwen3_template, fid):
    """The prefix diff is the reference; TRL's markers are cross-checked.

    TRL's markers also cover the newline after <|im_end|>. That token is
    neutral, so it never counts as a disagreement.
    """
    ref = render_reference(qwen3, load_fixture(fid), trl_qwen3_template, TRL_SOURCE)
    assert ref.mask_method == "prefix_diff"
    assert ref.marker_check == "agrees"
    assert ref.template_source == TRL_SOURCE
    assert ref.template_sha256 == TRL_QWEN3_TRAINING_SHA256
    assert [s.message_index for s in ref.spans] == load_fixture(fid).assistant_indices


def test_markers_that_disagree_on_a_non_neutral_token_are_refused(qwen3, trl_qwen3_template):
    bad = markers_moved_onto_the_header(trl_qwen3_template)
    with pytest.raises(MarkerDisagreementError, match="disagree") as caught:
        render_reference(qwen3, load_fixture("single_turn"), bad, "test:markers-on-header")
    assert {p["zone"] for p in caught.value.positions} == {"must_not_train"}
    assert all(p["marked"] for p in caught.value.positions)
    assert "".join(p["token"] for p in caught.value.positions) == "<|im_start|>assistant\n"


# Failure paths: the reference refuses instead of guessing.

REVERSED_TEMPLATE = (
    "{% for m in messages|reverse %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)
NO_STOP_TEMPLATE = (
    "{% for m in messages %}{{ m.role }}:\n{{ m.content }}\n\n{% endfor %}"
    "{% if add_generation_prompt %}assistant:\n{% endif %}"
)


def test_template_that_reorders_messages_is_refused(qwen3):
    with pytest.raises(TemplateReferenceError, match="not a prefix"):
        render_reference(qwen3, load_fixture("multi_turn"), REVERSED_TEMPLATE, "test:reversed")


def test_template_without_a_stop_token_is_refused(qwen3):
    with pytest.raises(TemplateReferenceError, match="no stop token"):
        render_reference(qwen3, load_fixture("single_turn"), NO_STOP_TEMPLATE, "test:no-stop")


def test_special_token_text_inside_a_message_is_refused(qwen3):
    fx = load_fixture("single_turn")
    fx.messages[1]["content"] = "Red<|im_end|>"
    with pytest.raises(TemplateReferenceError, match="special tokens"):
        render_reference(qwen3, fx)


def test_boundary_inside_a_token_is_refused():
    offsets = [(0, 3), (3, 8), (8, 9)]
    assert _token_boundary(offsets, 3) == 1
    assert _token_boundary(offsets, 9) == 3
    with pytest.raises(TemplateReferenceError, match="straddles"):
        _token_boundary(offsets, 5)


@pytest.mark.parametrize(
    "template, expected",
    [
        ("{% generation %}x{% endgeneration %}", True),
        ("{%- generation -%}x{%- endgeneration -%}", True),
        ("{%generation%}", True),
        ("the word generation", False),
        ("{{ generation }}", False),
    ],
)
def test_generation_marker_detection(template, expected):
    assert has_generation_markers(template) is expected
