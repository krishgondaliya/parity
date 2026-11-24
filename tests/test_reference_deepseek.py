"""Pre-fill anchoring in the reference, with goldens on DeepSeek V4.1 and Qwen3 in non-thinking mode.

DeepSeek's template ends the generation prompt with a pre-fill,
`<｜Assistant｜><think>`, and renders earlier assistant turns as
`<｜Assistant｜></think>...` (it emits `"<｜Assistant｜>" ~ ("<think>" if thinking
... else "</think>")`). The reference anchors content past such a missing
pre-fill only in a narrow case, and records it as a template property.

DeepSeek's history turns still refuse, for a second reason: when an
assistant turn is removed, the template merges the two user messages around
it ("...France?\\n\\nAnd of Japan?"), so no cut-out delimits the turn. That
is a separate rule, left refusing.
"""

import pytest

from parity.checks.common import official_template
from parity.checks.tier1 import TIER1_CHECKS
from parity.engine.runner import evaluate
from parity.fixtures import FIXTURE_IDS, load_all, load_fixture
from parity.references.properties import prefill_property, prefill_property_id
from parity.references.template import (
    TemplateReferenceError,
    Zone,
    prefill_missing_turns,
    render_reference,
    template_sha256,
)
from parity.release import load_release
from parity.targets import PHASE0_TARGETS

DEEPSEEK = next(t for t in PHASE0_TARGETS if t.repo_id == "deepseek-ai/DeepSeek-V4.1-Flash-1017")
EOS = "<｜end▁of▁sentence｜>"
QWEN3_EMPTY_THINK = "<think>\n\n</think>\n\n"


@pytest.fixture(scope="module")
def deepseek():
    return load_release(DEEPSEEK)


@pytest.fixture(scope="module")
def qwen3_non_thinking(qwen3):
    """Qwen3's own template with its non-thinking generation prompt: it pre-fills an empty think block."""
    template = official_template(qwen3)
    tail = template[template.index("{%- if add_generation_prompt %}") :]
    condition = "enable_thinking is defined and enable_thinking is false"
    assert tail.count(condition) == 1
    return template.replace(tail, tail.replace(condition, "true"))


def _decode(tokenizer, ids):
    return tokenizer.decode(list(ids), skip_special_tokens=False, clean_up_tokenization_spaces=False)


def _spans(release, ref):
    tok = release.tokenizer
    out = []
    for s in ref.spans:
        header = _decode(tok, ref.input_ids[s.token_message_start : s.token_start])
        out.append((header, _decode(tok, ref.input_ids[s.token_start : s.token_end]), s.prefill_missing))
    return out


# Golden 1, DeepSeek single_turn, from reading its template: the generation prompt "<｜Assistant｜><think>"
# is prompt side; the model writes "</think>" and the answer, then EOS; nothing follows the EOS.


def test_golden_deepseek_single_turn(deepseek):
    ref = render_reference(deepseek, load_fixture("single_turn"))
    assert _spans(deepseek, ref) == [("<｜Assistant｜><think>", "</think>Red, yellow and blue." + EOS, None)]
    assert Zone.NEUTRAL not in ref.zones
    # The template adds a default system prompt when the conversation has none.
    assert ref.text.startswith("<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 75")


# Golden 2, the anchoring itself, on Qwen3 in non-thinking mode: history turns render without the empty
# think block the generation prompt pre-fills, so their content is anchored right after the role header.


def test_golden_anchored_spans_qwen3_non_thinking(qwen3, qwen3_non_thinking):
    ref = render_reference(qwen3, load_fixture("multi_turn"), qwen3_non_thinking, "test:qwen3-non-thinking")
    header = "<|im_start|>assistant\n"
    assert _spans(qwen3, ref) == [
        (header, "Paris.<|im_end|>", QWEN3_EMPTY_THINK),
        (header, "Tokyo.<|im_end|>", QWEN3_EMPTY_THINK),
        (header + QWEN3_EMPTY_THINK, "Tokyo has more people than Paris.<|im_end|>", None),
    ]


def test_deepseek_history_turns_still_refuse_for_the_merge(deepseek):
    """Anything other than the missing pre-fill still refuses."""
    refused = {}
    for fid in FIXTURE_IDS:
        try:
            render_reference(deepseek, load_fixture(fid))
        except TemplateReferenceError as exc:
            refused[fid] = str(exc)
    assert set(refused) == {"multi_turn", "reasoning", "empty_assistant", "tool_call"}
    assert all("cutting it out of the full render" in reason for reason in refused.values())


def test_deepseek_has_the_pre_fill_property(deepseek):
    assert prefill_missing_turns(deepseek, load_fixture("multi_turn")) == [(1, "<think>"), (3, "<think>")]
    prop = prefill_property(deepseek, official_template(deepseek), "model", load_all())
    assert prop.kind == "prefill_missing_from_history"
    assert {d["fixture"]: [t["message_index"] for t in d["turns"]] for d in prop.details} == {
        "multi_turn": [1, 3],
        "reasoning": [1],
        "empty_assistant": [1],
    }


def test_verdicts_on_anchored_spans_name_the_property(qwen3, qwen3_non_thinking):
    from stubs import TemplateStub

    stub = TemplateStub(qwen3, chat_template=qwen3_non_thinking, template_source="test:qwen3-non-thinking")
    v = evaluate(TIER1_CHECKS[1], stub, load_fixture("multi_turn"))
    assert (v.result, v.self_test) == ("pass", "caught"), (v.reason, v.plants)
    assert v.template_properties == [prefill_property_id(qwen3, template_sha256(qwen3_non_thinking))]
    assert evaluate(TIER1_CHECKS[0], stub, load_fixture("multi_turn")).template_properties == []


def test_qwen3s_own_template_has_no_missing_pre_fill(qwen3):
    assert prefill_property(qwen3, official_template(qwen3), "model", load_all()) is None


def test_a_different_header_is_not_a_pre_fill(qwen3):
    """A generation prompt the full render never shows verbatim (here a capitalized role) still refuses."""
    template = official_template(qwen3)
    tail = template[template.index("{%- if add_generation_prompt %}") :]
    changed = template.replace(tail, tail.replace("<|im_start|>assistant", "<|im_start|>Assistant"))
    assert changed != template
    with pytest.raises(TemplateReferenceError, match="not a prefix"):
        render_reference(qwen3, load_fixture("multi_turn"), changed, "test:capitalized-generation-prompt")
