"""KD-0002 and KD-0003: Axolotl 0.21.1 leaves Qwen3's think-block opener untrained (P-0001), thinking mode only.

The stand-in below labels like Axolotl's default chat_template path, written independently of the signature:
the reference's labels, minus the think-tag and newline tokens at the very start of each assistant turn.
"""

from dataclasses import replace

import pytest
from stubs import official_stub

from parity.adapters.base import LABEL_IGNORE, Encoded
from parity.checks.common import official_reference
from parity.checks.known_differences import load_registry
from parity.checks.tier1 import AssistantMask, Truncation
from parity.engine.plants import PlantedAdapter
from parity.engine.runner import evaluate
from parity.fixtures import FIXTURE_IDS, load_fixture

THINK_STRUCTURE = {"<think>", "</think>", "\n", "\n\n"}


def axolotl_like(qwen3, *, name="axolotl", version="0.21.1", empty_turn_bug=False):
    """Untrain leading think structure of each assistant turn; optionally P-0002 (empty turn's end token)."""
    tok = qwen3.tokenizer

    def relabel(out, conv, cfg):
        if not isinstance(out, Encoded):
            return out
        labels = list(out.labels)
        ref = official_reference(qwen3, conv)
        for span in ref.spans:
            i = span.token_start
            while i < span.token_end and i < len(labels) and tok.decode([out.input_ids[i]]) in THINK_STRUCTURE:
                labels[i] = LABEL_IGNORE
                i += 1
            if empty_turn_bug and conv.messages[span.message_index]["content"] == "" and span.token_end - 1 < len(labels):
                labels[span.token_end - 1] = LABEL_IGNORE
        return replace(out, labels=tuple(labels))

    stub = official_stub(qwen3, name=name, version=version)
    return PlantedAdapter(stub, "axolotl-like labels", after_encode=relabel)


def test_the_entries_are_scoped_to_thinking_mode_and_pinned_to_axolotls_test():
    registry = {e.id: e for e in load_registry()}
    for kd, check in (("KD-0002", "t1.assistant_mask"), ("KD-0003", "t1.truncation")):
        e = registry[kd]
        assert (e.framework, e.versions, e.check, e.serving_modes) == ("axolotl", ("0.21.1",), check, ("thinking",))
        assert any("v0.21.1/tests/prompt_strategies/test_chat_templates_thinking.py#L143-L177" in d for d in e.docs)
        assert "docs gap" in e.docs_note


@pytest.mark.parametrize("fid", FIXTURE_IDS)
def test_axolotls_think_opener_is_kd_0002_in_thinking_mode(qwen3, fid):
    v = evaluate(AssistantMask(), axolotl_like(qwen3), load_fixture(fid))
    assert (v.result, v.kind, v.known_difference, v.self_test) == ("fail", "difference", "KD-0002", "caught")


def test_the_reasoning_text_and_closing_tag_stay_must_train(qwen3):
    """On `reasoning`, only "<think>" and "\\n" are explained; untraining the reasoning too is a bug."""
    v = evaluate(AssistantMask(), axolotl_like(qwen3), load_fixture("reasoning"))
    assert [t["token"] for t in v.evidence["must_train_unsupervised"]] == ["<think>", "\n"]

    def also_untrain_first_reasoning_token(out, conv, cfg):
        labels = list(out.labels)
        labels[53] = LABEL_IGNORE  # "9", the first reasoning token
        return replace(out, labels=tuple(labels))

    worse = PlantedAdapter(axolotl_like(qwen3), "untrain reasoning", after_encode=also_untrain_first_reasoning_token)
    v = evaluate(AssistantMask(), worse, load_fixture("reasoning"))
    assert (v.result, v.kind) == ("fail", "bug")


@pytest.mark.parametrize("fid", [f for f in FIXTURE_IDS if f != "reasoning"])
def test_non_thinking_mode_needs_no_known_difference(qwen3, fid):
    v = evaluate(AssistantMask("non_thinking"), axolotl_like(qwen3), load_fixture(fid))
    assert (v.serving_mode, v.result, v.known_difference) == ("non_thinking", "pass", None)


def test_kd_0003_covers_the_same_behavior_under_truncation(qwen3):
    v = evaluate(Truncation(), axolotl_like(qwen3), load_fixture("single_turn"))
    assert (v.result, v.kind, v.known_difference) == ("fail", "difference", "KD-0003")


@pytest.mark.parametrize("name,version", [("axolotl", "0.21.2"), ("trl", "1.18.2")])
def test_another_framework_or_version_is_not_covered(qwen3, name, version):
    v = evaluate(AssistantMask(), axolotl_like(qwen3, name=name, version=version), load_fixture("single_turn"))
    assert (v.result, v.kind, v.known_difference) == ("fail", "bug", None)


def test_a_second_root_cause_on_the_same_fixture_stays_a_bug_and_names_whats_left(qwen3):
    """empty_assistant with P-0002 too: KD-0002 explains the think opener, not the empty turn's <|im_end|>."""
    v = evaluate(AssistantMask(), axolotl_like(qwen3, empty_turn_bug=True), load_fixture("empty_assistant"))
    assert (v.result, v.kind, v.known_difference) == ("fail", "bug", None)
    [residual] = v.evidence["known_difference_residual"]
    assert residual["id"] == "KD-0002"
    left = residual["unexplained"]["must_train_unsupervised"]
    assert [(t["index"], t["token"]) for t in left] == [(14, "<|im_end|>")]
