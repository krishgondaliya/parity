"""KD-0004 and KD-0005: Axolotl 0.21.1 leaves DeepSeek V4.1's "</think>" untrained on conversations without reasoning,
thinking mode only. The same labeling rule as KD-0002 (traced: docs/evidence/2026-11-26-kd-deepseek-think-close/).
The stand-in below labels like Axolotl's default chat_template path, written independently of the signature: the
reference's labels, minus a "</think>" token that starts an assistant turn's span.
"""

from dataclasses import replace

import pytest
from stubs import official_stub

from parity.adapters.base import LABEL_IGNORE, Encoded
from parity.checks.common import official_reference, official_template
from parity.checks.known_differences import load_registry
from parity.checks.tier1 import AssistantMask, Truncation
from parity.engine.plants import PlantedAdapter
from parity.engine.root_causes import link_root_causes
from parity.engine.runner import evaluate
from parity.fixtures import load_fixture
from parity.references.template import template_sha256
from parity.release import load_release
from parity.targets import PHASE0_TARGETS

DEEPSEEK = next(t for t in PHASE0_TARGETS if t.repo_id == "deepseek-ai/DeepSeek-V4.1-Flash-1017")
WITHOUT_REASONING = ("single_turn", "multi_turn", "system_prompt", "long_answer", "non_ascii")


@pytest.fixture(scope="module")
def deepseek():
    return load_release(DEEPSEEK)


def axolotl_like(release, *, name="axolotl", version="0.21.1"):
    """Untrain a "</think>" token at the start of each assistant span."""
    tok = release.tokenizer

    def relabel(out, conv, cfg):
        if not isinstance(out, Encoded):
            return out
        labels = list(out.labels)
        ref = official_reference(release, conv, cfg.serving_mode)
        for span in ref.spans:
            i = span.token_start
            if i < len(labels) and tok.decode([out.input_ids[i]]) == "</think>":
                labels[i] = LABEL_IGNORE
        return replace(out, labels=tuple(labels))

    stub = official_stub(release, name=name, version=version)
    return PlantedAdapter(stub, "axolotl-like labels", after_encode=relabel)


def test_the_entries_are_scoped_like_kd_0002_and_share_one_root_cause(deepseek):
    registry = {e.id: e for e in load_registry()}
    for kd, check in (("KD-0004", "t1.assistant_mask"), ("KD-0005", "t1.truncation")):
        e = registry[kd]
        assert (e.framework, e.versions, e.check, e.serving_modes) == ("axolotl", ("0.21.1",), check, ("thinking",))
        assert e.docs == registry["KD-0002"].docs  # the same Axolotl test and code lines
        assert e.model_template_sha256 == e.framework_template_sha256 == template_sha256(official_template(deepseek))
    assert (registry["KD-0004"].finding, registry["KD-0005"].finding) == ("", "KD-0004")


@pytest.mark.parametrize("fid", WITHOUT_REASONING)
def test_the_untrained_think_close_is_kd_0004_without_reasoning(deepseek, fid):
    v = evaluate(AssistantMask(), axolotl_like(deepseek), load_fixture(fid))
    assert (v.result, v.kind, v.known_difference, v.self_test) == ("fail", "difference", "KD-0004", "caught")
    [linked] = link_root_causes([v])
    assert linked.root_cause == "KD-0004"


def test_kd_0005_covers_it_under_truncation_with_the_same_root_cause(deepseek):
    v = evaluate(Truncation(), axolotl_like(deepseek), load_fixture("single_turn"))
    assert (v.result, v.kind, v.known_difference) == ("fail", "difference", "KD-0005")
    [linked] = link_root_causes([v])
    assert linked.root_cause == "KD-0004"


def test_a_conversation_with_reasoning_is_not_covered(deepseek):
    """Reasoning anywhere takes a conversation out of scope (there Axolotl drops the reasoning, P-0007), even where
    the untrained token is the same: multi_turn's last turn starts with "</think>", and its first turn has reasoning,
    which DeepSeek's template doesn't render on a history turn."""
    conv = load_fixture("multi_turn")
    messages = [dict(m) for m in conv.messages]
    next(m for m in messages if m["role"] == "assistant")["reasoning_content"] = "Recall the capital."
    v = evaluate(AssistantMask(), axolotl_like(deepseek), replace(conv, messages=messages))
    assert [t["token"] for t in v.evidence["must_train_unsupervised"]] == ["</think>"]
    assert (v.result, v.kind, v.known_difference) == ("fail", "bug", None)


def test_untraining_more_than_the_think_close_stays_a_bug(deepseek):
    def also_untrain_the_first_content_token(out, conv, cfg):
        labels = list(out.labels)
        first = next(i for i, lab in enumerate(labels) if lab != LABEL_IGNORE)
        labels[first] = LABEL_IGNORE
        return replace(out, labels=tuple(labels))

    worse = PlantedAdapter(axolotl_like(deepseek), "untrain content", after_encode=also_untrain_the_first_content_token)
    v = evaluate(AssistantMask(), worse, load_fixture("single_turn"))
    assert (v.result, v.kind) == ("fail", "bug")


@pytest.mark.parametrize("name,version", [("axolotl", "0.21.2"), ("trl", "1.18.2")])
def test_another_framework_or_version_is_not_covered(deepseek, name, version):
    v = evaluate(AssistantMask(), axolotl_like(deepseek, name=name, version=version), load_fixture("single_turn"))
    assert (v.result, v.kind, v.known_difference) == ("fail", "bug", None)


def test_qwen3_is_not_covered_by_the_deepseek_entries(qwen3):
    """The new signature is pinned to DeepSeek's template: on Qwen3, KD-0002 still decides."""
    v = evaluate(AssistantMask(), axolotl_like(qwen3), load_fixture("single_turn"))
    assert v.known_difference != "KD-0004"
