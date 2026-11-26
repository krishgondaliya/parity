"""The serving mode: which chat-template arguments the reference renders with (parity.serving).

Decided after the Axolotl results on Qwen3 (P-0001): no neutral zone for mode-dependent tokens. The mode is
declared, defaults to the model's documented default, is recorded in every verdict, and decides those tokens' zone.
"""

import pytest
from stubs import official_stub

from parity.adapters.base import TrainCfg
from parity.checks.common import official_reference
from parity.checks.tier1 import AssistantMask
from parity.engine.runner import evaluate
from parity.fixtures import FIXTURE_IDS, load_fixture
from parity.references.template import TemplateReferenceError, Zone
from parity.serving import ServingModeError, framework_mode, resolve_mode, template_kwargs

THINK_BLOCK = ["<think>", "\n\n", "</think>", "\n\n"]


def test_train_cfg_validates_the_mode():
    with pytest.raises(ValueError):
        TrainCfg(serving_mode="fast")


def test_frameworks_are_told_the_mode_only_when_it_isnt_the_models_default(qwen3):
    """The primary (default-mode) rows are the framework's own defaults; another mode gets its documented setting."""
    assert framework_mode(qwen3, None) is None
    assert framework_mode(qwen3, "thinking") is None
    assert framework_mode(qwen3, "non_thinking") == "non_thinking"


def test_a_framework_with_no_way_to_set_the_mode_is_not_configurable_never_failed(qwen3):
    from stubs import official_stub

    from parity.engine.summary import summarize

    stub = official_stub(qwen3, mode_configurable=False, name="stub-no-mode")
    fx = load_fixture("single_turn")
    primary = evaluate(AssistantMask(), stub, fx)
    secondary = evaluate(AssistantMask("non_thinking"), stub, fx)
    assert primary.result == "pass"
    assert (secondary.result, secondary.kind, secondary.self_test) == ("not_configurable", None, None)
    assert "documents no way to set the serving mode" in secondary.reason
    [cell] = summarize([secondary])
    assert (cell.result, cell.ran_on, cell.not_configurable) == ("not_configurable", 0, 1)


def test_qwen3_defaults_to_thinking_mode(qwen3):
    assert resolve_mode(qwen3, None) == "thinking"
    assert template_kwargs(qwen3, "non_thinking") == {"enable_thinking": False}


def test_the_mode_moves_only_the_last_turns_empty_think_block(qwen3):
    """Same tokens in both modes; only the empty think block of the last turn changes zone."""
    tok = qwen3.tokenizer
    for fid in FIXTURE_IDS:
        if fid == "reasoning":
            continue
        fx = load_fixture(fid)
        thinking, non_thinking = official_reference(qwen3, fx, "thinking"), official_reference(qwen3, fx, "non_thinking")
        assert thinking.input_ids == non_thinking.input_ids, fid
        moved = [i for i, (a, b) in enumerate(zip(thinking.zones, non_thinking.zones)) if a != b]
        assert [tok.decode([thinking.input_ids[i]]) for i in moved] == THINK_BLOCK, fid
        assert all(thinking.zones[i] is Zone.MUST_TRAIN for i in moved)
        assert all(non_thinking.zones[i] is Zone.MUST_NOT_TRAIN for i in moved)
        last = non_thinking.spans[-1]
        assert moved == list(range(last.token_start - 4, last.token_start)), fid


def test_a_reasoning_trace_has_no_non_thinking_reading(qwen3):
    """Non-thinking mode's generation prompt closes the think block empty, so a last turn that reasons contradicts it.

    The runner never asks: such a fixture is not applicable in non-thinking mode (human decision, 2026-11-25).
    """
    with pytest.raises(TemplateReferenceError):
        official_reference(qwen3, load_fixture("reasoning"), "non_thinking")


def test_only_a_fixture_that_carries_reasoning_is_not_applicable_and_only_in_non_thinking_mode():
    from parity.fixtures import FIXTURE_IDS
    from parity.serving import not_applicable_in_mode

    for fid in FIXTURE_IDS:
        fx = load_fixture(fid)
        assert not_applicable_in_mode(fx, "thinking") is None, fid
        assert (not_applicable_in_mode(fx, "non_thinking") is not None) == (fid == "reasoning"), fid


def test_a_model_without_a_documented_non_thinking_mode_refuses(qwen3):
    from dataclasses import replace

    other = replace(qwen3, release=replace(qwen3.release, repo_id="example/no-modes"))
    assert resolve_mode(other, None) == "thinking"
    with pytest.raises(ServingModeError):
        template_kwargs(other, "non_thinking")


def test_the_verdict_records_the_mode_and_the_zone_follows_it(qwen3):
    """A stand-in that labels like non-thinking data fails in thinking mode and passes in non-thinking mode."""
    stub = official_stub(qwen3, labels_mode="non_thinking", name="stub-non-thinking-labels")
    fx = load_fixture("single_turn")
    primary = evaluate(AssistantMask(), stub, fx)
    secondary = evaluate(AssistantMask("non_thinking"), stub, fx)
    assert (primary.serving_mode, primary.result) == ("thinking", "fail")
    assert (secondary.serving_mode, secondary.result) == ("non_thinking", "pass")
    assert [t["token"] for t in primary.evidence["must_train_unsupervised"]] == THINK_BLOCK
    assert primary.reference["template_kwargs"] == {"enable_thinking": True}
    assert secondary.reference["template_kwargs"] == {"enable_thinking": False}


def test_every_phase0_targets_default_mode_is_cited_at_its_pinned_revision():
    """M5 prep (2026-11-25): each target's default mode is fixed and cited before any run, from permalinks at the
    pinned revision; the citation level says whether the model card or only the template sets it."""
    from parity.serving import MODE_SPECS
    from parity.targets import PHASE0_TARGETS

    for target in PHASE0_TARGETS:
        spec = MODE_SPECS[target.repo_id]
        assert spec.default in ("thinking", "non_thinking")
        assert spec.citation_level in ("model card", "template only")
        assert spec.sources, target.repo_id
        for url in spec.sources:
            assert url.startswith(f"https://huggingface.co/{target.repo_id}/blob/{target.revision}/"), url
