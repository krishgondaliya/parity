"""Unsloth 2026.11.10 adapter, on the calibration model. Runs inside envs/unsloth-2026.11.10; skipped elsewhere."""

import pytest

unsloth = pytest.importorskip("unsloth")
if unsloth.__version__ != "2026.11.10":
    pytest.skip("needs unsloth==2026.11.10 (envs/unsloth-2026.11.10)", allow_module_level=True)

from parity.adapters.base import TrainCfg  # noqa: E402
from parity.adapters.unsloth import UnslothAdapter  # noqa: E402
from parity.fixtures import FIXTURE_IDS, load_fixture  # noqa: E402
from parity.references.template import template_token_ids  # noqa: E402


@pytest.fixture(scope="module")
def adapter(qwen3):
    return UnslothAdapter(qwen3)


def test_declare_reads_the_recipes_live_objects(adapter, qwen3):
    d = adapter.declare(TrainCfg())
    assert d.template_source == "model"  # the loader leaves Qwen3's template unchanged
    assert (d.label_convention, d.truncation_mode, d.batching) == ("aligned", "right", "padded")
    assert d.pad_token_id == qwen3.tokenizer.convert_tokens_to_ids("<|endoftext|>")
    assert d.tool_arguments == "passed_through"
    assert d.serving_mode_setting.startswith("framework defaults")
    assert any("train_on_responses_only" in s for s in d.sources)


def test_a_non_default_mode_reaches_the_formatting_step(adapter):
    d = adapter.declare(TrainCfg(serving_mode="non_thinking"))
    assert "{'enable_thinking': False} passed to tokenizer.apply_chat_template" in d.serving_mode_setting


@pytest.mark.parametrize("fid", FIXTURE_IDS)
def test_encode_runs_on_every_fixture(adapter, fid):
    row = adapter.encode(load_fixture(fid), TrainCfg())
    assert len(row.input_ids) == len(row.labels) == len(row.attention_mask)
    assert any(label != -100 for label in row.labels)


def test_encode_batch_runs(adapter):
    batch = adapter.encode_batch([load_fixture("single_turn"), load_fixture("multi_turn")], TrainCfg())
    assert len({len(r.input_ids) for r in batch.rows}) == 1


@pytest.mark.parametrize("fid", FIXTURE_IDS)
def test_rows_are_the_declared_render_plus_one_appended_eos(adapter, qwen3, fid):
    """What the recipe actually produces, pinned exactly: TRL 1.15 appends the EOS token to a "text" row that doesn't
    end with it, and Qwen3's render ends with a newline. Under /triage (M3 Unsloth); if this changes, re-triage."""
    fx = load_fixture(fid)
    declared = adapter.declare(TrainCfg()).chat_template
    eos = qwen3.tokenizer.convert_tokens_to_ids("<|im_end|>")
    row = adapter.encode(fx, TrainCfg())
    assert list(row.input_ids) == template_token_ids(qwen3.tokenizer, fx, declared) + [eos]


@pytest.mark.xfail(strict=True, reason="rows carry one appended EOS (see the test above); under /triage, M3 Unsloth")
def test_rows_match_the_declared_template_exactly(adapter, qwen3):
    """The adapter contract (the new-adapter skill): rows equal apply_chat_template with the declared template."""
    fx = load_fixture("single_turn")
    declared = adapter.declare(TrainCfg()).chat_template
    assert list(adapter.encode(fx, TrainCfg()).input_ids) == template_token_ids(qwen3.tokenizer, fx, declared)
