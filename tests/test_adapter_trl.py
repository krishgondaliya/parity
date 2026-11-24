"""The TRL adapter on the calibration model. Runs only in TRL's environment:

    envs/trl-1.18.2/.venv/bin/python -m pytest tests/test_adapter_trl.py -q

It checks the adapter, not TRL: declare() reports TRL's documented defaults for our config, read from TRL's live
objects, and encode/encode_batch run on every fixture through TRL's own data path.
"""

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

trl = pytest.importorskip("trl")

from stubs import TRL_QWEN3_TRAINING_SHA256  # noqa: E402

from parity.adapters.base import LABEL_IGNORE, Dropped, TrainCfg  # noqa: E402
from parity.adapters.trl import ENV_LOCK, TRLAdapter  # noqa: E402
from parity.fixtures import FIXTURE_IDS, load_fixture  # noqa: E402
from parity.references.template import assistant_mask_from_markers, template_sha256  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def adapter(qwen3):
    return TRLAdapter(qwen3)


@pytest.mark.parametrize("cfg", [TrainCfg(), TrainCfg(max_len=128), TrainCfg(pad_to=64)], ids=["base", "cut", "pad"])
def test_declare_matches_trls_documented_defaults_for_our_config(adapter, cfg):
    d = adapter.declare(cfg)
    # assistant_only_loss=True makes TRL swap in its Qwen3 training template (docs: sft_trainer.md).
    assert d.template_source == "trl@1.18.2:qwen3_training.jinja"
    assert template_sha256(d.chat_template) == TRL_QWEN3_TRAINING_SHA256
    assert d.truncation_mode == "right"  # SFTConfig.truncation_mode defaults to "keep_start"
    assert d.batching == "padded"  # packing and padding_free default to False
    assert d.pad_token_id == adapter.release.tokenizer.pad_token_id  # <|endoftext|>, from the tokenizer
    assert d.label_convention == "aligned"
    assert any("assistant_only_loss=True" in s and "sft_trainer.md" in s for s in d.sources)
    assert any(f"max_length={cfg.max_len}" in s for s in d.sources)


def test_the_environment_is_pinned(adapter):
    assert adapter.env_lock == ENV_LOCK == "envs/trl-1.18.2/uv.lock"
    assert adapter.env_fingerprint == hashlib.sha256((REPO / ENV_LOCK).read_bytes()).hexdigest()
    assert trl.__version__ == "1.18.2"


@pytest.mark.parametrize("fid", FIXTURE_IDS)
def test_encode_runs_through_trl_on_every_fixture(qwen3, adapter, fid):
    """TRL's row equals its declared template's render, with loss exactly where its generation markers put it."""
    fx = load_fixture(fid)
    row = adapter.encode(fx, TrainCfg())
    assert not isinstance(row, Dropped)
    template = adapter.declare(TrainCfg()).chat_template
    ids, mask = assistant_mask_from_markers(qwen3.tokenizer, fx, template)
    assert list(row.input_ids) == ids
    assert [label != LABEL_IGNORE for label in row.labels] == [bool(m) for m in mask]
    assert all(label in (LABEL_IGNORE, token) for label, token in zip(row.labels, row.input_ids))
    assert set(row.attention_mask) == {1}


@pytest.mark.parametrize("fid", FIXTURE_IDS)
def test_encode_batch_pads_through_trls_collator(adapter, fid):
    partner = load_fixture("long_answer" if fid != "long_answer" else "single_turn")
    batch = adapter.encode_batch([load_fixture(fid), partner], replace(TrainCfg(), pad_to=600))
    assert len(batch.rows) == 2
    for conv, row in zip([load_fixture(fid), partner], batch.rows):
        single = adapter.encode(conv, TrainCfg())
        n = len(single.input_ids)
        assert len(row.input_ids) == 600
        assert row.input_ids[:n] == single.input_ids and row.labels[:n] == single.labels
        assert set(row.input_ids[n:]) == {adapter.declare(TrainCfg(pad_to=600)).pad_token_id}
        assert set(row.labels[n:]) == {LABEL_IGNORE} and set(row.attention_mask[n:]) == {0}


def test_truncation_keeps_the_start(adapter):
    full = adapter.encode(load_fixture("long_answer"), TrainCfg())
    cut = adapter.encode(load_fixture("long_answer"), TrainCfg(max_len=128))
    assert cut.input_ids == full.input_ids[:128] and cut.labels == full.labels[:128]


def test_trl_drops_a_row_left_with_nothing_to_train(adapter):
    """Documented in TRL's code: fully masked examples are dropped after truncation."""
    out = adapter.encode(replace(load_fixture("long_answer"), max_len=None), TrainCfg(max_len=20))
    assert isinstance(out, Dropped)
