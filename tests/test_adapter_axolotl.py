"""The Axolotl adapter on the calibration model. Runs only in Axolotl's environment (a few minutes):

    envs/axolotl-0.21.1/.venv/bin/python -m pytest tests/test_adapter_axolotl.py -q

It checks the adapter, not Axolotl: declare() reports Axolotl's documented defaults for our config, read from the
loaded config and live collator, and encode/encode_batch run on every fixture through Axolotl's own data path.
Which tokens Axolotl trains is the checks' job, not this file's.
"""

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

axolotl = pytest.importorskip("axolotl")
if axolotl.__version__ != "0.21.1":
    pytest.skip(f"the Axolotl adapter is for axolotl==0.21.1, found {axolotl.__version__}", allow_module_level=True)

from parity.adapters.axolotl import ENV_LOCK, AxolotlAdapter  # noqa: E402
from parity.adapters.base import LABEL_IGNORE, Dropped, TrainCfg  # noqa: E402
from parity.fixtures import FIXTURE_IDS, load_fixture  # noqa: E402
from parity.references.template import model_chat_template, template_token_ids  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def adapter(qwen3):
    return AxolotlAdapter(qwen3)


@pytest.mark.parametrize("cfg", [TrainCfg(), TrainCfg(max_len=128), TrainCfg(pad_to=64)], ids=["base", "cut", "pad"])
def test_declare_matches_axolotls_documented_defaults_for_our_config(adapter, cfg):
    d = adapter.declare(cfg)
    assert d.template_source == "model"  # chat_template defaults to tokenizer_default
    assert d.chat_template == model_chat_template(adapter.release.tokenizer)
    assert d.truncation_mode == "drop"  # excess_length_strategy unset: "drop"
    assert d.batching == "padded"  # sample_packing off
    assert d.pad_token_id == adapter.release.tokenizer.pad_token_id  # <|endoftext|>
    assert d.label_convention == "aligned"
    assert any("roles_to_train=['assistant']" in s and "train_on_eos='turn'" in s and "conversation.qmd" in s for s in d.sources)
    limit = cfg.max_len if cfg.max_len is not None else 8192
    assert any(f"sequence_len={limit}" in s for s in d.sources)


def test_the_environment_is_pinned_and_telemetry_is_off(adapter):
    import os

    assert adapter.env_lock == ENV_LOCK == "envs/axolotl-0.21.1/uv.lock"
    assert adapter.env_fingerprint == hashlib.sha256((REPO / ENV_LOCK).read_bytes()).hexdigest()
    assert os.environ["AXOLOTL_DO_NOT_TRACK"] == "1" and os.environ["DO_NOT_TRACK"] == "1"


@pytest.mark.parametrize("fid", FIXTURE_IDS)
def test_encode_runs_through_axolotl_on_every_fixture(qwen3, adapter, fid):
    """Axolotl's prepared row has exactly the declared template's tokens. Which of them it trains, the checks judge."""
    fx = load_fixture(fid)
    row = adapter.encode(fx, TrainCfg())
    assert not isinstance(row, Dropped)
    assert list(row.input_ids) == template_token_ids(qwen3.tokenizer, fx, adapter.declare(TrainCfg()).chat_template)
    assert all(label in (LABEL_IGNORE, token) for label, token in zip(row.labels, row.input_ids))
    assert set(row.attention_mask) == {1}


@pytest.mark.parametrize("fid", FIXTURE_IDS)
def test_encode_batch_pads_through_axolotls_collator(adapter, fid):
    partner = load_fixture("long_answer" if fid != "long_answer" else "single_turn")
    convs = [load_fixture(fid), partner]
    batch = adapter.encode_batch(convs, replace(TrainCfg(), pad_to=600))
    assert len(batch.rows) == 2
    for conv, row in zip(convs, batch.rows):
        single = adapter.encode(conv, TrainCfg())
        n = len(single.input_ids)
        assert len(row.input_ids) == 600
        assert row.input_ids[:n] == single.input_ids and row.labels[:n] == single.labels
        assert set(row.input_ids[n:]) == {adapter.declare(TrainCfg(pad_to=600)).pad_token_id}
        assert set(row.labels[n:]) == {LABEL_IGNORE} and set(row.attention_mask[n:]) == {0}


def test_an_over_length_row_is_dropped(adapter):
    """Documented default: excess_length_strategy unset means 'drop'."""
    assert isinstance(adapter.encode(load_fixture("long_answer"), TrainCfg(max_len=128)), Dropped)
