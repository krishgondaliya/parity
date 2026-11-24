"""The test stand-ins behave as documented, so check tests can rely on them."""

from dataclasses import replace

import pytest
from stubs import TRL_SOURCE, TemplateStub

from parity.adapters.base import LABEL_IGNORE, Dropped, TrainCfg
from parity.checks.common import mask_report, official_reference, reference_on
from parity.fixtures import FIXTURE_IDS, load_fixture


@pytest.mark.parametrize("fid", FIXTURE_IDS)
def test_official_stand_in_encodes_the_reference(qwen3, official, fid):
    fx = load_fixture(fid)
    ref = official_reference(qwen3, fx)
    row = official.encode(fx, TrainCfg())
    assert row.input_ids == ref.input_ids
    report = mask_report(ref.zones, row)
    assert not report.violated and report.neutral_convention == "not_trained"


@pytest.mark.parametrize("fid", FIXTURE_IDS)
def test_trl_stand_in_trains_the_neutral_newline_like_trl_markers(qwen3, trl_markers, trl_qwen3_template, fid):
    fx = load_fixture(fid)
    ref = reference_on(qwen3, fx, trl_qwen3_template, TRL_SOURCE)
    row = trl_markers.encode(fx, TrainCfg())
    assert row.input_ids == ref.input_ids
    report = mask_report(ref.zones, row)
    assert not report.violated and report.neutral_convention == "trained"


@pytest.mark.parametrize("mode", ["right", "left", "drop"])
def test_stand_in_truncation_modes(qwen3, mode):
    fx = load_fixture("long_answer")
    full = official_reference(qwen3, fx).input_ids
    out = TemplateStub(qwen3, truncation_mode=mode).encode(fx, TrainCfg(max_len=fx.max_len))
    if mode == "drop":
        assert isinstance(out, Dropped)
    else:
        assert out.input_ids == (full[: fx.max_len] if mode == "right" else full[-fx.max_len :])


@pytest.mark.parametrize("side", ["right", "left"])
def test_stand_in_pads_through_pad_to(qwen3, side):
    stub = TemplateStub(qwen3, padding_side=side)
    convs = [load_fixture("single_turn"), load_fixture("multi_turn")]
    batch = stub.encode_batch(convs, replace(TrainCfg(), pad_to=80))
    for conv, row in zip(convs, batch.rows):
        single = stub.encode(conv, TrainCfg())
        n = 80 - len(single.input_ids)
        assert len(row.input_ids) == 80 and sum(row.attention_mask) == len(single.input_ids)
        pads = slice(len(single.input_ids), None) if side == "right" else slice(0, n)
        assert set(row.input_ids[pads]) == {qwen3.tokenizer.pad_token_id}
        assert set(row.labels[pads]) == {LABEL_IGNORE}
