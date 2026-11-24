"""t1.special_tokens on the calibration model, against the test stand-ins.

Qwen3 has no BOS; its pad token <|endoftext|> is also one of its stop tokens
(generation_config), while assistant turns close with <|im_end|>.
"""

from dataclasses import replace

import pytest
from stubs import TemplateStub, plant_named, run_on_all_fixtures, with_a_plant_that_does_nothing

from parity.adapters.base import LABEL_IGNORE, Batch, Encoded, TrainCfg
from parity.checks.tier1.special_tokens import SpecialTokens, batch_plan
from parity.engine.plants import PlantedAdapter
from parity.engine.runner import evaluate
from parity.fixtures import FIXTURE_IDS, load_fixture

CHECK = SpecialTokens()
PLANTS = {"extra_leading_special", "pad_is_closing_token_masked"}


@pytest.mark.parametrize("stub", ["official", "trl_markers"])
def test_passes_on_both_stand_ins_and_catches_both_plants(request, stub):
    for fid, v in run_on_all_fixtures(CHECK, request.getfixturevalue(stub)).items():
        assert (v.result, v.self_test) == ("pass", "caught"), (fid, v.reason, v.evidence, v.plants)
        assert {p["plant"] for p in v.plants} == PLANTS
        assert v.evidence["parts"] == {"special_tokens": "pass", "padding": "pass"}
        assert v.conventions["padding_side"] == "right"
        assert v.conventions["pad_token"] == "<|endoftext|>" and v.conventions["pad_is_a_stop_token"] is True


def test_left_padding_passes(qwen3):
    for fid, v in run_on_all_fixtures(CHECK, TemplateStub(qwen3, padding_side="left")).items():
        assert (v.result, v.self_test, v.conventions["padding_side"]) == ("pass", "caught", "left"), fid


def test_a_packing_framework_has_its_padding_part_not_applicable(qwen3):
    for fid, v in run_on_all_fixtures(CHECK, TemplateStub(qwen3, batching="packed")).items():
        assert (v.result, v.self_test) == ("pass", "caught"), (fid, v.reason, v.plants)
        assert v.evidence["parts"]["padding"] == "not_applicable"
        assert v.conventions["batching"] == "packed"


@pytest.mark.parametrize("fid", FIXTURE_IDS)
def test_a_plant_that_does_nothing_makes_the_verdict_inconclusive(official, fid):
    v = evaluate(with_a_plant_that_does_nothing(CHECK), official, load_fixture(fid))
    assert (v.result, v.self_test) == ("inconclusive", "missed")


def test_batch_plan_mixes_lengths_and_pads_every_row(qwen3, official):
    fx = load_fixture("single_turn")
    partner, pad_to = batch_plan(qwen3, fx, official.declare(TrainCfg()))
    assert partner.id == "long_answer"  # the fixture whose length differs most from single_turn
    assert pad_to == 537 + 8


def _rules(v):
    return {p["rule"] for p in v.evidence["problems"]}


def test_extra_leading_special_is_reported(qwen3, official):
    fx = load_fixture("single_turn")
    plant = plant_named(CHECK, qwen3, fx, official.declare(TrainCfg()), "extra_leading_special")
    v = CHECK.run(plant.apply(official), qwen3, fx)
    assert (v.result, v.kind) == ("fail", "bug")
    lead = next(p for p in v.evidence["problems"] if p["rule"] == "leading_special_tokens")
    assert lead["expected"] == ["<|im_start|>"] and lead["got"] == ["<|endoftext|>", "<|im_start|>"]


def test_pad_equal_to_the_closing_token_reports_untrained_stop_tokens(qwen3, official):
    fx = load_fixture("multi_turn")
    plant = plant_named(CHECK, qwen3, fx, official.declare(TrainCfg()), "pad_is_closing_token_masked")
    v = CHECK.run(plant.apply(official), qwen3, fx)
    assert (v.result, v.kind) == ("fail", "bug")
    lost = [p for p in v.evidence["problems"] if p["rule"] == "stop_tokens_trained"]
    assert {p["where"] for p in lost} == {"row", "padded batch"}
    assert all(p["tokens"] == ["<|im_end|>"] * 3 for p in lost)


def _batch_edit(official, edit):
    return PlantedAdapter(official, "x", after_batch=lambda batch, convs, cfg: Batch(tuple(edit(r) for r in batch.rows)))


def test_trained_pad_positions_fail(qwen3, official):
    def train_pads(row: Encoded) -> Encoded:
        labels = tuple(t if a == 0 else label for t, a, label in zip(row.input_ids, row.attention_mask, row.labels))
        return replace(row, labels=labels)

    v = CHECK.run(_batch_edit(official, train_pads), qwen3, load_fixture("single_turn"))
    assert v.result == "fail" and "pads_untrained" in _rules(v)


def test_a_collator_that_masks_the_real_stop_token_fails(qwen3, official):
    """Id-based pad masking with pad == <|im_end|>, only in the collator: the unpadded row looks fine."""
    im_end = qwen3.tokenizer.convert_tokens_to_ids("<|im_end|>")

    def mask_im_end(row: Encoded) -> Encoded:
        return replace(row, labels=tuple(LABEL_IGNORE if t == im_end else lab for t, lab in zip(row.input_ids, row.labels)))

    v = CHECK.run(_batch_edit(official, mask_im_end), qwen3, load_fixture("single_turn"))
    assert v.result == "fail"
    assert {"padding_keeps_the_row", "stop_tokens_trained"} <= _rules(v)
    assert v.evidence["parts"] == {"special_tokens": "pass", "padding": "fail"}


def test_a_row_not_padded_to_pad_to_is_inconclusive(qwen3, official):
    def strip_pads(row: Encoded) -> Encoded:
        keep = [i for i, a in enumerate(row.attention_mask) if a == 1]
        return Encoded(*(tuple(seq[i] for i in keep) for seq in (row.input_ids, row.labels, row.attention_mask)))

    v = CHECK.run(_batch_edit(official, strip_pads), qwen3, load_fixture("single_turn"))
    assert v.result == "inconclusive" and "pad_to" in v.reason
    assert v.evidence["parts"]["padding"] == "inconclusive"
