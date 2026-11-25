"""t1.special_tokens on the calibration model, against the test stand-ins.

Qwen3 has no BOS; its pad token <|endoftext|> is also one of its stop tokens
(generation_config), while assistant turns close with <|im_end|>.
"""

from dataclasses import replace

import pytest
from stubs import TemplateStub, plant_named, run_on_all_fixtures, with_a_plant_that_does_nothing

from parity.adapters.base import LABEL_IGNORE, Batch, Encoded, TrainCfg
from parity.checks.tier1 import special_tokens as special_tokens_module
from parity.checks.tier1.special_tokens import SpecialTokens, _PaddingResult, batch_plan
from parity.engine.plants import PlantedAdapter
from parity.engine.runner import evaluate
from parity.fixtures import FIXTURE_IDS, load_fixture

CHECK = SpecialTokens()
PLANTS = {"extra_leading_special", "pad_is_closing_token_masked", "closing_token_masked_per_row", "pad_to_ignored"}
ALL_PASS = {"special_tokens": "pass", "padding_width": "pass", "pad_handling": "pass"}


@pytest.mark.parametrize("stub", ["official", "trl_markers"])
def test_passes_on_both_stand_ins_and_catches_both_plants(request, stub):
    for fid, v in run_on_all_fixtures(CHECK, request.getfixturevalue(stub)).items():
        assert (v.result, v.self_test) == ("pass", "caught"), (fid, v.reason, v.evidence, v.plants)
        assert {p["plant"] for p in v.plants} == PLANTS
        assert v.evidence["parts"] == ALL_PASS
        assert v.conventions["padding_side"] == "right"
        assert v.conventions["pad_token"] == "<|endoftext|>" and v.conventions["pad_is_a_stop_token"] is True


def test_left_padding_passes(qwen3):
    for fid, v in run_on_all_fixtures(CHECK, TemplateStub(qwen3, padding_side="left")).items():
        assert (v.result, v.self_test, v.conventions["padding_side"]) == ("pass", "caught", "left"), fid


def test_a_packing_framework_leaves_the_padding_part_inconclusive(qwen3):
    """Phase 0 never asks for packing; until the Phase 2 packing-isolation check exists, it can't pass."""
    for fid, v in run_on_all_fixtures(CHECK, TemplateStub(qwen3, batching="packed")).items():
        assert (v.result, v.self_test) == ("inconclusive", "caught"), (fid, v.reason, v.plants)
        assert v.evidence["parts"] == {"special_tokens": "pass", "padding_width": "inconclusive", "pad_handling": "inconclusive"}
        assert "Phase 2" in v.reason and v.conventions["batching"] == "packed"


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
    plant = plant_named(CHECK, qwen3, fx, official.declare, "extra_leading_special")
    v = CHECK.run(plant.apply(official), qwen3, fx)
    assert (v.result, v.kind) == ("fail", "bug")
    lead = next(p for p in v.evidence["problems"] if p["rule"] == "leading_special_tokens")
    assert lead["expected"] == ["<|im_start|>"] and lead["got"] == ["<|endoftext|>", "<|im_start|>"]


@pytest.mark.parametrize(
    "name, where",
    [("pad_is_closing_token_masked", {"padded batch"}), ("closing_token_masked_per_row", {"row", "padded batch"})],
)
def test_pad_equal_to_the_closing_token_reports_untrained_stop_tokens(qwen3, official, name, where):
    fx = load_fixture("multi_turn")
    plant = plant_named(CHECK, qwen3, fx, official.declare, name)
    v = CHECK.run(plant.apply(official), qwen3, fx)
    assert (v.result, v.kind) == ("fail", "bug")
    lost = [p for p in v.evidence["problems"] if p["rule"] == "stop_tokens_trained"]
    assert {p["where"] for p in lost} == where
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
    assert v.evidence["parts"] == {"special_tokens": "pass", "padding_width": "pass", "pad_handling": "fail"}


def _drops_pad_to(adapter):
    """A framework whose collator pads only to the longest row, as P-0005's swapped collator does."""
    return PlantedAdapter(adapter, "x", after_batch=lambda batch, convs, cfg: special_tokens_module._trim_to_longest(batch))


@pytest.mark.parametrize("fid", ["single_turn", "long_answer"])  # the shorter and the longer row of one batch
def test_a_dropped_width_fails_rule_2a_and_pads_are_still_judged_at_the_produced_width(qwen3, official, fid):
    v = evaluate(CHECK, _drops_pad_to(official), load_fixture(fid))
    assert (v.result, v.kind, v.self_test) == ("fail", "bug", "caught"), (v.reason, v.plants)
    assert v.evidence["parts"] == {"special_tokens": "pass", "padding_width": "fail", "pad_handling": "pass"}
    [width] = v.evidence["problems"]
    assert (width["rule"], width["pad_to"], width["widths"]) == ("pad_to_honored", 545, [537])
    assert all(p["caught"] for p in v.plants)  # every plant, pad handling's included, is caught at width 537


def test_bad_pads_are_caught_even_when_the_width_is_dropped(qwen3, official):
    def train_pads(row: Encoded) -> Encoded:
        return replace(row, labels=tuple(t if a == 0 else lab for t, a, lab in zip(row.input_ids, row.attention_mask, row.labels)))

    dropped = _drops_pad_to(official)
    v = CHECK.run(PlantedAdapter(dropped, "y", after_batch=lambda b, c, cfg: Batch(tuple(train_pads(r) for r in b.rows))),
                  qwen3, load_fixture("single_turn"))  # fmt: skip
    assert v.evidence["parts"] == {"special_tokens": "pass", "padding_width": "fail", "pad_handling": "fail"}
    assert {"pad_to_honored", "pads_untrained"} <= _rules(v)


def test_the_partner_rows_pads_are_judged_too(qwen3, official):
    """single_turn is the shorter row; long_answer, its partner, is padded only when the width is honored."""
    def train_partner_pads(batch, convs, cfg):
        a, b = batch.rows
        b = replace(b, labels=tuple(t if m == 0 else lab for t, m, lab in zip(b.input_ids, b.attention_mask, b.labels)))
        return Batch((a, b))

    v = CHECK.run(PlantedAdapter(official, "x", after_batch=train_partner_pads), qwen3, load_fixture("single_turn"))
    trained = [p for p in v.evidence["problems"] if p["rule"] == "pads_untrained"]
    assert v.result == "fail" and [p["row"] for p in trained] == ["long_answer"]


def test_packed_rows_when_padding_was_declared_are_an_adapter_error(qwen3, official):
    def pack(batch, convs, cfg):
        a, b = batch.rows
        keep_a = [i for i, m in enumerate(a.attention_mask) if m]
        keep_b = [i for i, m in enumerate(b.attention_mask) if m]
        ids = tuple(a.input_ids[i] for i in keep_a) + tuple(b.input_ids[i] for i in keep_b)
        labels = tuple(a.labels[i] for i in keep_a) + tuple(b.labels[i] for i in keep_b)
        positions = tuple(range(len(keep_a))) + tuple(range(len(keep_b)))
        return Batch((Encoded(ids, labels, (1,) * len(ids), positions),))

    v = CHECK.run(PlantedAdapter(official, "x", after_batch=pack), qwen3, load_fixture("single_turn"))
    assert v.result == "inconclusive" and v.reason.startswith("declared/actual mismatch")
    assert v.declaration["other_calls"][0]["verified"]["batching"].startswith("contradicted")
    assert "declared_actual_mismatch" in v.flags and v.evidence["triage"] == "declared/actual mismatch"


def test_pads_that_are_all_another_token_are_an_adapter_error(qwen3, official):
    im_start = qwen3.tokenizer.convert_tokens_to_ids("<|im_start|>")

    def repad(row: Encoded) -> Encoded:
        return replace(row, input_ids=tuple(t if a else im_start for t, a in zip(row.input_ids, row.attention_mask)))

    v = CHECK.run(_batch_edit(official, repad), qwen3, load_fixture("single_turn"))
    assert v.result == "inconclusive" and v.reason.startswith("declared/actual mismatch")
    assert v.declaration["other_calls"][0]["verified"]["pad_token_id"].startswith("contradicted")
    assert "declared_actual_mismatch" in v.flags


def test_pads_with_several_different_tokens_fail(qwen3, official):
    def mixed(row: Encoded) -> Encoded:
        pads = [i for i, a in enumerate(row.attention_mask) if not a]
        ids = list(row.input_ids)
        ids[pads[0]] = qwen3.tokenizer.convert_tokens_to_ids("<|im_start|>")
        return replace(row, input_ids=tuple(ids))

    v = CHECK.run(_batch_edit(official, mixed), qwen3, load_fixture("single_turn"))
    assert (v.result, v.kind) == ("fail", "bug") and "pad_token" in _rules(v)


def test_each_encode_call_records_the_declaration_for_its_own_config(official):
    v = evaluate(CHECK, official, load_fixture("single_turn"))
    row, [batch] = v.declaration, v.declaration["other_calls"]
    assert row["cfg"]["pad_to"] is None and batch["cfg"]["pad_to"] == 537 + 8
    assert row["verified"]["chat_template"] == "consistent"
    assert (batch["verified"]["batching"], batch["verified"]["pad_token_id"]) == ("consistent", "consistent")
    assert v.flags == []


# Each part of the check is guarded by a planted fault of its own (found by the M2 skeptic review:
# the pad fault used to be caught by the row rule alone, so the padding part was unguarded).


@pytest.mark.parametrize(
    "name, parts",
    [
        ("pad_is_closing_token_masked", {"special_tokens": "pass", "padding_width": "pass", "pad_handling": "fail"}),
        ("pad_to_ignored", {"special_tokens": "pass", "padding_width": "fail", "pad_handling": "pass"}),
    ],
)
def test_each_padding_rule_is_guarded_by_its_own_planted_fault(qwen3, official, name, parts):
    fx = load_fixture("single_turn")
    v = CHECK.run(plant_named(CHECK, qwen3, fx, official.declare, name).apply(official), qwen3, fx)
    assert v.evidence["parts"] == parts


def test_without_the_padding_rules_both_padding_plants_are_missed(qwen3, official, monkeypatch):
    monkeypatch.setattr(SpecialTokens, "_padding", lambda self, *args: _PaddingResult(side="right"))
    v = evaluate(CHECK, official, load_fixture("single_turn"))
    assert (v.result, v.self_test) == ("inconclusive", "missed")
    assert {p["plant"] for p in v.plants if not p["caught"]} == {"pad_is_closing_token_masked", "pad_to_ignored"}


def test_the_per_row_pad_fault_guards_the_row_rule_when_the_framework_packs(qwen3, monkeypatch):
    packed = TemplateStub(qwen3, batching="packed")
    fx = load_fixture("multi_turn")
    names = {p.name for p in CHECK.plants(qwen3, fx, packed.declare)}
    assert names == {"extra_leading_special", "closing_token_masked_per_row"}  # no collator fault without padding

    original = special_tokens_module._untrained_stop_tokens
    monkeypatch.setattr(
        special_tokens_module,
        "_untrained_stop_tokens",
        lambda tokenizer, release, ref, row, where: [] if where == "row" else original(tokenizer, release, ref, row, where),
    )
    v = evaluate(CHECK, packed, fx)
    assert (v.result, v.self_test) == ("inconclusive", "missed")
