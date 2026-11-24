"""t1.end_of_turn on the calibration model, against the test stand-ins."""

import pytest
from stubs import plant_named, run_on_all_fixtures, with_a_plant_that_does_nothing

from parity.adapters.base import Encoded, TrainCfg
from parity.checks.common import official_reference
from parity.checks.tier1.end_of_turn import EndOfTurn
from parity.engine.plants import encoded_plant
from parity.engine.runner import evaluate
from parity.fixtures import FIXTURE_IDS, load_fixture

CHECK = EndOfTurn()


@pytest.mark.parametrize("stub", ["official", "trl_markers"])
def test_passes_on_both_stand_ins_and_catches_the_plant(request, stub):
    for fid, v in run_on_all_fixtures(CHECK, request.getfixturevalue(stub)).items():
        assert (v.result, v.self_test) == ("pass", "caught"), (fid, v.reason, v.plants)
        assert [p["plant"] for p in v.plants] == ["mask_closing_tokens"]
        assert {t["status"] for t in v.evidence["turns"]} == {"trained"}
        assert len(v.evidence["turns"]) == len(load_fixture(fid).assistant_indices)


@pytest.mark.parametrize("fid", FIXTURE_IDS)
def test_a_plant_that_does_nothing_makes_the_verdict_inconclusive(official, fid):
    v = evaluate(with_a_plant_that_does_nothing(CHECK), official, load_fixture(fid))
    assert (v.result, v.self_test) == ("inconclusive", "missed")


def test_the_plant_reports_every_turn_untrained(qwen3, official):
    fx = load_fixture("multi_turn")
    plant = plant_named(CHECK, qwen3, fx, official.declare(TrainCfg()), "mask_closing_tokens")
    v = CHECK.run(plant.apply(official), qwen3, fx)
    assert (v.result, v.kind) == ("fail", "bug")
    assert [t["status"] for t in v.evidence["turns"]] == ["not_trained"] * 3
    assert [t["message_index"] for t in v.evidence["turns"]] == fx.assistant_indices


def _edit_at(position: int, token: int | None):
    """Replace (or, with token=None, delete) one token, keeping its label aligned."""

    def mutate(row: Encoded) -> Encoded:
        ids, labels, attn = list(row.input_ids), list(row.labels), list(row.attention_mask)
        if token is None:
            del ids[position], labels[position], attn[position]
        else:
            if labels[position] == ids[position]:
                labels[position] = token
            ids[position] = token
        return Encoded(tuple(ids), tuple(labels), tuple(attn))

    return mutate


def test_a_dropped_closing_token_fails_and_later_turns_are_unchecked(qwen3, official):
    fx = load_fixture("multi_turn")
    first_close = official_reference(qwen3, fx).spans[0].token_end - 1
    dropper = encoded_plant("x", "x", _edit_at(first_close, None)).apply(official)
    v = CHECK.run(dropper, qwen3, fx)
    assert (v.result, v.kind) == ("fail", "bug")
    assert [t["status"] for t in v.evidence["turns"]] == ["missing_or_replaced", "unchecked", "unchecked"]


def test_a_swapped_closing_token_fails(qwen3, official):
    fx = load_fixture("single_turn")
    close = official_reference(qwen3, fx).spans[0].token_end - 1
    endoftext = qwen3.tokenizer.convert_tokens_to_ids("<|endoftext|>")
    v = CHECK.run(encoded_plant("x", "x", _edit_at(close, endoftext)).apply(official), qwen3, fx)
    [turn] = v.evidence["turns"]
    assert (v.result, turn["status"], turn["got_token"]) == ("fail", "missing_or_replaced", "<|endoftext|>")


def test_a_divergence_inside_the_content_is_inconclusive(qwen3, official):
    fx = load_fixture("single_turn")
    start = official_reference(qwen3, fx).spans[0].token_start
    other = qwen3.tokenizer.convert_tokens_to_ids("Ġthe")
    assert isinstance(other, int) and other != qwen3.tokenizer.unk_token_id
    v = CHECK.run(encoded_plant("x", "x", _edit_at(start + 1, other)).apply(official), qwen3, fx)
    assert v.result == "inconclusive" and "t1.template_parity" in v.reason
    assert v.evidence["turns"][0]["status"] == "unchecked"
