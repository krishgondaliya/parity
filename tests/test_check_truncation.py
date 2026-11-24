"""t1.truncation on the calibration model, against the test stand-ins.

long_answer renders to 537 tokens on Qwen3 and has max_len 128; the other
seven fixtures run at their exact length (the boundary).
"""

from dataclasses import replace

import pytest
from stubs import TemplateStub, plant_named, run_on_all_fixtures, with_a_plant_that_does_nothing

from parity.adapters.base import TrainCfg
from parity.checks.common import official_reference
from parity.checks.tier1.truncation import Truncation
from parity.engine.plants import PlantedAdapter
from parity.engine.runner import evaluate
from parity.fixtures import FIXTURE_IDS, load_fixture

CHECK = Truncation()

PLANTS_ON_LONG_ANSWER = {
    "right": {"off_by_one", "lose_answer_labels", "reappend_stop"},
    "left": {"off_by_one", "lose_answer_labels"},
    "drop": {"lose_answer_labels"},
}


@pytest.mark.parametrize("mode", ["right", "left", "drop"])
def test_every_declared_mode_passes_and_catches_its_plants(qwen3, mode):
    for fid, v in run_on_all_fixtures(CHECK, TemplateStub(qwen3, truncation_mode=mode)).items():
        assert (v.result, v.self_test) == ("pass", "caught"), (fid, v.reason, v.evidence, v.plants)
        assert v.conventions["truncation_mode"] == mode
        plants = {p["plant"] for p in v.plants}
        if fid == "long_answer":
            assert v.conventions["cut"] is True and v.conventions["max_len"] == 128
            assert plants == PLANTS_ON_LONG_ANSWER[mode]
        else:
            assert v.conventions["cut"] is False
            assert plants == {"off_by_one"}


def test_passes_on_trls_template(trl_markers):
    for fid, v in run_on_all_fixtures(CHECK, trl_markers).items():
        assert (v.result, v.self_test) == ("pass", "caught"), (fid, v.reason, v.plants)


@pytest.mark.parametrize("fid", FIXTURE_IDS)
def test_a_plant_that_does_nothing_makes_the_verdict_inconclusive(official, fid):
    v = evaluate(with_a_plant_that_does_nothing(CHECK), official, load_fixture(fid))
    assert (v.result, v.self_test) == ("inconclusive", "missed")


def test_the_kept_answer_is_trained_after_a_right_cut(official):
    v = evaluate(CHECK, official, load_fixture("long_answer"))
    assert v.result == "pass"
    assert v.evidence["supervised_tokens"] == 128 - 23  # the answer starts at token 23


@pytest.mark.parametrize(
    "actual, declared",
    [("left", "right"), ("right", "left"), ("right", "drop"), ("drop", "right")],
)
def test_output_that_is_exactly_another_mode_is_an_adapter_error(qwen3, actual, declared):
    """The declaration is a claim: behavior that is exactly another mode contradicts it (rule 4)."""
    stub = TemplateStub(qwen3, truncation_mode=actual)
    lying = PlantedAdapter(stub, "wrong-mode", on_declare=lambda d: replace(d, truncation_mode=declared))
    v = CHECK.run(lying, qwen3, load_fixture("long_answer"))
    assert v.result == "inconclusive" and v.reason.startswith("declared/actual mismatch"), v.evidence
    assert v.declaration["truncation_mode"] == declared
    assert v.declaration["verified"]["truncation_mode"] == (
        f"contradicted: the framework's row is exactly what {actual!r} truncation gives"
    )


def test_output_that_matches_no_mode_fails(qwen3, official):
    v = _planted(qwen3, official, "long_answer", "off_by_one")
    assert (v.result, v.kind) == ("fail", "bug")
    assert v.declaration["verified"]["truncation_mode"].startswith("mismatch")


def test_the_mode_is_verified_only_when_something_is_cut(official):
    cut = evaluate(CHECK, official, load_fixture("long_answer"))
    boundary = evaluate(CHECK, official, load_fixture("single_turn"))
    assert cut.declaration["verified"]["truncation_mode"] == "consistent"
    assert boundary.declaration["verified"]["truncation_mode"].startswith("unverified")


def _planted(qwen3, adapter, fid, name):
    fx = load_fixture(fid)
    plant = plant_named(CHECK, qwen3, fx, adapter.declare, name)
    return CHECK.run(plant.apply(adapter), qwen3, fx)


def test_off_by_one_at_the_boundary_loses_the_last_token(qwen3, official):
    v = _planted(qwen3, official, "single_turn", "off_by_one")
    assert (v.result, v.evidence["expected_len"], v.evidence["got_len"]) == ("fail", 25, 24)


def test_lost_answer_labels_are_reported(qwen3, official):
    v = _planted(qwen3, official, "long_answer", "lose_answer_labels")
    assert v.result == "fail"
    assert v.evidence["counts"]["must_train_unsupervised"] == 128 - 23


def test_reappended_stop_token_is_reported(qwen3, official):
    v = _planted(qwen3, official, "long_answer", "reappend_stop")
    assert v.result == "fail" and v.evidence["first_divergence"] == 127
    assert v.evidence["got"] == ["<|im_end|>"]


def test_a_drop_mode_framework_that_keeps_a_label_less_row_fails(qwen3):
    v = _planted(qwen3, TemplateStub(qwen3, truncation_mode="drop"), "long_answer", "lose_answer_labels")
    assert (v.result, v.kind) == ("fail", "bug")


def test_tokens_that_differ_without_a_limit_are_inconclusive(qwen3, official, trl_qwen3_template):
    lying = PlantedAdapter(official, "x", on_declare=lambda d: replace(d, chat_template=trl_qwen3_template))
    v = CHECK.run(lying, qwen3, load_fixture("multi_turn"))
    assert v.result == "inconclusive" and v.reason.startswith("declared/actual mismatch")
    assert v.declaration["verified"]["chat_template"].startswith("contradicted")


def test_long_answer_reference_is_long_enough_to_cut(qwen3):
    ref = official_reference(qwen3, load_fixture("long_answer"))
    assert len(ref.input_ids) == 537 and ref.spans[0].token_start == 23


def test_a_cut_that_leaves_zero_trainable_tokens_passes_but_is_flagged(official):
    """long_answer at max_len 20: the prompt (23 tokens) doesn't even fit, so a right cut keeps no answer."""
    fx = replace(load_fixture("long_answer"), max_len=20)
    v = evaluate(CHECK, official, fx)
    assert (v.result, v.self_test) == ("pass", "caught"), (v.reason, v.plants)
    assert v.flags == ["zero_trainable_tokens"] and v.evidence["supervised_tokens"] == 0


class _DeclaresByConfig:
    """Declares 'right' with no length limit and 'left' with one, and really cuts left."""

    def __init__(self, inner):
        self.inner, self.name, self.version = inner, inner.name, inner.version
        self.release, self.env_lock = inner.release, inner.env_lock

    def declare(self, cfg):
        return replace(self.inner.declare(cfg), truncation_mode="left" if cfg.max_len is not None else "right")

    def encode(self, conv, cfg):
        return self.inner.encode(conv, cfg)

    def encode_batch(self, convs, cfg):
        return self.inner.encode_batch(convs, cfg)


def test_the_mode_comes_from_the_declaration_for_the_cut_config(qwen3):
    v = evaluate(CHECK, _DeclaresByConfig(TemplateStub(qwen3, truncation_mode="left")), load_fixture("long_answer"))
    assert (v.result, v.self_test) == ("pass", "caught"), (v.reason, v.plants)
    assert v.conventions["truncation_mode"] == "left"
    assert v.declaration["cfg"]["max_len"] == 128 and v.declaration["truncation_mode"] == "left"
    [uncut] = v.declaration["other_calls"]
    assert uncut["cfg"]["max_len"] is None and uncut["truncation_mode"] == "right"


def test_a_contradicted_mode_is_flagged_for_triage(qwen3):
    stub = TemplateStub(qwen3, truncation_mode="left")
    lying = PlantedAdapter(stub, "wrong-mode", on_declare=lambda d: replace(d, truncation_mode="right"))
    v = CHECK.run(lying, qwen3, load_fixture("long_answer"))
    assert v.flags == ["declared_actual_mismatch"] and v.evidence["triage"] == "declared/actual mismatch"


def test_a_loss_only_fail_confirms_the_mode_instead_of_denying_it(qwen3, official):
    """Found by the P-0003 skeptic: a row whose tokens are exactly the declared cut, with the wrong loss, used to be
    recorded as "the row matches no truncation mode". The tokens confirm the mode; the fail is about the loss."""
    cut = _planted(qwen3, official, "long_answer", "lose_answer_labels")
    assert (cut.result, cut.kind) == ("fail", "bug")
    assert cut.declaration["verified"]["truncation_mode"] == "consistent"
    no_loss = PlantedAdapter(
        official, "no loss", after_encode=lambda out, conv, cfg: replace(out, labels=(-100,) * len(out.labels))
    )
    boundary = CHECK.run(no_loss, qwen3, load_fixture("single_turn"))
    assert (boundary.result, boundary.kind) == ("fail", "bug")
    assert boundary.declaration["verified"]["truncation_mode"].startswith("unverified")
