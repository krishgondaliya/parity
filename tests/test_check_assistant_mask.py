"""t1.assistant_mask on the calibration model, against the test stand-ins."""

from dataclasses import replace

import pytest
from stubs import (
    TemplateStub,
    markers_moved_onto_the_header,
    plant_named,
    run_on_all_fixtures,
    with_a_plant_that_does_nothing,
)

from parity.adapters.base import TrainCfg
from parity.checks.common import official_reference
from parity.checks.tier1.assistant_mask import AssistantMask
from parity.engine.plants import PlantedAdapter
from parity.engine.runner import evaluate
from parity.fixtures import FIXTURE_IDS, load_fixture
from parity.references.template import Zone

CHECK = AssistantMask()
PLANTS = {"unmask_prompt", "zero_mask", "shift_labels_left", "late_start"}


def test_passes_on_the_models_own_template_and_catches_every_plant(official):
    for fid, v in run_on_all_fixtures(CHECK, official).items():
        assert (v.result, v.self_test) == ("pass", "caught"), (fid, v.reason, v.plants)
        assert {p["plant"] for p in v.plants} == PLANTS
        assert v.conventions["label_convention"] == "aligned"
        assert v.conventions["neutral"] == "not_trained"


def test_passes_on_trls_template_where_the_markers_also_train_the_neutral_newline(trl_markers):
    for fid, v in run_on_all_fixtures(CHECK, trl_markers).items():
        assert (v.result, v.self_test) == ("pass", "caught"), (fid, v.reason, v.plants)
        assert v.conventions["neutral"] == "trained"
        assert v.conventions["neutral_trained"] == len(load_fixture(fid).assistant_indices)


@pytest.mark.parametrize("fid", FIXTURE_IDS)
def test_a_plant_that_does_nothing_makes_the_verdict_inconclusive(official, fid):
    v = evaluate(with_a_plant_that_does_nothing(CHECK), official, load_fixture(fid))
    assert (v.result, v.self_test) == ("inconclusive", "missed")


def _planted_run(release, adapter, fid, name):
    fx = load_fixture(fid)
    plant = plant_named(CHECK, release, fx, adapter.declare(TrainCfg()), name)
    return CHECK.run(plant.apply(adapter), release, fx)


def test_unmask_prompt_reports_the_user_turn_as_trained(qwen3, official):
    v = _planted_run(qwen3, official, "system_prompt", "unmask_prompt")
    assert (v.result, v.kind) == ("fail", "bug")
    shown = "".join(p["token"] for p in v.evidence["must_not_train_supervised"])
    assert shown.startswith("<|im_start|>system\nYou are")


def test_zero_mask_reports_every_must_train_token(qwen3, official):
    v = _planted_run(qwen3, official, "multi_turn", "zero_mask")
    ref = official_reference(qwen3, load_fixture("multi_turn"))
    assert v.evidence["counts"]["must_train_unsupervised"] == sum(z == Zone.MUST_TRAIN for z in ref.zones)
    assert v.evidence["supervised_tokens"] == 0


def test_shifted_labels_are_caught_as_label_mismatches(qwen3, official):
    v = _planted_run(qwen3, official, "single_turn", "shift_labels_left")
    assert v.evidence["counts"]["label_mismatch"] > 0


def test_late_start_reports_the_first_token_of_each_span(qwen3, official):
    fx = load_fixture("empty_assistant")
    v = _planted_run(qwen3, official, "empty_assistant", "late_start")
    ref = official_reference(qwen3, fx)
    assert [p["index"] for p in v.evidence["must_train_unsupervised"]] == [s.token_start for s in ref.spans]


def test_label_convention_is_recorded(qwen3):
    stub = TemplateStub(qwen3, label_convention="pre_shifted")
    v = evaluate(CHECK, stub, load_fixture("single_turn"))
    assert v.result == "pass" and v.conventions["label_convention"] == "pre_shifted"


def test_tokens_that_differ_from_the_declared_template_are_inconclusive(qwen3, official, trl_qwen3_template):
    """Renders with the official template but declares TRL's: the mask can't be lined up."""
    lying = PlantedAdapter(official, "wrong-declaration", on_declare=lambda d: replace(d, chat_template=trl_qwen3_template))
    v = CHECK.run(lying, qwen3, load_fixture("multi_turn"))
    assert v.result == "inconclusive" and "t1.template_parity" in v.reason
    assert v.evidence["first_divergence"] is not None


def test_a_template_whose_markers_are_wrong_is_inconclusive_and_flagged(qwen3, trl_qwen3_template):
    bad = markers_moved_onto_the_header(trl_qwen3_template)
    stub = TemplateStub(qwen3, chat_template=bad, template_source="test:bad-markers", labels_from="markers")
    v = evaluate(CHECK, stub, load_fixture("single_turn"))
    assert v.result == "inconclusive"
    assert v.evidence["triage"] == "possible model-template finding"
