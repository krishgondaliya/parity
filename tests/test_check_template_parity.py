"""t1.template_parity on the calibration model, against the test stand-ins.

Expected (docs/DECISIONS.md, KD-0001): TRL's Qwen3 training template differs
from Qwen3's own template on exactly the four fixtures with more than one
assistant turn, and only by the documented think block.
"""

from dataclasses import replace

import pytest
from stubs import (
    TRL_QWEN3_TRAINING_SHA256,
    run_on_all_fixtures,
    trl_template_stub,
    with_a_plant_that_does_nothing,
)

from parity.adapters.base import Dropped, TrainCfg
from parity.checks.common import official_reference, official_template
from parity.checks.known_differences import (
    insert_think_block_where_official_has_none,
    load_registry,
    match_known_difference,
)
from parity.checks.tier1.template_parity import TemplateParity
from parity.engine.plants import PlantedAdapter
from parity.engine.runner import evaluate
from parity.fixtures import FIXTURE_IDS, load_fixture
from parity.references.template import template_sha256
from parity.targets import CALIBRATION

CHECK = TemplateParity()
DIFFERS_UNDER_KD_0001 = {"multi_turn", "tool_call", "reasoning", "empty_assistant"}


def test_passes_on_the_models_own_template_and_catches_both_plants(official):
    verdicts = run_on_all_fixtures(CHECK, official)
    for fid, v in verdicts.items():
        assert (v.result, v.self_test) == ("pass", "caught"), (fid, v.reason, v.plants)
        assert {p["plant"] for p in v.plants} == {"drop_newline", "swap_stop_token", "rstrip_render"}


def test_trl_template_is_kd_0001_on_exactly_the_documented_fixtures(qwen3, trl_qwen3_template):
    trl = trl_template_stub(qwen3, trl_qwen3_template, name="trl", version="1.18.2")
    verdicts = run_on_all_fixtures(CHECK, trl)
    differences = {fid for fid, v in verdicts.items() if v.result == "fail"}
    assert differences == DIFFERS_UNDER_KD_0001
    for fid, v in verdicts.items():
        assert v.self_test == "caught", (fid, v.plants)
        if fid in DIFFERS_UNDER_KD_0001:
            assert (v.kind, v.known_difference) == ("difference", "KD-0001")
        else:
            assert v.result == "pass"


def test_the_registry_never_covers_another_framework_name(trl_markers):
    """The same tokens from a framework the entry doesn't name stay a bug."""
    verdicts = run_on_all_fixtures(CHECK, trl_markers)
    assert {fid for fid, v in verdicts.items() if v.result == "fail" and v.kind == "bug"} == DIFFERS_UNDER_KD_0001


def test_the_registry_never_covers_an_unverified_version(qwen3, trl_qwen3_template):
    trl_next = trl_template_stub(qwen3, trl_qwen3_template, name="trl", version="1.18.3")
    v = evaluate(CHECK, trl_next, load_fixture("multi_turn"))
    assert (v.result, v.kind, v.known_difference) == ("fail", "bug", None)


def test_a_near_miss_of_kd_0001_is_a_bug(qwen3, trl_qwen3_template):
    """KD-0001's divergence plus one dropped newline is not KD-0001."""
    trl = trl_template_stub(qwen3, trl_qwen3_template, name="trl", version="1.18.2")
    v = evaluate(CHECK, trl, load_fixture("multi_turn"))
    drop = next(p for p in v.plants if p["plant"] == "drop_newline")
    assert (drop["result"], drop["kind"]) == ("fail", "bug")


def test_registry_matcher_needs_the_exact_token_sequence(qwen3, trl_qwen3_template):
    fx = load_fixture("reasoning")
    official = official_template(qwen3)
    good = insert_think_block_where_official_has_none(qwen3, fx, official)
    kwargs = dict(
        check_id=CHECK.id,
        framework="trl",
        version="1.18.2",
        release=qwen3,
        conv=fx,
        official_template=official,
        declared_template=trl_qwen3_template,
        mode="thinking",
    )
    assert match_known_difference(got_ids=good, **kwargs).id == "KD-0001"
    assert match_known_difference(got_ids=good, **{**kwargs, "mode": "non_thinking"}).id == "KD-0001"
    assert match_known_difference(got_ids=good[:-1], **kwargs) is None
    assert match_known_difference(got_ids=good, **{**kwargs, "declared_template": official}) is None
    assert match_known_difference(got_ids=good, **{**kwargs, "check_id": "t1.assistant_mask"}) is None


def test_registry_entries_are_exact_and_match_the_pinned_templates(qwen3):
    registry = {e.id: e for e in load_registry()}
    assert sorted(registry) == ["KD-0001", "KD-0002", "KD-0003", "KD-0004", "KD-0005"]
    kd = registry["KD-0001"]
    assert kd.versions == ("1.18.2",) and kd.serving_modes == ("thinking", "non_thinking")
    assert kd.model_template_sha256 == template_sha256(official_template(qwen3))
    assert kd.framework_template_sha256 == TRL_QWEN3_TRAINING_SHA256


@pytest.mark.parametrize("fid", FIXTURE_IDS)
def test_a_plant_that_does_nothing_makes_the_verdict_inconclusive(official, fid):
    v = evaluate(with_a_plant_that_does_nothing(CHECK), official, load_fixture(fid))
    assert (v.result, v.self_test) == ("inconclusive", "missed")
    assert v.evidence["clean_result"] == "pass"


def test_dropping_the_conversation_without_a_limit_is_a_bug(official):
    dropper = PlantedAdapter(official, "drop", after_encode=lambda out, conv, cfg: Dropped("skipped"))
    v = CHECK.run(dropper, official.release, load_fixture("single_turn"))
    assert (v.result, v.kind) == ("fail", "bug")


def test_fail_evidence_shows_the_first_divergence(qwen3, official):
    fx = load_fixture("system_prompt")
    planted = next(p for p in CHECK.plants(qwen3, fx, official.declare) if p.name == "swap_stop_token")
    v = CHECK.run(planted.apply(official), qwen3, fx)
    ref = official_reference(qwen3, fx)
    first_stop = next(i for i, t in enumerate(ref.input_ids) if t in qwen3.stop_token_ids)
    assert v.evidence["first_divergence"] == first_stop
    assert v.evidence["expected"][0] == "<|im_end|>" and v.evidence["got"][0] == "<|endoftext|>"
    assert len(v.evidence["expected"]) == 8


def test_verdict_records_the_reference_and_versions(official):
    v = evaluate(CHECK, official, load_fixture("single_turn"))
    assert v.reference["template_source"] == "model"
    assert v.reference["template_sha256"] == v.reference["official_template_sha256"]
    assert {"transformers", "tokenizers", "trl"} <= set(v.reference)
    assert v.model == CALIBRATION.spec
    assert v.framework == "stub-official@0"


def test_a_contradicted_template_declaration_is_an_adapter_error(qwen3, official, trl_qwen3_template):
    """Renders exactly Qwen3's own template but declares TRL's: an adapter error, not a pass."""
    lying = PlantedAdapter(official, "x", on_declare=lambda d: replace(d, chat_template=trl_qwen3_template))
    v = CHECK.run(lying, qwen3, load_fixture("multi_turn"))
    assert v.result == "inconclusive" and v.reason.startswith("declared/actual mismatch")
    assert v.declaration["verified"]["chat_template"].startswith("contradicted")


def test_the_trl_stand_ins_declaration_is_consistent_even_where_it_differs(qwen3, trl_qwen3_template):
    trl = trl_template_stub(qwen3, trl_qwen3_template, name="trl", version="1.18.2")
    v = evaluate(CHECK, trl, load_fixture("multi_turn"))
    assert v.known_difference == "KD-0001"
    assert v.declaration["verified"]["chat_template"] == "consistent"


def test_rstrip_drops_only_the_final_separator(qwen3, official):
    fx = load_fixture("single_turn")
    planted = next(p for p in CHECK.plants(qwen3, fx, official.declare) if p.name == "rstrip_render")
    v = CHECK.run(planted.apply(official), qwen3, fx)
    ref = official_reference(qwen3, fx)
    assert (v.result, v.kind) == ("fail", "bug")
    assert (v.evidence["first_divergence"], v.evidence["got_len"]) == (len(ref.input_ids) - 1, len(ref.input_ids) - 1)
    assert v.evidence["expected"] == ["\n"]
