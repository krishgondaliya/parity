"""All six Tier 1 checks together on the calibration model, against both test stand-ins.

A stand-in run, not calibration: real calibration (M4) runs real adapters.
This proves the checks agree with each other and the engine: every applicable
verdict passes (or is KD-0001 on the TRL template), every planted fault is
caught, and the summary counts what each check ran on.
"""

import sys

import pytest
from stubs import trl_template_stub

from parity.adapters.base import Encoded

from parity.checks.tier1 import TIER1_CHECKS
from parity.engine.runner import evaluate
from parity.engine.summary import summarize
from parity.engine.verdict import read_verdict, write_verdict
from parity.fixtures import FIXTURE_IDS, load_fixture

EXPECTED_IDS = [
    "t1.template_parity",
    "t1.assistant_mask",
    "t1.end_of_turn",
    "t1.special_tokens",
    "t1.truncation",
    "t1.multiturn_prefix",
]


def test_the_suite_is_the_six_blueprint_checks():
    assert [c.id for c in TIER1_CHECKS] == EXPECTED_IDS


def _run_all(adapter):
    return [evaluate(check, adapter, load_fixture(fid)) for check in TIER1_CHECKS for fid in FIXTURE_IDS]


def test_official_template_stand_in_is_fully_green(official, tmp_path):
    verdicts = _run_all(official)
    assert len(verdicts) == 48
    for v in verdicts:
        if v.result == "not_applicable":
            assert v.check == "t1.multiturn_prefix"
            continue
        assert (v.result, v.self_test) == ("pass", "caught"), (v.check, v.fixture, v.reason, v.plants)
        assert all(p["caught"] for p in v.plants) and v.plants

    cells = {c.check: c for c in summarize(verdicts)}
    assert {c.result for c in cells.values()} == {"pass"}
    assert {check: c.ran_on for check, c in cells.items()} == {
        **{check: 8 for check in EXPECTED_IDS},
        "t1.multiturn_prefix": 4,
    }

    for v in verdicts:  # every verdict survives the JSON round trip
        assert read_verdict(write_verdict(v, tmp_path)) == v


def test_trl_template_stand_in_is_green_with_kd_0001_only(qwen3, trl_qwen3_template):
    trl = trl_template_stub(qwen3, trl_qwen3_template, name="trl", version="1.18.2")
    verdicts = _run_all(trl)
    for v in verdicts:
        if v.result == "not_applicable":
            continue
        assert v.self_test == "caught", (v.check, v.fixture, v.plants)
        if v.result == "fail":
            assert (v.check, v.kind, v.known_difference) == ("t1.template_parity", "difference", "KD-0001")
        else:
            assert v.result == "pass", (v.check, v.fixture, v.reason)
    cells = {c.check: c for c in summarize(verdicts)}
    assert cells["t1.template_parity"].result == "difference"
    assert cells["t1.template_parity"].known_differences == ("KD-0001",)
    assert {c.result for check, c in cells.items() if check != "t1.template_parity"} == {"pass"}


def test_every_verdict_records_the_declaration_and_every_plant_its_origin(official):
    for v in _run_all(official):
        assert set(v.declaration["verified"]) == {
            "chat_template",
            "label_convention",
            "truncation_mode",
            "batching",
            "pad_token_id",
            "tool_arguments",
        }, (v.check, v.fixture)
        for p in v.plants:
            assert p["origin"] in ("real-world", "synthetic")
            if p["origin"] == "real-world":
                assert "https://" in p["source"], p


def test_real_world_and_synthetic_plants(qwen3, official):
    """Which planted faults reproduce a bug someone actually hit, and which are synthetic."""
    origins = {}
    for check in TIER1_CHECKS:
        for fid in FIXTURE_IDS:
            if check.not_applicable(load_fixture(fid)) is None:
                for p in check.plants(qwen3, load_fixture(fid), official.declare):
                    origins[p.name] = p.origin
    assert {name for name, origin in origins.items() if origin == "real-world"} == {
        "mask_closing_tokens",
        "pad_is_closing_token_masked",
        "pad_to_ignored",  # P-0005 (2026-11-25)
    }
    assert len(origins) == 22


@pytest.mark.parametrize("check", TIER1_CHECKS, ids=lambda c: c.id)
def test_every_check_documents_its_bug_class_reference_pass_rule_and_plants(check):
    module_doc = sys.modules[type(check).__module__].__doc__
    for heading in ("Bug class", "Reference", "Pass rule", "Planted fault"):
        assert heading in module_doc, (check.id, heading)


def test_one_root_cause_one_finding(qwen3, official):
    """A framework that prepends a stray special token fails template parity; special tokens' fail is blocked."""
    from parity.engine.plants import encoded_plant
    from parity.engine.runner import evaluate_suite

    stray = qwen3.tokenizer.pad_token_id

    def prepend(row):
        return Encoded((stray,) + row.input_ids, (-100,) + row.labels, (1,) + row.attention_mask)

    framework = encoded_plant("stray", "stray", prepend, origin="synthetic").apply(official)
    verdicts = {v.check: v for v in evaluate_suite(TIER1_CHECKS, framework, [load_fixture("single_turn")])}
    parity = verdicts["t1.template_parity"]
    assert (parity.result, parity.kind, parity.flags) == ("fail", "bug", [])
    blocked = verdicts["t1.special_tokens"]
    assert blocked.result == "inconclusive" and blocked.reason.startswith("blocked by t1.template_parity")
    assert blocked.flags == ["blocked_by_template_parity"] and blocked.evidence["blocked_result"] == "fail"
    assert not any(v.result == "fail" for c, v in verdicts.items() if c != "t1.template_parity")


def test_a_known_difference_blocks_nothing(qwen3, trl_qwen3_template):
    from parity.engine.runner import evaluate_suite

    trl = trl_template_stub(qwen3, trl_qwen3_template, name="trl", version="1.18.2")
    verdicts = evaluate_suite(TIER1_CHECKS, trl, [load_fixture("multi_turn")])
    assert not any("blocked_by_template_parity" in v.flags for v in verdicts)
