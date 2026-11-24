"""The engine: verdict rules, evidence, the runner's self-test logic, and the summary."""

from __future__ import annotations

import pytest

from parity.adapters.base import Declaration, Encoded, TrainCfg
from parity.checks.common import Tier1Check
from parity.engine.evidence import first_divergence, token_evidence
from parity.engine.plants import Plant, PlantedAdapter, supervised_runs
from parity.engine.runner import evaluate
from parity.engine.summary import summarize
from parity.engine.verdict import Verdict, read_verdict, write_verdict
from parity.fixtures import load_fixture
from parity.references.template import MarkerDisagreementError

# Verdict record


def _v(**kw) -> Verdict:
    base = dict(check="t1.x", model="org/m@" + "0" * 40, framework="f@1", fixture="single_turn", result="pass")
    return Verdict(**{**base, **kw})


def test_a_fail_needs_a_kind():
    with pytest.raises(ValueError, match="kind"):
        _v(result="fail")


def test_kind_only_on_a_fail():
    with pytest.raises(ValueError, match="only set on a fail"):
        _v(result="pass", kind="bug")


def test_a_difference_names_its_known_difference_entry():
    with pytest.raises(ValueError, match="known_difference"):
        _v(result="fail", kind="difference")
    with pytest.raises(ValueError, match="known_difference"):
        _v(result="fail", kind="bug", known_difference="KD-0001")
    assert _v(result="fail", kind="difference", known_difference="KD-0001").known_difference == "KD-0001"


@pytest.mark.parametrize("result", ["inconclusive", "not_applicable"])
def test_inconclusive_and_not_applicable_need_a_reason(result):
    with pytest.raises(ValueError, match="reason"):
        _v(result=result)


def test_unknown_result_is_refused():
    with pytest.raises(ValueError, match="result"):
        _v(result="skipped")


def test_verdict_json_round_trip(tmp_path):
    v = _v(result="fail", kind="bug", evidence={"first_divergence": 3}, plants=[{"plant": "p", "caught": True}])
    path = write_verdict(v, tmp_path)
    assert path.name == "t1.x__single_turn.json"
    assert read_verdict(path) == v
    assert path.read_text(encoding="utf-8").startswith('{\n  "schema": "parity.verdict/v0"')


# Evidence


@pytest.mark.parametrize(
    "a, b, expected",
    [([1, 2, 3], [1, 2, 3], None), ([1, 2, 3], [1, 9, 3], 1), ([1, 2], [1, 2, 3], 2), ([1, 2, 3], [1, 2], 2), ([], [5], 0)],
)
def test_first_divergence(a, b, expected):
    assert first_divergence(a, b) == expected


def test_token_evidence_shows_eight_decoded_tokens_each_side(qwen3):
    ids = list(range(1000, 1030))
    got = ids[:12] + [2000] + ids[13:]
    ev = token_evidence(qwen3.tokenizer, ids, got)
    assert ev["first_divergence"] == 12
    assert len(ev["expected"]) == 8 and len(ev["got"]) == 8 and len(ev["context_before"]) == 8
    assert ev["expected"][0] != ev["got"][0] and ev["expected"][1:] == ev["got"][1:]


def test_supervised_runs():
    row = Encoded((1, 2, 3, 4, 5, 6), (-100, 2, 3, -100, 5, 6), (1,) * 6)
    assert supervised_runs(row) == [(1, 3), (4, 6)]


# Runner: the self-test rules (hard rule 3)


class FakeCheck(Tier1Check):
    """Passes or fails on demand; its planted runs fail only if `catches` says so."""

    id = "t1.fake"

    def __init__(self, clean="pass", catches=(True,), na=None, raises=None):
        self.clean, self.catches, self.na, self.raises = clean, catches, na, raises

    def not_applicable(self, fixture):
        return self.na

    def run(self, adapter, release, fixture):
        plant = getattr(adapter, "plant_name", None)
        if plant is None:
            if self.raises:
                raise self.raises
            kind = "bug" if self.clean == "fail" else None
            reason = "x" if self.clean == "inconclusive" else None
            return self.verdict(adapter, fixture, self.clean, kind=kind, reason=reason)
        caught = self.catches[int(plant.removeprefix("p"))]
        return self.verdict(adapter, fixture, "fail" if caught else "pass", kind="bug" if caught else None)

    def plants(self, release, fixture, declaration):
        return [
            Plant(f"p{i}", "fake", lambda a, i=i: PlantedAdapter(a, f"p{i}")) for i in range(len(self.catches))
        ]


@pytest.fixture
def single_turn():
    return load_fixture("single_turn")


def test_clean_pass_with_every_plant_caught_is_a_pass(official, single_turn):
    v = evaluate(FakeCheck(catches=(True, True)), official, single_turn)
    assert (v.result, v.self_test) == ("pass", "caught")
    assert [p["caught"] for p in v.plants] == [True, True]


def test_one_missed_plant_makes_a_pass_inconclusive(official, single_turn):
    v = evaluate(FakeCheck(catches=(True, False)), official, single_turn)
    assert (v.result, v.self_test, v.kind) == ("inconclusive", "missed", None)
    assert "p1" in v.reason and v.evidence["clean_result"] == "pass"


def test_a_missed_plant_makes_even_a_fail_inconclusive(official, single_turn):
    v = evaluate(FakeCheck(clean="fail", catches=(False,)), official, single_turn)
    assert (v.result, v.self_test) == ("inconclusive", "missed")
    assert (v.evidence["clean_result"], v.evidence["clean_kind"]) == ("fail", "bug")


def test_no_applicable_plant_is_inconclusive(official, single_turn):
    v = evaluate(FakeCheck(catches=()), official, single_turn)
    assert v.result == "inconclusive" and "no planted fault applies" in v.reason


def test_a_crash_is_inconclusive_never_pass(official, single_turn):
    v = evaluate(FakeCheck(raises=KeyError("boom")), official, single_turn)
    assert v.result == "inconclusive" and "KeyError" in v.reason


def test_marker_disagreement_is_inconclusive_and_flagged_for_triage(official, single_turn):
    err = MarkerDisagreementError("markers disagree", [{"index": 3, "zone": "must_not_train", "marked": True}])
    v = evaluate(FakeCheck(raises=err), official, single_turn)
    assert v.result == "inconclusive"
    assert v.evidence["triage"] == "possible model-template finding"
    assert v.evidence["marker_disagreement"][0]["index"] == 3


def test_not_applicable_runs_nothing(official, single_turn):
    v = evaluate(FakeCheck(na="nothing to test", raises=AssertionError("must not run")), official, single_turn)
    assert (v.result, v.self_test, v.plants, v.reason) == ("not_applicable", None, [], "nothing to test")


def test_planted_adapter_passes_through_by_default(official, single_turn):
    planted = PlantedAdapter(official, "nothing")
    cfg = TrainCfg()
    assert planted.encode(single_turn, cfg) == official.encode(single_turn, cfg)
    assert planted.declare(cfg) == official.declare(cfg)
    assert isinstance(planted.declare(cfg), Declaration)
    assert (planted.name, planted.version, planted.release) == (official.name, official.version, official.release)


# Summary


def _cell(results):
    vs = []
    for i, r in enumerate(results):
        kw = {"result": r, "fixture": f"f{i}"}
        if r == "fail":
            kw["kind"] = "bug"
        elif r == "difference":
            kw.update(result="fail", kind="difference", known_difference="KD-0001")
        elif r in ("inconclusive", "not_applicable"):
            kw["reason"] = "x"
        vs.append(_v(**kw))
    [cell] = summarize(vs)
    return cell


def test_summary_counts_how_many_fixtures_the_check_ran_on():
    cell = _cell(["pass"] * 4 + ["not_applicable"] * 4)
    assert (cell.fixtures, cell.ran_on, cell.not_applicable, cell.result) == (8, 4, 4, "pass")


def test_a_check_that_ran_on_zero_fixtures_is_inconclusive():
    assert _cell(["not_applicable"] * 8).result == "inconclusive"


@pytest.mark.parametrize(
    "results, expected",
    [
        (["pass", "fail", "inconclusive", "difference"], "fail"),
        (["pass", "inconclusive", "difference"], "inconclusive"),
        (["pass", "difference", "not_applicable"], "difference"),
        (["pass", "pass"], "pass"),
    ],
)
def test_summary_cell_result(results, expected):
    cell = _cell(results)
    assert cell.result == expected
    if "difference" in results:
        assert cell.known_differences == ("KD-0001",)
