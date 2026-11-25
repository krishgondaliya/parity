"""Ingesting the GPU run's results zip: verification passes on the real zip, and refuses anything altered or unsafe."""

from __future__ import annotations

import importlib.util
import shutil
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs" / "evidence" / "2026-11-25-m3-unsloth-gpu"
RAW = next((EVIDENCE / "raw").glob("parity-gpu-results-*.zip"))

spec = importlib.util.spec_from_file_location("gpu_ingest", EVIDENCE / "ingest.py")
ingest = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ingest
spec.loader.exec_module(ingest)


def test_the_committed_zip_verifies_against_its_commit(tmp_path):
    root = ingest.safe_extract(RAW, tmp_path)
    facts = ingest.verify(RAW, root)
    assert facts["commit"].startswith("15ed6df97528") and facts["archive_rebuilt_here"] is True
    assert (facts["files_verified"], facts["verdicts_verified"], facts["unsloth_device_type"]) == (110, 96, "cuda")


def test_a_changed_file_is_refused(tmp_path):
    root = ingest.safe_extract(RAW, tmp_path)
    verdict = next((root / "results").rglob("t1.end_of_turn__single_turn.json"))
    verdict.write_text(verdict.read_text().replace('"pass"', '"fail"', 1))
    with pytest.raises(ingest.Refused, match="sha256 mismatch"):
        ingest.verify(RAW, root)


def test_an_extra_file_is_refused(tmp_path):
    root = ingest.safe_extract(RAW, tmp_path)
    (root / "results" / "planted.json").write_text("{}")
    with pytest.raises(ingest.Refused, match="files and manifests differ"):
        ingest.verify(RAW, root)


@pytest.mark.parametrize("name", ["../escape.txt", "/etc/escape.txt", "top/../../escape.txt"])
def test_entry_names_that_leave_the_folder_are_refused(tmp_path, name):
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("top/ok.txt", "ok")
        zf.writestr(name, "no")
    with pytest.raises(ingest.Refused):
        ingest.safe_extract(bad, tmp_path / "out")
    assert not (tmp_path / "escape.txt").exists()


def test_a_zip_from_an_unknown_commit_is_refused(tmp_path):
    root = ingest.safe_extract(RAW, tmp_path)
    copy = tmp_path / "copy.zip"
    shutil.copyfile(RAW, copy)
    manifest = root / "MANIFEST.json"
    manifest.write_text(manifest.read_text().replace("15ed6df97528", "0000000000ab", 1))
    with pytest.raises(ingest.Refused):
        ingest.verify(copy, root)


def test_both_prediction_tables_are_read():
    predicted = ingest.predictions()
    assert len(predicted) == 96
    assert predicted[("thinking", "t1.special_tokens", "single_turn")] == "pass"
    assert predicted[("non_thinking", "t1.assistant_mask", "single_turn")] == "fail"
    assert predicted[("non_thinking", "t1.truncation", "reasoning")] == "inconclusive"
    assert predicted[("thinking", "t1.multiturn_prefix", "long_answer")] == "not_applicable"


# The P-0005-only trip (docs/evidence/2026-11-25-p0005-gpu/), through the same parity.gpu_results checks

P0005_EVIDENCE = ROOT / "docs" / "evidence" / "2026-11-25-p0005-gpu"
P0005_RAW = next((P0005_EVIDENCE / "raw").glob("parity-gpu-results-*.zip"))


def test_the_p0005_zip_verifies_as_a_p0005_only_run(tmp_path):
    from parity.gpu_results import safe_extract, verify_zip

    root = safe_extract(P0005_RAW, tmp_path)
    facts = verify_zip(P0005_RAW, root, ROOT, mode="p0005-only")
    assert facts["commit"].startswith("25c3dc83997d") and facts["mode"] == "p0005-only"
    assert (facts["files_verified"], facts["unsloth_device_type"]) == (12, "cuda")


def test_a_zip_from_the_other_kind_of_run_is_refused(tmp_path):
    from parity.gpu_results import Refused, safe_extract, verify_zip

    with pytest.raises(Refused, match="expected a 'full' run"):
        verify_zip(P0005_RAW, safe_extract(P0005_RAW, tmp_path / "a"), ROOT, mode="full")
    with pytest.raises(Refused, match="expected a 'p0005-only' run"):
        verify_zip(RAW, safe_extract(RAW, tmp_path / "b"), ROOT, mode="p0005-only")


def test_the_p0005_ingest_refuses_a_missing_run(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("p0005_ingest", P0005_EVIDENCE / "ingest.py")
    p0005 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(p0005)
    monkeypatch.setattr(p0005, "OUT", tmp_path)  # never touch the committed evidence
    monkeypatch.setattr(p0005, "EXPECTED_RUNS", (*p0005.EXPECTED_RUNS, "impact"))
    with pytest.raises(p0005.Refused, match="expected the runs"):
        p0005.ingest(P0005_RAW)
    assert not any(tmp_path.iterdir())
