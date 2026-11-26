"""Predictions first, per pair (parity.engine.predictions): the guard, and reading the predicted cells back."""

import subprocess
from collections import Counter
from pathlib import Path

import pytest

from parity.engine.predictions import (
    PredictionsMissing,
    archived_predictions,
    committed_predictions,
    needs_predictions,
    parse_cells,
    title_names,
)
from parity.targets import CALIBRATION, PHASE0_TARGETS

REPO = Path(__file__).resolve().parents[1]
DEEPSEEK = next(t for t in PHASE0_TARGETS if t.repo_id.startswith("deepseek-ai/"))
INKLING = next(t for t in PHASE0_TARGETS if t.repo_id.startswith("thinkingmachines/"))

# The counts each M5 predictions file states, mode by mode (results in the parser's vocabulary).
STATED = {
    ("TRL", "1.18.2", INKLING.spec): (
        {"pass": 38, "fail": 1, "inconclusive": 5, "not_applicable": 4},
        {"not_applicable": 48},
    ),
    ("TRL", "1.18.2", DEEPSEEK.spec): (
        {"not_supported": 44, "not_applicable": 4},
        {"not_supported": 38, "not_applicable": 10},
    ),
    ("Axolotl", "0.21.1", DEEPSEEK.spec): (
        {"pass": 21, "fail": 11, "inconclusive": 12, "not_applicable": 4},
        {"pass": 26, "inconclusive": 12, "not_applicable": 10},
    ),
    ("Unsloth", "2026.11.10", DEEPSEEK.spec): (
        {"pass": 14, "fail": 12, "inconclusive": 18, "not_applicable": 4},
        {"not_configurable": 38, "not_applicable": 10},
    ),
}


@pytest.mark.parametrize("pair", STATED, ids=lambda p: f"{p[0]}-{p[2].split('/')[1][:12]}")
def test_each_m5_predictions_file_is_found_and_reads_as_it_states(pair):
    predictions = committed_predictions(REPO, *pair)
    cells = parse_cells(predictions.text)
    thinking, non_thinking = STATED[pair]
    assert Counter(v for (m, _, _), v in cells.items() if m == "thinking") == thinking
    assert Counter(v for (m, _, _), v in cells.items() if m == "non_thinking") == non_thinking


def test_titles_name_the_framework_version_and_exact_release():
    title = f"# Predictions: TRL 1.18.2 (vlm variant) on {INKLING.spec} (both serving modes)"
    assert title_names(title, "TRL", "1.18.2", INKLING.spec)
    assert not title_names(title, "TRL", "1.18.20", INKLING.spec)
    assert not title_names(title.replace("1.18.2", "1.18.20"), "TRL", "1.18.2", INKLING.spec)
    assert not title_names(title, "TRL", "1.18.2", DEEPSEEK.spec)
    assert not needs_predictions(CALIBRATION.spec) and needs_predictions(DEEPSEEK.spec)


def _repo(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    for key, value in (("user.email", "t@example.invalid"), ("user.name", "t"), ("commit.gpgsign", "false")):
        subprocess.run(["git", "-C", str(tmp_path), "config", key, value], check=True)
    return tmp_path


def _commit(repo: Path, rel: str, text: str) -> None:
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", rel], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "x"], check=True)


def test_the_guard_needs_the_pairs_predictions_committed_and_unchanged(tmp_path):
    repo = _repo(tmp_path)
    rel = "docs/evidence/x-m5-axolotl-deepseek/predictions.md"
    text = f"# Predictions: Axolotl 0.21.1 on {DEEPSEEK.spec}\n\nbody\n"
    with pytest.raises(PredictionsMissing):
        committed_predictions(repo, "Axolotl", "0.21.1", DEEPSEEK.spec)  # nothing committed yet
    _commit(repo, rel, text)
    assert committed_predictions(repo, "Axolotl", "0.21.1", DEEPSEEK.spec).folder == "docs/evidence/x-m5-axolotl-deepseek"
    with pytest.raises(PredictionsMissing):
        committed_predictions(repo, "Axolotl", "0.21.2", DEEPSEEK.spec)  # another version
    with pytest.raises(PredictionsMissing):
        committed_predictions(repo, "TRL", "1.18.2", DEEPSEEK.spec)  # another framework
    (repo / rel).write_text(text + "edited after the commit\n", encoding="utf-8")
    with pytest.raises(PredictionsMissing, match="differs from HEAD"):
        committed_predictions(repo, "Axolotl", "0.21.1", DEEPSEEK.spec)


def test_two_files_naming_the_same_pair_are_refused(tmp_path):
    repo = _repo(tmp_path)
    text = f"# Predictions: Axolotl 0.21.1 on {DEEPSEEK.spec}\n"
    _commit(repo, "docs/evidence/a/predictions.md", text)
    _commit(repo, "docs/evidence/b/predictions.md", text)
    with pytest.raises(PredictionsMissing, match="2 committed"):
        committed_predictions(repo, "Axolotl", "0.21.1", DEEPSEEK.spec)


def test_the_kit_finds_predictions_in_its_archive_tree(tmp_path):
    folder = tmp_path / "docs" / "evidence" / "x-m5-unsloth-deepseek"
    folder.mkdir(parents=True)
    (folder / "predictions.md").write_text(f"# Predictions: Unsloth 2026.11.10 on {DEEPSEEK.spec}, GPU\n", "utf-8")
    assert archived_predictions(tmp_path, "Unsloth", "2026.11.10", DEEPSEEK.spec).folder.endswith("x-m5-unsloth-deepseek")
    with pytest.raises(PredictionsMissing):
        archived_predictions(tmp_path, "Unsloth", "2026.11.10", INKLING.spec)


def test_a_predictions_file_missing_cells_is_refused():
    text = "## Predicted cells: thinking (primary)\n\n| Check | single_turn |\n| --- | --- |\n| t1.template_parity | pass |\n"
    with pytest.raises(ValueError):
        parse_cells(text)
