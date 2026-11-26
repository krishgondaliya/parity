"""Predictions first, per pair (M5): a framework runs on a release only once that pair's predictions are committed.

The guard hook (.claude/hooks/guard.py) checks the framework and version a command names. This module checks the
pair: a `docs/evidence/*/predictions.md` whose first line names the framework, its version and the release's exact
`org/name@<revision>`. Two places use it:
- the M5 runner, against git HEAD: the file must be in HEAD, and the working copy must equal HEAD's (no uncommitted
  edit after the predictions were committed);
- the GPU kit, which runs from a `git archive` of HEAD (no .git there): every file in the archive is committed, so
  the file only has to be in it. `build_kit.py` also checks HEAD before it builds.
The calibration model has its own M3/M4 predictions and is exempt.

It also reads the predicted cells back out of the file, for the predicted-vs-actual comparison. Three layouts are
used: one table per mode under `## Predicted cells: <mode> ...`; one table with a Mode column under
`## Predicted cells`; and the line `All 48 cells: n/a ...` under a mode's heading. A row labelled `five checks`
stands for every check but t1.multiturn_prefix.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from parity.fixtures import FIXTURE_IDS
from parity.serving import SERVING_MODES
from parity.targets import CALIBRATION

CHECKS = (
    "t1.template_parity",
    "t1.assistant_mask",
    "t1.end_of_turn",
    "t1.special_tokens",
    "t1.truncation",
    "t1.multiturn_prefix",
)
FIVE = CHECKS[:5]
RESULTS = {"pass", "fail", "difference", "inconclusive", "not_applicable", "not_configurable", "not_supported"}


class PredictionsMissing(RuntimeError):
    """No committed predictions for this framework, version and release: the pair must not run."""


@dataclass(frozen=True)
class Predictions:
    path: str  # repo-relative, e.g. docs/evidence/2026-11-26-m5-trl-inkling/predictions.md
    title: str
    text: str

    @property
    def folder(self) -> str:
        return self.path.rsplit("/", 1)[0]


def needs_predictions(spec: str) -> bool:
    return spec != CALIBRATION.spec


def title_names(title: str, framework: str, version: str, spec: str) -> bool:
    """Does a predictions title name this framework and version (as words) and this exact release?"""
    named = re.search(rf"\b{re.escape(framework)}\s+{re.escape(version)}(?![\w.])", title, re.IGNORECASE)
    return bool(named) and spec in title


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=60)


def committed_predictions(repo: Path, framework: str, version: str, spec: str) -> Predictions:
    """The predictions.md in git HEAD for this pair, unchanged in the working tree. Raises PredictionsMissing."""
    listing = _git(repo, "ls-tree", "-r", "--name-only", "HEAD", "docs/evidence")
    if listing.returncode != 0:
        raise PredictionsMissing(f"can't list HEAD: {listing.stderr.strip()}")
    found = []
    for path in listing.stdout.splitlines():
        if not path.endswith("/predictions.md"):
            continue
        shown = _git(repo, "show", f"HEAD:{path}")
        if shown.returncode != 0:
            continue
        title = shown.stdout.split("\n", 1)[0]
        if title_names(title, framework, version, spec):
            found.append(Predictions(path, title, shown.stdout))
    if len(found) != 1:
        raise PredictionsMissing(
            f"{framework} {version} on {spec}: {len(found)} committed predictions.md name this pair (need exactly 1): "
            + ", ".join(p.path for p in found)
        )
    working = repo / found[0].path
    if not working.is_file() or working.read_text(encoding="utf-8") != found[0].text:
        raise PredictionsMissing(f"{found[0].path} differs from HEAD: commit the predictions before the run")
    return found[0]


def archived_predictions(root: Path, framework: str, version: str, spec: str) -> Predictions:
    """The predictions.md for this pair inside a git archive of HEAD (the GPU kit's unpacked code)."""
    found = []
    for path in sorted((root / "docs" / "evidence").glob("*/predictions.md")):
        text = path.read_text(encoding="utf-8")
        if title_names(text.split("\n", 1)[0], framework, version, spec):
            found.append(Predictions(path.relative_to(root).as_posix(), text.split("\n", 1)[0], text))
    if len(found) != 1:
        raise PredictionsMissing(
            f"{framework} {version} on {spec}: {len(found)} predictions.md in the archive name this pair (need 1)"
        )
    return found[0]


def _cells_of(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _normalize(value: str) -> str:
    v = value.strip().strip("*").strip().lower()
    v = "not_applicable" if v in ("n/a", "not applicable") else v
    if v not in RESULTS:
        raise ValueError(f"not a predicted result: {value!r}")
    return v


def parse_cells(text: str) -> dict[tuple[str, str, str], str]:
    """(mode, check, fixture) -> predicted result, for every cell of both modes. Raises ValueError if any is missing."""
    cells: dict[tuple[str, str, str], str] = {}
    mode: str | None = None
    in_section = False
    fixtures: list[str] | None = None
    mode_column = False
    for line in text.splitlines():
        heading = re.match(r"## Predicted cells(?::\s*(\w+))?", line)
        if heading:
            in_section, mode, fixtures = True, heading.group(1), None
            continue
        if line.startswith("## "):
            in_section, mode, fixtures = False, None, None
            continue
        if not in_section:
            continue
        if mode and re.match(r"All 48 cells: n/a", line):
            cells.update({(mode, c, f): "not_applicable" for c in CHECKS for f in FIXTURE_IDS})
            continue
        if line.startswith("| Check |") or line.startswith("| Mode | Check |"):
            parts = _cells_of(line)
            mode_column = parts[0] == "Mode"
            fixtures = parts[2:] if mode_column else parts[1:]
            if sorted(fixtures) != sorted(FIXTURE_IDS):
                raise ValueError(f"a predictions table's columns aren't the 8 fixtures: {fixtures}")
            continue
        if fixtures and line.startswith("|") and not line.startswith("| ---"):
            parts = _cells_of(line)
            row_mode, label, values = (parts[0], parts[1], parts[2:]) if mode_column else (mode, parts[0], parts[1:])
            checks = FIVE if label == "five checks" else (label,)
            if row_mode not in SERVING_MODES or any(c not in CHECKS for c in checks) or len(values) != len(fixtures):
                raise ValueError(f"can't read this predictions row: {line!r}")
            for check in checks:
                for fixture, value in zip(fixtures, values):
                    cells[(row_mode, check, fixture)] = _normalize(value)
    expected = {(m, c, f) for m in SERVING_MODES for c in CHECKS for f in FIXTURE_IDS}
    if set(cells) != expected:
        missing = sorted(expected - set(cells))
        raise ValueError(f"predictions cover {len(cells)} of {len(expected)} cells; missing e.g. {missing[:3]}")
    return cells


def actual(verdict) -> str:
    """A verdict's result in the predictions' vocabulary: a fail of kind difference is 'difference'."""
    if verdict.result == "fail":
        return "fail" if verdict.kind == "bug" else "difference"
    return verdict.result
