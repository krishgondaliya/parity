"""Root causes: every fail links to the finding that explains it, and fails are counted by root cause.

One root cause often fails several cells: P-0002 (Axolotl's empty turn) fails t1.end_of_turn, t1.special_tokens,
t1.assistant_mask and t1.truncation on the same fixture, in both serving modes. The report card and Gate A count
root causes, not cells; every cell keeps its own verdict.

How a verdict gets its root cause (`Verdict.root_cause`):
- A fail of kind `difference` links to the finding its known-differences entry came from (`KnownDifference.finding`),
  or to the entry itself when it came from no finding (KD-0001).
- A fail of kind `bug` links to a triaged finding only by an exact pin in `findings/root_causes.json`: the same
  framework version, model, check, fixture and serving mode, and the same evidence (sha256 of its canonical JSON).
  A pin is written after /triage and the skeptic, from the verdict itself (`pin()`).
- No wildcards. If the failure changes in any way, its evidence hash changes, the link breaks, and the cell is
  untriaged again. Untriaged fails are counted one per cell, so nothing is hidden by grouping.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from parity.checks.known_differences import load_registry
from parity.engine.verdict import Verdict

SCHEMA = "parity.root_causes/v0"
DEFAULT_PATH = Path(__file__).resolve().parents[3] / "findings" / "root_causes.json"


@dataclass(frozen=True)
class Pin:
    check: str
    fixture: str
    serving_mode: str
    evidence_sha256: str


@dataclass(frozen=True)
class RootCause:
    id: str  # the finding id, e.g. "P-0002"
    record: str  # findings/<id>/record.md
    status: str  # the finding's status: candidate, confirmed, fixed, difference, ...
    framework: str  # "<name>@<version>", exactly as verdicts record it
    model: str  # "<org>/<name>@<revision>"
    summary: str
    pins: tuple[Pin, ...]


def evidence_sha256(v: Verdict) -> str:
    canonical = json.dumps(v.evidence, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def pin(v: Verdict) -> dict[str, str]:
    """The pin to add to a root cause for this exact verdict (after /triage and the skeptic)."""
    if v.result != "fail" or v.kind != "bug":
        raise ValueError(f"only a fail of kind bug is pinned, not {v.result}/{v.kind}")
    return {"check": v.check, "fixture": v.fixture, "serving_mode": v.serving_mode or "", "evidence_sha256": evidence_sha256(v)}


def load_root_causes(path: Path = DEFAULT_PATH) -> tuple[RootCause, ...]:
    if not path.exists():
        return ()
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema") != SCHEMA:
        raise ValueError(f"{path}: expected schema {SCHEMA!r}")
    out = []
    for e in raw["root_causes"]:
        pins = tuple(Pin(**p) for p in e["pins"])
        if not pins or any(len(p.evidence_sha256) != 64 for p in pins):
            raise ValueError(f"{path} {e['id']}: every pin needs a full evidence sha256")
        out.append(
            RootCause(
                id=e["id"],
                record=e["record"],
                status=e["status"],
                framework=e["framework"],
                model=e["model"],
                summary=e["summary"],
                pins=pins,
            )
        )
    ids = [r.id for r in out]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate root cause ids {ids}")
    return tuple(out)


def link_root_causes(verdicts: Iterable[Verdict], root_causes: Iterable[RootCause] | None = None) -> list[Verdict]:
    """Set `root_cause` on every fail that a known difference or an exact pin explains."""
    causes = tuple(root_causes) if root_causes is not None else load_root_causes()
    kd_finding = {e.id: e.finding or e.id for e in load_registry()}
    out = []
    for v in verdicts:
        cause = None
        if v.result == "fail" and v.kind == "difference":
            cause = kd_finding.get(v.known_difference or "")
        elif v.result == "fail" and v.kind == "bug":
            key = Pin(v.check, v.fixture, v.serving_mode or "", evidence_sha256(v))
            matches = [r.id for r in causes if r.framework == v.framework and r.model == v.model and key in r.pins]
            if len(matches) > 1:
                raise ValueError(f"{v.check}/{v.fixture}: pinned to more than one root cause {matches}")
            cause = matches[0] if matches else None
        out.append(replace(v, root_cause=cause))
    return out


@dataclass(frozen=True)
class FailCount:
    root_causes: dict[str, int]  # root cause id -> how many fail verdicts it explains
    untriaged: tuple[str, ...]  # "<framework> <check>/<fixture> (<mode>)" for each fail with no root cause
    cells: int  # fail verdicts in all

    @property
    def total(self) -> int:
        """Root causes, with every untriaged fail counted as its own."""
        return len(self.root_causes) + len(self.untriaged)


def count_fails(verdicts: Iterable[Verdict]) -> FailCount:
    causes: dict[str, int] = {}
    untriaged = []
    cells = 0
    for v in verdicts:
        if v.result != "fail":
            continue
        cells += 1
        if v.root_cause:
            causes[v.root_cause] = causes.get(v.root_cause, 0) + 1
        else:
            untriaged.append(f"{v.framework} {v.check}/{v.fixture} ({v.serving_mode or 'unrecorded'})")
    return FailCount(dict(sorted(causes.items())), tuple(sorted(untriaged)), cells)


def describe(causes: Iterable[RootCause]) -> dict[str, Any]:
    return {r.id: {"status": r.status, "record": r.record, "summary": r.summary} for r in causes}
