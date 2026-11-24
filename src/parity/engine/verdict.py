"""The verdict record (`parity.verdict/v0`) and its JSON writer.

One verdict per (check, model, framework, fixture). Rules the record enforces:

- `result` is pass, fail, inconclusive or not_applicable.
- `not_applicable` means the check has nothing to test on this fixture. That
  is decided from the fixture alone, before any framework output exists. It
  is never a pass, and the report counts how many fixtures each check ran on.
- `kind` is set only on a fail: `bug`, or `difference` when the divergence
  matches a known-differences entry exactly (then `known_difference` names it).
- `self_test` is `caught` only if every planted fault that applies to this
  fixture was caught in the same run. Otherwise it is `missed` and the result
  is inconclusive (hard rule 3). It is None only for not_applicable.
- `declaration` records what the adapter declared and, per field, whether the
  framework's behavior confirmed it (`consistent`), exactly matched a
  different value (`contradicted: ...`, an adapter error, so the verdict is
  inconclusive), matched no value (`mismatch: ...`), or couldn't show it
  (`unverified`).
- `template_properties` names model-level template properties (shown once per
  model in the report) that a verdict depends on, e.g. a pass that inherits
  its template's prefix breaks.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SCHEMA = "parity.verdict/v0"
RESULTS = ("pass", "fail", "inconclusive", "not_applicable")
KINDS = ("bug", "difference")
SELF_TESTS = ("caught", "missed")


@dataclass(frozen=True)
class Verdict:
    check: str
    model: str  # "<org>/<name>@<revision>"
    framework: str  # "<name>@<version>"
    fixture: str
    result: str
    kind: str | None = None
    self_test: str | None = None
    known_difference: str | None = None
    reason: str | None = None  # why the result is inconclusive or not_applicable
    evidence: dict[str, Any] = field(default_factory=dict)
    conventions: dict[str, Any] = field(default_factory=dict)  # the framework's recorded choices, e.g. truncation mode
    reference: dict[str, Any] = field(default_factory=dict)  # template source and hashes, library versions
    plants: list[dict[str, Any]] = field(default_factory=list)  # one record per planted fault run
    declaration: dict[str, Any] = field(default_factory=dict)  # what the adapter declared, and what was verified
    template_properties: list[str] = field(default_factory=list)  # model-level template properties this verdict relies on
    env_lock: str | None = None
    repro: str | None = None
    schema: str = SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SCHEMA:
            raise ValueError(f"unknown schema {self.schema!r}")
        if self.result not in RESULTS:
            raise ValueError(f"result must be one of {RESULTS}, got {self.result!r}")
        if self.result == "fail":
            if self.kind not in KINDS:
                raise ValueError(f"a fail needs kind in {KINDS}, got {self.kind!r}")
        elif self.kind is not None:
            raise ValueError(f"kind is only set on a fail, not on {self.result!r}")
        if (self.known_difference is not None) != (self.kind == "difference"):
            raise ValueError("known_difference is set exactly when kind is 'difference'")
        if self.self_test is not None and self.self_test not in SELF_TESTS:
            raise ValueError(f"self_test must be one of {SELF_TESTS} or None, got {self.self_test!r}")
        if self.result in ("inconclusive", "not_applicable") and not self.reason:
            raise ValueError(f"a {self.result} verdict needs a reason")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return {"schema": data.pop("schema"), **data}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n"


def make_verdict(check_id: str, adapter: Any, fixture_id: str, result: str, **fields: Any) -> Verdict:
    """A verdict with the model, framework and lockfile filled in from the adapter."""
    return Verdict(
        check=check_id,
        model=adapter.release.release.spec,
        framework=f"{adapter.name}@{adapter.version}",
        fixture=fixture_id,
        result=result,
        env_lock=adapter.env_lock,
        **fields,
    )


def verdict_filename(verdict: Verdict) -> str:
    return f"{verdict.check}__{verdict.fixture}.json"


def write_verdict(verdict: Verdict, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / verdict_filename(verdict)
    path.write_text(verdict.to_json(), encoding="utf-8")
    return path


def read_verdict(path: Path) -> Verdict:
    return Verdict(**json.loads(path.read_text(encoding="utf-8")))
