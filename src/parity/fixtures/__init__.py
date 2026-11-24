"""The 8 test conversations every Tier 1 check runs on.

Each fixture is a JSON file in `data/`, named by its id. The id is what a
verdict records in its "fixture" field. Messages use the OpenAI-style chat
format that Hugging Face chat templates expect: `role`, `content`, and
optionally `reasoning_content`, `tool_calls`, `tool_call_id` and `name`.

`load_fixture` parses the file fresh every time, so a check or planted fault
that mutates a fixture can never leak that change into another run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files
from typing import Any

FIXTURE_IDS: tuple[str, ...] = (
    "single_turn",
    "multi_turn",
    "system_prompt",
    "tool_call",
    "reasoning",
    "empty_assistant",
    "long_answer",
    "non_ascii",
)


@dataclass(frozen=True)
class Fixture:
    id: str
    title: str
    notes: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None
    max_len: int | None  # set only on fixtures meant to hit a length limit

    @property
    def assistant_indices(self) -> list[int]:
        return [i for i, m in enumerate(self.messages) if m["role"] == "assistant"]


def fixture_path(fixture_id: str):
    if fixture_id not in FIXTURE_IDS:
        raise KeyError(f"unknown fixture {fixture_id!r}; expected one of {FIXTURE_IDS}")
    return files("parity.fixtures") / "data" / f"{fixture_id}.json"


def load_fixture(fixture_id: str) -> Fixture:
    raw = json.loads(fixture_path(fixture_id).read_bytes().decode("utf-8"))
    if raw["id"] != fixture_id:
        raise ValueError(f"fixture file {fixture_id}.json declares id {raw['id']!r}")
    return Fixture(
        id=raw["id"],
        title=raw["title"],
        notes=raw["notes"],
        messages=raw["messages"],
        tools=raw["tools"],
        max_len=raw["max_len"],
    )


def load_all() -> list[Fixture]:
    return [load_fixture(fid) for fid in FIXTURE_IDS]
