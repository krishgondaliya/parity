"""The 8 test conversations every Tier 1 check runs on.

Each fixture is a JSON file in `data/`, named by its id. The id is what a
verdict records in its "fixture" field. Messages use the OpenAI-style chat
format that Hugging Face chat templates expect: `role`, `content`, and
optionally `reasoning_content`, `tool_calls`, `tool_call_id` and `name`.

`load_fixture` parses the file fresh every time, so a check or planted fault
that mutates a fixture can never leak that change into another run.

A `Fixture` is a `Conversation` with a title, notes and an optional length
limit. Checks also build shorter conversations from a fixture (for example
`messages[:j]` for the multi-turn prefix check), and adapters and the
reference accept any `Conversation`.
"""

from __future__ import annotations

import copy
import hashlib
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
class Conversation:
    """Messages plus optional tool definitions: what an adapter is asked to encode."""

    id: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None

    @property
    def assistant_indices(self) -> list[int]:
        return [i for i, m in enumerate(self.messages) if m["role"] == "assistant"]

    def prefix(self, j: int) -> Conversation:
        """A fresh copy of the first j messages, with the same tools."""
        if not 1 <= j <= len(self.messages):
            raise ValueError(f"prefix length {j} out of range for {len(self.messages)} messages")
        return Conversation(
            id=f"{self.id}[:{j}]",
            messages=copy.deepcopy(self.messages[:j]),
            tools=copy.deepcopy(self.tools),
        )

    def with_messages(self, messages: list[dict[str, Any]]) -> Conversation:
        """The same conversation id and tools with different messages (used by planted faults)."""
        return Conversation(id=self.id, messages=messages, tools=copy.deepcopy(self.tools))

    @property
    def content_key(self) -> str:
        """A hash of the messages and tools, for caching.

        Key order is kept, not sorted: templates render dicts in their own key
        order (transformers' `tojson` doesn't sort), so two conversations that
        differ only in key order render differently and must not share a key.
        """
        blob = json.dumps({"messages": self.messages, "tools": self.tools}, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Fixture(Conversation):
    title: str
    notes: str
    max_len: int | None  # set only on fixtures meant to hit a length limit


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
