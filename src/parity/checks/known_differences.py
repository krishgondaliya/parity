"""The known-differences registry: documented, deliberate framework behaviors that are not bugs.

A divergence counts as a `difference` only if one entry matches it exactly
(docs/DECISIONS.md, "Known-differences registry"):

- same check, same framework, and a framework version listed in the entry;
- the model's official template and the framework's declared template both
  have exactly the sha256 the entry names;
- the framework's tokens equal, token for token, what the entry's signature
  function computes from the official render.

There are no wildcards. Anything else stays a `fail` of kind `bug`.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cache
from importlib.resources import files

from parity.fixtures import Conversation
from parity.references.template import TemplateReferenceError, render_text, template_sha256
from parity.release import LoadedRelease

REGISTRY_SCHEMA = "parity.known_differences/v0"
_SHA256_HEX = 64


@dataclass(frozen=True)
class KnownDifference:
    id: str
    framework: str
    versions: tuple[str, ...]
    check: str
    model_family: str
    model_template_sha256: str
    framework_template_sha256: str
    signature: str
    summary: str
    docs: tuple[str, ...]
    decision: str


# Signatures: each computes the exact token ids a framework produces under the documented behavior.


def insert_think_block_where_official_has_none(
    release: LoadedRelease, conv: Conversation, official_template: str
) -> list[int]:
    """KD-0001: TRL's Qwen3 training template always emits a think block.

    Start from the official render. For every assistant turn the official
    template renders without a think block, insert
    "<think>\\n" + reasoning_content.strip("\\n") + "\\n</think>\\n\\n"
    right after that turn's generation prompt ("<|im_start|>assistant\\n").
    """
    tokenizer = release.tokenizer
    text = render_text(tokenizer, conv.messages, conv.tools, official_template)
    inserts = []
    for k in conv.assistant_indices:
        prompt = render_text(tokenizer, conv.messages[:k], conv.tools, official_template, add_generation_prompt=True)
        if not text.startswith(prompt):
            raise TemplateReferenceError(f"KD signature: message {k}'s generation prompt is not a prefix of the render")
        position = len(prompt)
        if text.startswith("<think>", position):
            continue
        reasoning = conv.messages[k].get("reasoning_content")
        reasoning = reasoning if isinstance(reasoning, str) else ""
        inserts.append((position, "<think>\n" + reasoning.strip("\n") + "\n</think>\n\n"))
    for position, block in reversed(inserts):
        text = text[:position] + block + text[position:]
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


SIGNATURES: dict[str, Callable[[LoadedRelease, Conversation, str], list[int]]] = {
    "insert_think_block_where_official_has_none": insert_think_block_where_official_has_none,
}


@cache
def load_registry() -> tuple[KnownDifference, ...]:
    raw = json.loads((files("parity.checks") / "known_differences.json").read_bytes().decode("utf-8"))
    if raw.get("schema") != REGISTRY_SCHEMA:
        raise ValueError(f"known_differences.json: expected schema {REGISTRY_SCHEMA!r}")
    entries = []
    for e in raw["entries"]:
        entry = KnownDifference(
            id=e["id"],
            framework=e["framework"],
            versions=tuple(e["versions"]),
            check=e["check"],
            model_family=e["model_family"],
            model_template_sha256=e["model_template_sha256"],
            framework_template_sha256=e["framework_template_sha256"],
            signature=e["signature"],
            summary=e["summary"],
            docs=tuple(e["docs"]),
            decision=e["decision"],
        )
        _validate(entry)
        entries.append(entry)
    ids = [e.id for e in entries]
    if len(ids) != len(set(ids)):
        raise ValueError(f"known_differences.json: duplicate ids in {ids}")
    return tuple(entries)


def _validate(e: KnownDifference) -> None:
    where = f"known_differences.json {e.id}"
    if not e.versions or any(not v or any(c in v for c in "*?<>=~^, ") for v in e.versions):
        raise ValueError(f"{where}: versions must be exact version strings, got {e.versions}")
    for name in ("model_template_sha256", "framework_template_sha256"):
        value = getattr(e, name)
        if len(value) != _SHA256_HEX or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"{where}: {name} must be a full sha256 hex digest")
    if e.signature not in SIGNATURES:
        raise ValueError(f"{where}: unknown signature {e.signature!r}")
    if not e.docs:
        raise ValueError(f"{where}: needs at least one doc link")


def match_known_difference(
    *,
    check_id: str,
    framework: str,
    version: str,
    release: LoadedRelease,
    conv: Conversation,
    official_template: str,
    declared_template: str,
    got_ids: Sequence[int],
) -> KnownDifference | None:
    """The one entry that explains these tokens exactly, or None."""
    for entry in load_registry():
        if entry.check != check_id or entry.framework != framework or version not in entry.versions:
            continue
        if template_sha256(official_template) != entry.model_template_sha256:
            continue
        if template_sha256(declared_template) != entry.framework_template_sha256:
            continue
        if list(got_ids) == SIGNATURES[entry.signature](release, conv, official_template):
            return entry
    return None
