"""The known-differences registry: documented, deliberate framework behaviors that are not bugs.

A divergence counts as a `difference` only if one entry matches it exactly
(docs/DECISIONS.md, "Known-differences registry"):

- same check, same framework, and a framework version listed in the entry;
- the model's official template and the framework's declared template both
  have exactly the sha256 the entry names;
- the verdict's serving mode is one the entry lists;
- the signature explains the whole divergence:
  - token signatures (t1.template_parity): the framework's tokens equal, token
    for token, what the signature computes from the official render;
  - zone signatures (t1.assistant_mask, t1.truncation): the signature moves
    named tokens of the reference to the zone the documented behavior gives
    them, and the framework's labels pass the check's own rule against those
    zones, with no violation left over.

There are no wildcards. Anything else stays a `fail` of kind `bug`; when an
entry is in scope but leaves violations unexplained, the verdict's evidence
says which, so triage can link them to their own root cause.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from functools import cache
from importlib.resources import files

from parity.fixtures import Conversation
from parity.references.template import TemplateReference, TemplateReferenceError, Zone, render_text, template_sha256
from parity.release import LoadedRelease
from parity.serving import SERVING_MODES

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
    serving_modes: tuple[str, ...]
    signature: str
    summary: str
    docs: tuple[str, ...]
    decision: str
    docs_note: str = ""  # what the framework's user docs say (or don't) about this behavior


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


QWEN3_THINK_OPENERS = ("<think>\n\n</think>\n\n", "<think>\n")  # the empty block first, so it wins over its prefix


def think_opener_untrained(release: LoadedRelease, conv: Conversation, ref: TemplateReference) -> tuple[Zone, ...]:
    """KD-0002 / KD-0003: Axolotl's default chat_template labeling, with Qwen3's own template, in thinking mode.

    Axolotl labels each turn from its first field (reasoning_content, else content) to its end token; template
    structure before the first field gets no loss. With Qwen3's own template that structure is the think block's
    opener at the start of a turn: the whole empty block "<think>\n\n</think>\n\n" when the turn has no reasoning,
    "<think>\n" when it has. Exactly those tokens move from must-train to must-not-train; every other zone, and
    every other turn, is unchanged.
    """
    tokenizer = release.tokenizer
    zones = list(ref.zones)
    for span in ref.spans:
        for opener in QWEN3_THINK_OPENERS:
            opener_ids = list(tokenizer(opener, add_special_tokens=False)["input_ids"])
            end = span.token_start + len(opener_ids)
            if list(ref.input_ids[span.token_start : end]) != opener_ids:
                continue
            if any(z is not Zone.MUST_TRAIN for z in zones[span.token_start : end]):
                raise TemplateReferenceError(f"KD signature: message {span.message_index}'s think opener isn't must-train")
            zones[span.token_start : end] = [Zone.MUST_NOT_TRAIN] * len(opener_ids)
            break
    return tuple(zones)


SIGNATURES: dict[str, Callable[[LoadedRelease, Conversation, str], list[int]]] = {
    "insert_think_block_where_official_has_none": insert_think_block_where_official_has_none,
}
ZONE_SIGNATURES: dict[str, Callable[[LoadedRelease, Conversation, TemplateReference], tuple[Zone, ...]]] = {
    "think_opener_untrained": think_opener_untrained,
}
TOKEN_CHECKS = frozenset({"t1.template_parity"})
ZONE_CHECKS = frozenset({"t1.assistant_mask", "t1.truncation"})


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
            serving_modes=tuple(e["serving_modes"]),
            signature=e["signature"],
            summary=e["summary"],
            docs=tuple(e["docs"]),
            decision=e["decision"],
            docs_note=e.get("docs_note", ""),
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
    if not e.serving_modes or any(m not in SERVING_MODES for m in e.serving_modes):
        raise ValueError(f"{where}: serving_modes must list modes from {SERVING_MODES}, got {e.serving_modes}")
    if e.check in TOKEN_CHECKS:
        if e.signature not in SIGNATURES:
            raise ValueError(f"{where}: {e.check} needs a token signature, got {e.signature!r}")
    elif e.check in ZONE_CHECKS:
        if e.signature not in ZONE_SIGNATURES:
            raise ValueError(f"{where}: {e.check} needs a zone signature, got {e.signature!r}")
    else:
        raise ValueError(f"{where}: no known differences are defined for check {e.check!r}")
    if not e.docs:
        raise ValueError(f"{where}: needs at least one doc link")


def entries_in_scope(
    *, check_id: str, framework: str, version: str, official_template: str, declared_template: str, mode: str
) -> list[KnownDifference]:
    """Entries for exactly this check, framework version, template pair and serving mode."""
    official, declared = template_sha256(official_template), template_sha256(declared_template)
    return [
        e
        for e in load_registry()
        if e.check == check_id
        and e.framework == framework
        and version in e.versions
        and e.model_template_sha256 == official
        and e.framework_template_sha256 == declared
        and mode in e.serving_modes
    ]


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
    mode: str,
) -> KnownDifference | None:
    """The one token-signature entry that explains these tokens exactly, or None."""
    for entry in entries_in_scope(
        check_id=check_id,
        framework=framework,
        version=version,
        official_template=official_template,
        declared_template=declared_template,
        mode=mode,
    ):
        if list(got_ids) == SIGNATURES[entry.signature](release, conv, official_template):
            return entry
    return None


def known_zones(entry: KnownDifference, release: LoadedRelease, conv: Conversation, ref: TemplateReference) -> TemplateReference:
    """The reference with the zones this zone-signature entry's documented behavior gives."""
    return replace(ref, zones=ZONE_SIGNATURES[entry.signature](release, conv, ref))


def describe(entry: KnownDifference) -> dict[str, object]:
    return {"id": entry.id, "signature": entry.signature, "docs": list(entry.docs)}
