"""Model-level template properties: facts about a model's chat template, judged before any framework.

The first property is prefix breaks (docs/DECISIONS.md, "Template properties
are judged before frameworks"): boundaries where rendering a conversation up
to an assistant turn gives something that is not the start of rendering the
whole conversation. Qwen3's own template has them (it drops the empty think
block from earlier assistant turns).

A framework that breaks the prefix exactly as its template does passes
t1.multiturn_prefix, but that pass must not read as "prefix preserved". So the
verdict names the property it inherits (`Verdict.template_properties`), and
the report shows each property once per model, in a model-level section.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from parity.engine.evidence import decode_each, first_divergence
from parity.fixtures import Conversation
from parity.references.template import template_sha256, template_token_ids
from parity.release import LoadedRelease

# (tokens from the break to the end of the prefix, tokens removed from the prefix, tokens inserted from the full)
Break = tuple[int, tuple[int, ...], tuple[int, ...]]


@dataclass(frozen=True)
class TemplateProperty:
    id: str
    model: str
    template_source: str
    template_sha256: str
    kind: str
    summary: str
    details: tuple[dict[str, Any], ...]


def boundaries(conv: Conversation) -> list[int]:
    """Prefix lengths j < len(messages) whose last message is an assistant turn."""
    return [j for j in range(1, len(conv.messages)) if conv.messages[j - 1]["role"] == "assistant"]


def break_signature(prefix: Sequence[int], full: Sequence[int]) -> Break | None:
    """How `full` re-renders `prefix`, or None when `prefix` is exactly the start of `full`.

    Anchored at both ends: c is the length of the common prefix; the longest tail
    of `prefix` that occurs in `full` at or after c (its earliest occurrence, at
    m) re-synchronizes them. The later turns appended after the prefix are never
    part of a break.
    """
    prefix, full = tuple(prefix), tuple(full)
    c = first_divergence(prefix, full)
    if c is None or c == len(prefix):
        return None
    tail = prefix[c:]
    kept, m = 0, c
    for size in range(len(tail), 0, -1):  # longest tail first, then its earliest occurrence
        found = _find(full, tail[-size:], start=c)
        if found is not None:
            kept, m = size, found
            break
    return (len(prefix) - c, tail[: len(tail) - kept], full[c:m])


def _find(haystack: tuple[int, ...], needle: tuple[int, ...], start: int) -> int | None:
    last = len(haystack) - len(needle)
    first = needle[0]
    for i in range(start, last + 1):
        if haystack[i] == first and haystack[i : i + len(needle)] == needle:
            return i
    return None


def describe_break(tokenizer: Any, signature: Break | None) -> dict[str, Any] | None:
    if signature is None:
        return None
    from_end, removed, inserted = signature
    return {
        "tokens_from_end": from_end,
        "removed": decode_each(tokenizer, removed),
        "inserted": decode_each(tokenizer, inserted),
    }


def prefix_break_property_id(release: LoadedRelease, template_sha: str) -> str:
    r = release.release
    return f"{r.repo_id}@{r.revision[:12]}#prefix-breaks@{template_sha[:12]}"


def prefix_break_property(
    release: LoadedRelease, template: str, source: str, conversations: Iterable[Conversation]
) -> TemplateProperty | None:
    """The template's prefix breaks on these conversations, or None if it preserves every prefix."""
    tokenizer = release.tokenizer
    details = []
    for conv in conversations:
        full = template_token_ids(tokenizer, conv, template)
        breaks = []
        for j in boundaries(conv):
            signature = break_signature(template_token_ids(tokenizer, conv.prefix(j), template), full)
            if signature is not None:
                breaks.append({"boundary": j, "edit": describe_break(tokenizer, signature)})
        if breaks:
            details.append({"fixture": conv.id, "breaks": breaks})
    if not details:
        return None
    sha = template_sha256(template)
    n = sum(len(d["breaks"]) for d in details)
    return TemplateProperty(
        id=prefix_break_property_id(release, sha),
        model=release.release.spec,
        template_source=source,
        template_sha256=sha,
        kind="prefix_breaks",
        summary=(
            f"Not prefix-preserving: at {n} boundary(ies) across {len(details)} fixture(s), rendering a "
            "conversation up to an assistant turn is not the start of rendering the whole conversation. "
            "Frameworks that use this template inherit these breaks."
        ),
        details=tuple(details),
    )
