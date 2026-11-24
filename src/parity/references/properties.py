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

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from parity.engine.evidence import decode_each, first_divergence
from parity.fixtures import Conversation
from parity.references.template import (
    MarkerDisagreementError,
    TemplateReferenceError,
    prefill_missing_turns,
    render_reference,
    template_sha256,
    template_token_ids,
)
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


def marker_disagreement_property_id(release: LoadedRelease, template_sha: str) -> str:
    r = release.release
    return f"{r.repo_id}@{r.revision[:12]}#marker-disagreement@{template_sha[:12]}"


def marker_disagreement_property(
    release: LoadedRelease, template: str, source: str, conversations: Iterable[Conversation]
) -> TemplateProperty | None:
    """Where the template's own generation markers disagree with the prefix diff on a non-neutral token.

    Every check on such a fixture is inconclusive; this puts the disagreement on the report card as a
    possible model-template finding instead of leaving it in reason text.
    """
    details = []
    for conv in conversations:
        try:
            render_reference(release, conv, template, source)
        except MarkerDisagreementError as exc:
            details.append({"fixture": conv.id, "count": len(exc.positions), "positions": exc.positions[:16]})
        except TemplateReferenceError:
            continue  # refused for another reason: not a marker disagreement
    if not details:
        return None
    sha = template_sha256(template)
    return TemplateProperty(
        id=marker_disagreement_property_id(release, sha),
        model=release.release.spec,
        template_source=source,
        template_sha256=sha,
        kind="marker_disagreement",
        summary=(
            f"The template's generation markers disagree with the prefix diff on {len(details)} fixture(s): "
            "they mark a must-not-train token as trained, or leave a must-train token out. Every check on "
            "those fixtures is inconclusive. Possible model-template finding: send to /triage."
        ),
        details=tuple(details),
    )


def prefill_property_id(release: LoadedRelease, template_sha: str) -> str:
    r = release.release
    return f"{r.repo_id}@{r.revision[:12]}#prefill-missing@{template_sha[:12]}"


def prefill_property(
    release: LoadedRelease, template: str, source: str, conversations: Iterable[Conversation]
) -> TemplateProperty | None:
    """Assistant turns that render without the generation prompt's pre-fill (DeepSeek V4.1's `<think>`).

    The reference anchors those turns' content where the generation prompt and the history render stop
    agreeing. It is also a train/serve mismatch for Tier 3: at inference the model is prompted with the
    pre-fill; in training history it never sees it.
    """
    details = []
    for conv in conversations:
        try:
            missing = prefill_missing_turns(release, conv, template)
        except TemplateReferenceError:
            continue
        if missing:
            details.append({"fixture": conv.id, "turns": [{"message_index": k, "prefill": p} for k, p in missing]})
    if not details:
        return None
    sha = template_sha256(template)
    prefills = sorted({t["prefill"] for d in details for t in d["turns"]})
    return TemplateProperty(
        id=prefill_property_id(release, sha),
        model=release.release.spec,
        template_source=source,
        template_sha256=sha,
        kind="prefill_missing_from_history",
        summary=(
            f"The generation prompt ends in a pre-fill ({', '.join(json.dumps(p, ensure_ascii=False) for p in prefills)}) "
            "that earlier assistant turns render without. Where the reference can delimit those turns, it "
            "anchors their content where the two renders stop agreeing. Also a train/serve mismatch for "
            "Tier 3: served turns start with the pre-fill, trained history turns don't."
        ),
        details=tuple(details),
    )


def template_properties_for(
    release: LoadedRelease, template: str, source: str, conversations: Iterable[Conversation]
) -> list[TemplateProperty]:
    """Every model-level property of this template on these conversations, for the report."""
    conversations = list(conversations)
    found = [
        prefix_break_property(release, template, source, conversations),
        marker_disagreement_property(release, template, source, conversations),
        prefill_property(release, template, source, conversations),
    ]
    return [p for p in found if p is not None]
