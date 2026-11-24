"""Tier 1 reference: the tokens and assistant mask a model's own chat template defines.

This is the independent answer every Tier 1 check compares a framework against.
It only ever calls the model's own template through transformers; it never
re-implements a template.

Tokens
    `input_ids` come straight from `tokenizer.apply_chat_template(messages,
    tools=...)` with default options, using the template the model ships. An
    explicit `chat_template` override exists only for cross-checks (for
    example TRL's training template for the same family).

Assistant spans: the prefix diff, cross-checked against generation markers
    Each assistant message k is located inside the full rendering F using
    only further renders of the same template (the prefix diff, below). This
    is the reference for every template, with or without markers.

    If the template also wraps assistant output in
    `{% generation %}...{% endgeneration %}`, the mask transformers reads from
    those markers (`return_assistant_tokens_mask`) is cross-checked against
    the prefix diff. Neutral tokens (below) never count as a disagreement.
    Any other disagreement raises MarkerDisagreementError: the template's own
    markers may be wrong, which is a possible model-template finding for
    /triage, and the check verdict becomes inconclusive. (This revises the
    earlier rule that markers define the mask; see docs/DECISIONS.md.)

    The prefix diff:

         message start = len(render(messages[:k]))
         content start = len(render(messages[:k], add_generation_prompt=True))
         message end   = message start + len(F) - len(render(messages without k))

       Both prefix renders must be exact prefixes of F, and cutting the message
       out of F at its start must give exactly the "without k" render.
       Otherwise we raise instead of guessing.

       One narrow exception: a template whose generation prompt ends in a
       pre-fill (DeepSeek V4.1: `<｜Assistant｜><think>`) that history turns
       render without. Content is anchored where the two renders stop agreeing,
       only if the prompt is exactly the shared tokens plus the pre-fill's own
       tokens and the same generation prompt appears verbatim at another
       assistant turn (see `_prefill_anchor`). The span records the missing
       pre-fill, and the template gets a "pre-fill missing from history"
       property (parity.references.properties).

       The supervised span runs from the content start through the last stop
       token inside the message: the tokens a model must generate before it
       stops. Anything after that stop token inside the message (usually a
       newline separator) is recorded as `token_message_end` but not
       supervised, because the model never generates it.

       Why not simply render messages[:k] and messages[:k+1] and diff them?
       Many templates (Qwen3 included) render a turn differently when it is
       the last message than when later messages follow. That prefix render
       then does not match the full conversation, so every boundary here is
       anchored in F itself.

Zones (docs/DECISIONS.md, "Mask zones")
    Every reference token is in exactly one zone. must-train: from the
    content start through the turn's last stop token. neutral: after that
    stop token, to the end of the message (Qwen3: the newline after
    <|im_end|>). must-not-train: everything else.

Every ambiguity raises TemplateReferenceError. The check engine turns that into
an `inconclusive` verdict, never a pass.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import transformers
from transformers import PreTrainedTokenizerBase

from parity.fixtures import Conversation
from parity.release import LoadedRelease

_GENERATION_TAG = re.compile(r"\{%-?\s*generation\s*-?%\}")


class TemplateReferenceError(RuntimeError):
    """The reference cannot be computed without guessing."""


class MarkerDisagreementError(TemplateReferenceError):
    """The template's generation markers disagree with the prefix diff on a non-neutral token."""

    def __init__(self, message: str, positions: list[dict[str, Any]], template_sha256: str = ""):
        super().__init__(message)
        self.positions = positions
        self.template_sha256 = template_sha256


class Zone(IntEnum):
    MUST_NOT_TRAIN = 0
    MUST_TRAIN = 1
    NEUTRAL = 2


def template_sha256(template: str) -> str:
    return hashlib.sha256(template.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class AssistantSpan:
    """Where one assistant message sits in the reference token sequence."""

    message_index: int
    token_start: int  # first supervised token (right after the generation prompt)
    token_end: int  # one past the turn's last stop token
    token_message_end: int  # one past the whole rendered message (may include a separator)
    token_message_start: int | None = None  # first token of the message (its role header); None if mid-token
    prefill_missing: str | None = None  # the generation prompt's pre-fill, when this turn renders without it


@dataclass(frozen=True)
class TemplateReference:
    release: str
    fixture_id: str
    template_source: str  # "model", or a label like "trl@1.18.2:qwen3_training.jinja"
    template_sha256: str
    mask_method: str  # always "prefix_diff"
    marker_check: str  # "agrees" (markers cross-checked) or "no_markers"
    text: str
    input_ids: tuple[int, ...]
    assistant_mask: tuple[int, ...]  # 1 on must-train tokens
    zones: tuple[Zone, ...]
    spans: tuple[AssistantSpan, ...]
    prefix_breaks: tuple[int, ...]  # see prefix_breaks()
    transformers_version: str


def has_generation_markers(template: str) -> bool:
    return bool(_GENERATION_TAG.search(template))


def model_chat_template(tokenizer: PreTrainedTokenizerBase) -> str:
    template = tokenizer.chat_template
    if isinstance(template, dict):
        if "default" not in template:
            raise TemplateReferenceError(f"several named templates and no 'default': {sorted(template)}")
        template = template["default"]
    if not template:
        raise TemplateReferenceError("tokenizer has no chat template")
    return template


def render_reference(
    loaded: LoadedRelease,
    fixture: Conversation,
    chat_template: str | None = None,
    template_source: str = "model",
) -> TemplateReference:
    """Compute the reference tokens, zones and assistant mask for one conversation."""
    tokenizer = loaded.tokenizer
    template = chat_template if chat_template is not None else model_chat_template(tokenizer)
    _reject_special_tokens_in_content(tokenizer, fixture)

    text = render_text(tokenizer, fixture.messages, fixture.tools, template)
    input_ids = template_token_ids(tokenizer, fixture, template)
    ids_from_text, _ = _tokenize_with_offsets(tokenizer, text)
    if ids_from_text != input_ids:
        raise TemplateReferenceError("tokenizing the rendered text does not reproduce apply_chat_template's ids")

    spans = assistant_spans_by_prefix_diff(loaded, fixture, template)
    zones = zones_from_spans(len(input_ids), spans)
    marker_check = "no_markers"
    if has_generation_markers(template):
        _cross_check_markers(tokenizer, fixture, template, input_ids, zones)
        marker_check = "agrees"

    return TemplateReference(
        release=loaded.release.spec,
        fixture_id=fixture.id,
        template_source=template_source,
        template_sha256=template_sha256(template),
        mask_method="prefix_diff",
        marker_check=marker_check,
        text=text,
        input_ids=tuple(input_ids),
        assistant_mask=tuple(1 if z == Zone.MUST_TRAIN else 0 for z in zones),
        zones=zones,
        spans=spans,
        prefix_breaks=prefix_breaks(tokenizer, fixture, template),
        transformers_version=transformers.__version__,
    )


def zones_from_spans(length: int, spans: tuple[AssistantSpan, ...]) -> tuple[Zone, ...]:
    zones = [Zone.MUST_NOT_TRAIN] * length
    for span in spans:
        zones[span.token_start : span.token_end] = [Zone.MUST_TRAIN] * (span.token_end - span.token_start)
        zones[span.token_end : span.token_message_end] = [Zone.NEUTRAL] * (span.token_message_end - span.token_end)
    return tuple(zones)


def _cross_check_markers(
    tokenizer: PreTrainedTokenizerBase,
    conv: Conversation,
    template: str,
    input_ids: list[int],
    zones: tuple[Zone, ...],
) -> None:
    marker_ids, marker_mask = assistant_mask_from_markers(tokenizer, conv, template)
    if marker_ids != input_ids:
        raise TemplateReferenceError("the generation-marker render produced different ids")
    disagree = [
        i
        for i, (zone, marked) in enumerate(zip(zones, marker_mask))
        if (zone == Zone.MUST_TRAIN and not marked) or (zone == Zone.MUST_NOT_TRAIN and marked)
    ]
    if disagree:
        positions = [
            {
                "index": i,
                "token": tokenizer.decode([input_ids[i]], skip_special_tokens=False),
                "zone": zones[i].name.lower(),
                "marked": bool(marker_mask[i]),
            }
            for i in disagree
        ]
        first = positions[0]
        raise MarkerDisagreementError(
            f"the template's generation markers disagree with the prefix-diff reference on {len(disagree)} "
            f"token(s); first at {first['index']} ({first['token']!r}, zone {first['zone']}, "
            f"marked {first['marked']}). Possible model-template finding: send to /triage.",
            positions,
            template_sha256(template),
        )


def template_token_ids(tokenizer: PreTrainedTokenizerBase, conv: Conversation, chat_template: str) -> list[int]:
    """Token ids of a conversation rendered with `chat_template`, exactly as apply_chat_template returns them."""
    encoded = tokenizer.apply_chat_template(
        conv.messages, tools=conv.tools, chat_template=chat_template, tokenize=True, return_dict=True
    )
    return list(encoded["input_ids"])


def assistant_mask_from_markers(
    tokenizer: PreTrainedTokenizerBase, fixture: Conversation, chat_template: str
) -> tuple[list[int], list[int]]:
    """Return (input_ids, assistant mask) as declared by the template's generation markers."""
    if not has_generation_markers(chat_template):
        raise TemplateReferenceError("template has no {% generation %} markers")
    out = tokenizer.apply_chat_template(
        fixture.messages,
        tools=fixture.tools,
        chat_template=chat_template,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
    )
    return list(out["input_ids"]), list(out["assistant_masks"])


def assistant_spans_by_prefix_diff(
    loaded: LoadedRelease, fixture: Conversation, chat_template: str | None = None
) -> tuple[AssistantSpan, ...]:
    """Locate every assistant message in the full rendering (see module docstring)."""
    tokenizer = loaded.tokenizer
    template = chat_template if chat_template is not None else model_chat_template(tokenizer)
    messages, tools = fixture.messages, fixture.tools

    full = render_text(tokenizer, messages, tools, template)
    ids, offsets = _tokenize_with_offsets(tokenizer, full)

    renders, seen_in_full = _generation_prompt_renders(tokenizer, fixture, template, full)

    spans = []
    for k in fixture.assistant_indices:
        before, prompt = renders[k]
        without = render_text(tokenizer, messages[:k] + messages[k + 1 :], tools, template)
        msg_start, prefill = len(before), None
        if full.startswith(prompt):
            content_start = len(prompt)
        else:
            anchored = _prefill_anchor(tokenizer, full, before, prompt, seen_in_full)
            if anchored is None:
                raise TemplateReferenceError(
                    f"message {k}: generation prompt render is not a prefix of the full render"
                )
            content_start, prefill = anchored
        msg_end = msg_start + len(full) - len(without)
        if msg_end <= msg_start or full[:msg_start] + full[msg_end:] != without:
            raise TemplateReferenceError(
                f"message {k}: cutting it out of the full render does not give the render without it"
            )
        if not msg_start <= content_start <= msg_end:
            raise TemplateReferenceError(f"message {k}: generation prompt runs past the end of the message")
        if content_start == msg_start and not has_generation_markers(template):
            # Without a generation prompt the assistant's header would count as content
            # (must-train). Only the template's own markers could say otherwise.
            raise TemplateReferenceError(
                f"message {k}: the generation prompt adds nothing, so the assistant header can't be told apart "
                "from its content, and the template has no generation markers to say where content starts"
            )

        t_content = _token_boundary(offsets, content_start)
        t_msg_end = _token_boundary(offsets, msg_end)
        stops = [i for i in range(t_content, t_msg_end) if ids[i] in loaded.stop_token_ids]
        if not stops:
            raise TemplateReferenceError(f"message {k}: no stop token inside the assistant message")
        t_end = stops[-1] + 1

        _check_decodes_exactly(tokenizer, ids, offsets, full, t_content, t_end, k)
        try:
            t_msg_start: int | None = _token_boundary(offsets, msg_start)
        except TemplateReferenceError:
            t_msg_start = None  # only the header planted fault needs it; it doesn't apply then
        spans.append(AssistantSpan(k, t_content, t_end, t_msg_end, t_msg_start, prefill))
    return tuple(spans)


def _generation_prompt_renders(
    tokenizer: PreTrainedTokenizerBase, conv: Conversation, template: str, full: str
) -> tuple[dict[int, tuple[str, str]], set[str]]:
    """For each assistant message k: (render of messages[:k], the same with the generation prompt).

    Also returns the generation prompts the full render shows verbatim at the start of some assistant turn.
    """
    renders = {}
    for k in conv.assistant_indices:
        if k == 0:
            raise TemplateReferenceError("an assistant message with nothing before it is not supported")
        before = render_text(tokenizer, conv.messages[:k], conv.tools, template)
        prompt = render_text(tokenizer, conv.messages[:k], conv.tools, template, add_generation_prompt=True)
        if not full.startswith(before):
            raise TemplateReferenceError(f"message {k}: render of messages[:{k}] is not a prefix of the full render")
        if not prompt.startswith(before):
            raise TemplateReferenceError(f"message {k}: the generation prompt render doesn't extend messages[:{k}]")
        renders[k] = (before, prompt)
    seen_in_full = {prompt[len(before) :] for before, prompt in renders.values() if full.startswith(prompt)}
    return renders, seen_in_full


def prefill_missing_turns(
    loaded: LoadedRelease, conv: Conversation, chat_template: str | None = None
) -> list[tuple[int, str]]:
    """Assistant turns that render without the generation prompt's trailing pre-fill: [(message index, pre-fill)].

    Uses only the generation-prompt renders, so it works even where the full
    reference refuses for another reason (DeepSeek V4.1 merges the user
    messages around a removed assistant turn, so no cut-out delimits it).
    """
    tokenizer = loaded.tokenizer
    template = chat_template if chat_template is not None else model_chat_template(tokenizer)
    full = render_text(tokenizer, conv.messages, conv.tools, template)
    renders, seen_in_full = _generation_prompt_renders(tokenizer, conv, template, full)
    found = []
    for k, (before, prompt) in renders.items():
        if not full.startswith(prompt):
            anchored = _prefill_anchor(tokenizer, full, before, prompt, seen_in_full)
            if anchored is not None:
                found.append((k, anchored[1]))
    return found


def _prefill_anchor(
    tokenizer: PreTrainedTokenizerBase, full: str, before: str, prompt: str, seen_in_full: set[str]
) -> tuple[int, str] | None:
    """Where message content starts when the generation prompt's trailing pre-fill is missing from history.

    Allowed only in this narrow case (anything else returns None, and the
    reference refuses):
    - the prompt's tokens are the tokens it shares with the full render,
      followed by a pre-fill whose own tokens are exactly the rest of the
      prompt's tokens (token for token);
    - the shared tokens cover all of messages[:k] and at least one token of the
      generation prompt (the role header), and end on a token boundary of the
      full render;
    - the same generation prompt appears verbatim at the start of some other
      assistant turn in the full render, so the pre-fill is something this
      template really emits there, not just a different header.
    Returns (content start, pre-fill text).
    """
    if prompt[len(before) :] not in seen_in_full:
        return None
    p_ids, p_offsets = _tokenize_with_offsets(tokenizer, prompt)
    f_ids, f_offsets = _tokenize_with_offsets(tokenizer, full)
    shared = 0
    while shared < min(len(p_ids), len(f_ids)) and p_ids[shared] == f_ids[shared]:
        shared += 1
    if shared == 0 or shared == len(p_ids):
        return None
    shared_end = p_offsets[shared - 1][1]
    if shared_end <= len(before) or f_offsets[shared - 1][1] != shared_end or full[:shared_end] != prompt[:shared_end]:
        return None
    prefill = prompt[shared_end:]
    if list(tokenizer(prefill, add_special_tokens=False)["input_ids"]) != list(p_ids[shared:]):
        return None
    return shared_end, prefill


def prefix_breaks(
    tokenizer: PreTrainedTokenizerBase, fixture: Conversation, chat_template: str | None = None
) -> tuple[int, ...]:
    """Boundaries j where render(messages[:j]) is NOT a prefix of the full render.

    An empty result means the template is prefix-preserving on this
    conversation: appending messages never changes how earlier ones render.
    A non-empty result is a property of the template itself (for example,
    reasoning dropped from earlier turns), not of any framework.
    """
    template = chat_template if chat_template is not None else model_chat_template(tokenizer)
    messages, tools = fixture.messages, fixture.tools
    full = render_text(tokenizer, messages, tools, template)
    return tuple(j for j in range(1, len(messages)) if not full.startswith(render_text(tokenizer, messages[:j], tools, template)))


def render_text(
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    chat_template: str,
    add_generation_prompt: bool = False,
) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tools=tools,
        chat_template=chat_template,
        add_generation_prompt=add_generation_prompt,
        tokenize=False,
    )


def _tokenize_with_offsets(tokenizer: PreTrainedTokenizerBase, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(enc["input_ids"])
    offsets = [(int(s), int(e)) for s, e in enc["offset_mapping"]]
    for i in range(1, len(offsets)):
        if offsets[i][0] < offsets[i - 1][0] or offsets[i][1] < offsets[i - 1][1]:
            raise TemplateReferenceError(f"token offsets go backwards at token {i}")
    return ids, offsets


def _token_boundary(offsets: list[tuple[int, int]], pos: int) -> int:
    """Index of the first token starting at or after char `pos`.

    Raises if a token starts before `pos` but ends after it, because then the
    boundary falls inside a token and the mask would have to guess.
    """
    for i, (start, end) in enumerate(offsets):
        if start >= pos:
            return i
        if end > pos:
            raise TemplateReferenceError(f"token {i} (chars {start}-{end}) straddles boundary at char {pos}")
    return len(offsets)


def _check_decodes_exactly(
    tokenizer: PreTrainedTokenizerBase,
    ids: list[int],
    offsets: list[tuple[int, int]],
    text: str,
    start: int,
    end: int,
    message_index: int,
) -> None:
    expected = text[offsets[start][0] : offsets[end - 1][1]]
    got = tokenizer.decode(ids[start:end], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    if got != expected:
        raise TemplateReferenceError(f"message {message_index}: span tokens decode to {got!r}, text has {expected!r}")


def _reject_special_tokens_in_content(tokenizer: PreTrainedTokenizerBase, fixture: Conversation) -> None:
    """Special-token text inside a message would make span boundaries ambiguous."""
    specials = [t for t in tokenizer.all_special_tokens if t]
    for i, message in enumerate(fixture.messages):
        for field in ("content", "reasoning_content"):
            value = message.get(field)
            if isinstance(value, str):
                hits = [t for t in specials if t in value]
                if hits:
                    raise TemplateReferenceError(f"message {i} {field} contains special tokens {hits}")
