"""Tier 1 reference: the tokens and assistant mask a model's own chat template defines.

This is the independent answer every Tier 1 check compares a framework against.
It only ever calls the model's own template through transformers; it never
re-implements a template.

Tokens
    `input_ids` come straight from `tokenizer.apply_chat_template(messages,
    tools=...)` with default options, using the template the model ships. An
    explicit `chat_template` override exists only for cross-checks (for
    example TRL's training template for the same family).

Assistant spans: two methods
    1. Generation markers. If the template wraps assistant output in
       `{% generation %}...{% endgeneration %}`, its author has declared what is
       supervised, and transformers' `return_assistant_tokens_mask` reads it.

    2. Prefix diff, for templates without markers. Each assistant message k is
       located inside the full rendering F using only further renders of the
       same template:

         message start = len(render(messages[:k]))
         content start = len(render(messages[:k], add_generation_prompt=True))
         message end   = message start + len(F) - len(render(messages without k))

       Both prefix renders must be exact prefixes of F, and cutting the message
       out of F at its start must give exactly the "without k" render.
       Otherwise we raise instead of guessing.

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

Every ambiguity raises TemplateReferenceError. The check engine turns that into
an `inconclusive` verdict, never a pass.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import transformers
from transformers import PreTrainedTokenizerBase

from parity.fixtures import Fixture
from parity.release import LoadedRelease

_GENERATION_TAG = re.compile(r"\{%-?\s*generation\s*-?%\}")


class TemplateReferenceError(RuntimeError):
    """The reference cannot be computed without guessing."""


@dataclass(frozen=True)
class AssistantSpan:
    """Where one assistant message sits in the reference token sequence."""

    message_index: int
    token_start: int  # first supervised token (right after the generation prompt)
    token_end: int  # one past the turn's last stop token
    token_message_end: int  # one past the whole rendered message (may include a separator)


@dataclass(frozen=True)
class TemplateReference:
    release: str
    fixture_id: str
    template_source: str  # "model", or a label like "trl@1.18.2:qwen3_training.jinja"
    mask_method: str  # "generation_markers" or "prefix_diff"
    text: str
    input_ids: tuple[int, ...]
    assistant_mask: tuple[int, ...]
    spans: tuple[AssistantSpan, ...]  # empty when mask_method == "generation_markers"
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
    fixture: Fixture,
    chat_template: str | None = None,
    template_source: str = "model",
) -> TemplateReference:
    """Compute the reference tokens and assistant mask for one fixture."""
    tokenizer = loaded.tokenizer
    template = chat_template if chat_template is not None else model_chat_template(tokenizer)
    _reject_special_tokens_in_content(tokenizer, fixture)

    text = _render(tokenizer, fixture.messages, fixture.tools, template)
    encoded = tokenizer.apply_chat_template(
        fixture.messages, tools=fixture.tools, chat_template=template, tokenize=True, return_dict=True
    )
    input_ids = list(encoded["input_ids"])
    ids_from_text, _ = _tokenize_with_offsets(tokenizer, text)
    if ids_from_text != input_ids:
        raise TemplateReferenceError("tokenizing the rendered text does not reproduce apply_chat_template's ids")

    if has_generation_markers(template):
        marker_ids, mask = assistant_mask_from_markers(tokenizer, fixture, template)
        if marker_ids != input_ids:
            raise TemplateReferenceError("the generation-marker render produced different ids")
        spans: tuple[AssistantSpan, ...] = ()
        method = "generation_markers"
    else:
        spans = assistant_spans_by_prefix_diff(loaded, fixture, template)
        mask = [0] * len(input_ids)
        for span in spans:
            mask[span.token_start : span.token_end] = [1] * (span.token_end - span.token_start)
        method = "prefix_diff"

    return TemplateReference(
        release=loaded.release.spec,
        fixture_id=fixture.id,
        template_source=template_source,
        mask_method=method,
        text=text,
        input_ids=tuple(input_ids),
        assistant_mask=tuple(mask),
        spans=spans,
        prefix_breaks=prefix_breaks(tokenizer, fixture, template),
        transformers_version=transformers.__version__,
    )


def assistant_mask_from_markers(
    tokenizer: PreTrainedTokenizerBase, fixture: Fixture, chat_template: str
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
    loaded: LoadedRelease, fixture: Fixture, chat_template: str | None = None
) -> tuple[AssistantSpan, ...]:
    """Locate every assistant message in the full rendering (see module docstring)."""
    tokenizer = loaded.tokenizer
    template = chat_template if chat_template is not None else model_chat_template(tokenizer)
    messages, tools = fixture.messages, fixture.tools

    full = _render(tokenizer, messages, tools, template)
    ids, offsets = _tokenize_with_offsets(tokenizer, full)

    spans = []
    for k in fixture.assistant_indices:
        if k == 0:
            raise TemplateReferenceError("an assistant message with nothing before it is not supported")
        before = _render(tokenizer, messages[:k], tools, template)
        prompt = _render(tokenizer, messages[:k], tools, template, add_generation_prompt=True)
        without = _render(tokenizer, messages[:k] + messages[k + 1 :], tools, template)

        if not full.startswith(before):
            raise TemplateReferenceError(f"message {k}: render of messages[:{k}] is not a prefix of the full render")
        if not full.startswith(prompt):
            raise TemplateReferenceError(f"message {k}: generation prompt render is not a prefix of the full render")
        msg_start, content_start = len(before), len(prompt)
        msg_end = msg_start + len(full) - len(without)
        if msg_end <= msg_start or full[:msg_start] + full[msg_end:] != without:
            raise TemplateReferenceError(
                f"message {k}: cutting it out of the full render does not give the render without it"
            )
        if not msg_start <= content_start <= msg_end:
            raise TemplateReferenceError(f"message {k}: generation prompt runs past the end of the message")

        t_content = _token_boundary(offsets, content_start)
        t_msg_end = _token_boundary(offsets, msg_end)
        stops = [i for i in range(t_content, t_msg_end) if ids[i] in loaded.stop_token_ids]
        if not stops:
            raise TemplateReferenceError(f"message {k}: no stop token inside the assistant message")
        t_end = stops[-1] + 1

        _check_decodes_exactly(tokenizer, ids, offsets, full, t_content, t_end, k)
        spans.append(AssistantSpan(k, t_content, t_end, t_msg_end))
    return tuple(spans)


def prefix_breaks(
    tokenizer: PreTrainedTokenizerBase, fixture: Fixture, chat_template: str | None = None
) -> tuple[int, ...]:
    """Boundaries j where render(messages[:j]) is NOT a prefix of the full render.

    An empty result means the template is prefix-preserving on this
    conversation: appending messages never changes how earlier ones render.
    A non-empty result is a property of the template itself (for example,
    reasoning dropped from earlier turns), not of any framework.
    """
    template = chat_template if chat_template is not None else model_chat_template(tokenizer)
    messages, tools = fixture.messages, fixture.tools
    full = _render(tokenizer, messages, tools, template)
    return tuple(j for j in range(1, len(messages)) if not full.startswith(_render(tokenizer, messages[:j], tools, template)))


def _render(
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


def _reject_special_tokens_in_content(tokenizer: PreTrainedTokenizerBase, fixture: Fixture) -> None:
    """Special-token text inside a message would make span boundaries ambiguous."""
    specials = [t for t in tokenizer.all_special_tokens if t]
    for i, message in enumerate(fixture.messages):
        for field in ("content", "reasoning_content"):
            value = message.get(field)
            if isinstance(value, str):
                hits = [t for t in specials if t in value]
                if hits:
                    raise TemplateReferenceError(f"message {i} {field} contains special tokens {hits}")
