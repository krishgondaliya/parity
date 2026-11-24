"""t1.multiturn_prefix: does the framework render earlier turns the same way once later turns follow?

Bug class: a framework re-renders conversation history differently from how
it rendered the same turns when they were the latest ones (for example JSON
`false` re-rendered as Python `False`, Prime Intellect's example), so the
tokens a model is trained on for its history never match the tokens it
actually produced.

Reference: the template the framework declares it uses, rendered on the full
conversation and on each prefix `messages[:j]`.

Boundaries: every j < len(messages) where messages[j-1] is an assistant
message, so the prefix is itself a training conversation. A fixture with no
such boundary (fewer than two assistant turns) is not applicable, decided
from the fixture alone.

Pass rule (refines docs/DECISIONS.md, "Template properties are judged before
frameworks"): at each boundary, compare the framework's encoding of the
prefix P with its encoding of the full conversation F, and the template's
render of the prefix with its render of the full conversation. A break is
anchored at both ends: c is the length of the common prefix of P and F; the
longest tail of P that occurs in F at or after c re-synchronizes them (its
earliest occurrence, at m). The break signature is exact: how far c is from
the end of P, the tokens of P removed (P[c:] before that tail), and the
tokens of F inserted (F[c:m]). No break when P is a prefix of F. The
framework passes a boundary if it doesn't break there, or if its break
signature is exactly the template's. Any other break fails. Breaks the
framework inherits from its template are recorded in the verdict and
attributed to the template, not to the framework.

Planted faults:
- history_false_as_python: re-render the tool-call arguments of every
  assistant message that isn't the last as a Python repr (`false` becomes
  `False`). Applies when a non-final assistant message calls a tool with a
  boolean argument (the tool_call fixture).
- history_edit: re-render every assistant message that isn't the last with
  one character changed (the first letter's case swapped, or "." for empty
  content). Synthetic, but applies to every applicable fixture.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Any

from parity.adapters.base import Declaration, Dropped
from parity.checks.common import Tier1Check, declared_reference, reference_info, token_ids_on
from parity.engine.evidence import decode_each, first_divergence
from parity.engine.plants import Plant, PlantedAdapter
from parity.engine.verdict import Verdict
from parity.fixtures import Conversation, Fixture
from parity.release import LoadedRelease

# (tokens from the break to the end of the prefix, tokens removed from the prefix, tokens inserted from the full)
Break = tuple[int, tuple[int, ...], tuple[int, ...]]


class MultiturnPrefix(Tier1Check):
    id = "t1.multiturn_prefix"

    def not_applicable(self, fixture: Fixture) -> str | None:
        if boundaries(fixture):
            return None
        return "no prefix of this conversation ends in an assistant turn before the last message"

    def run(self, adapter: Any, release: LoadedRelease, fixture: Fixture) -> Verdict:
        tokenizer = release.tokenizer
        declaration = adapter.declare(self.base_cfg)
        template = declaration.chat_template
        info = reference_info(release, declaration, declared_reference(release, fixture, declaration))
        full = adapter.encode(fixture, self.base_cfg)
        if isinstance(full, Dropped):
            return self.verdict(
                adapter, fixture, "fail", kind="bug", reference=info, evidence={"dropped": full.reason, "at": "full"}
            )

        template_full = token_ids_on(release, fixture, template)
        checked = []
        for j in boundaries(fixture):
            prefix = fixture.prefix(j)
            framework_prefix = adapter.encode(prefix, self.base_cfg)
            if isinstance(framework_prefix, Dropped):
                checked.append({"boundary": j, "ok": False, "dropped": framework_prefix.reason})
                continue
            template_break = break_signature(token_ids_on(release, prefix, template), template_full)
            framework_break = break_signature(framework_prefix.input_ids, full.input_ids)
            checked.append(
                {
                    "boundary": j,
                    "ok": not framework_break or framework_break == template_break,
                    "template_breaks": bool(template_break),
                    "framework_breaks": bool(framework_break),
                    "same_break_as_template": framework_break == template_break,
                    "template_edits": describe(tokenizer, template_break),
                    "framework_edits": describe(tokenizer, framework_break),
                }
            )

        conventions = {
            "boundaries": [c["boundary"] for c in checked],
            "template_breaks_at": [c["boundary"] for c in checked if c.get("template_breaks")],
            "framework_breaks_at": [c["boundary"] for c in checked if c.get("framework_breaks")],
        }
        result = "pass" if all(c["ok"] for c in checked) else "fail"
        return self.verdict(
            adapter,
            fixture,
            result,
            kind="bug" if result == "fail" else None,
            reference=info,
            conventions=conventions,
            evidence={"boundaries": checked},
        )

    def plants(self, release: LoadedRelease, fixture: Fixture, declaration: Declaration) -> list[Plant]:
        if not boundaries(fixture):
            return []
        out = []
        if any(_has_boolean_tool_argument(m) for m in fixture.messages[:-1] if m["role"] == "assistant"):
            out.append(
                _history_plant(
                    "history_false_as_python",
                    "re-render earlier tool-call arguments as a Python repr (false -> False)",
                    _arguments_as_python_repr,
                )
            )
        out.append(
            _history_plant("history_edit", "re-render earlier assistant turns with one character changed", _edit_one_char)
        )
        return out


def boundaries(conv: Conversation) -> list[int]:
    """Prefix lengths j < len(messages) whose last message is an assistant turn."""
    return [j for j in range(1, len(conv.messages)) if conv.messages[j - 1]["role"] == "assistant"]


def break_signature(prefix: Sequence[int], full: Sequence[int]) -> Break | None:
    """How `full` re-renders `prefix`, or None when `prefix` is exactly the start of `full`.

    The later turns appended after the prefix are never part of a break.
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


def describe(tokenizer: Any, signature: Break | None) -> dict[str, Any] | None:
    if signature is None:
        return None
    from_end, removed, inserted = signature
    return {
        "tokens_from_end": from_end,
        "removed": decode_each(tokenizer, removed),
        "inserted": decode_each(tokenizer, inserted),
    }


def _history_plant(name: str, description: str, rewrite: Any) -> Plant:
    """A fault in how history is re-rendered: rewrite every assistant message that isn't the last one."""

    def before(conv: Conversation) -> Conversation:
        messages = copy.deepcopy(conv.messages)
        for i, message in enumerate(messages[:-1]):
            if message["role"] == "assistant":
                messages[i] = rewrite(message)
        return conv.with_messages(messages)

    return Plant(name, description, lambda adapter: PlantedAdapter(adapter, name, before_encode=before))


def _has_boolean_tool_argument(message: dict[str, Any]) -> bool:
    for call in message.get("tool_calls") or []:
        arguments = call.get("function", call).get("arguments")
        if isinstance(arguments, dict) and any(isinstance(v, bool) for v in arguments.values()):
            return True
    return False


def _arguments_as_python_repr(message: dict[str, Any]) -> dict[str, Any]:
    for call in message.get("tool_calls") or []:
        function = call.get("function", call)
        if isinstance(function.get("arguments"), dict):
            function["arguments"] = repr(function["arguments"])
    return message


def _edit_one_char(message: dict[str, Any]) -> dict[str, Any]:
    content = message.get("content")
    if not isinstance(content, str) or not content:
        message["content"] = "."
        return message
    i = next((k for k, ch in enumerate(content) if ch.isalpha()), 0)
    message["content"] = content[:i] + content[i].swapcase() + content[i + 1 :]
    return message
