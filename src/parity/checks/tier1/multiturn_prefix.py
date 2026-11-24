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
framework inherits from its template are attributed to the template: the
verdict records `prefix_preserved: false` and names the model-level template
property it inherits (`parity.references.properties`, shown once per model in
the report), so the pass never reads as "prefix preserved".

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
from typing import Any

from parity.adapters.base import Declaration, Dropped
from parity.checks.common import (
    Tier1Check,
    declared_reference,
    is_contradicted,
    reference_info,
    template_status,
    token_ids_on,
    unaligned_verdict,
)
from parity.engine.plants import Plant, PlantedAdapter
from parity.engine.verdict import Verdict
from parity.fixtures import Conversation, Fixture
from parity.references.properties import (
    boundaries,
    break_signature,
    describe_break,
    prefix_break_property_id,
)
from parity.references.template import template_sha256, template_token_ids
from parity.release import LoadedRelease


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
        ref = declared_reference(release, fixture, declaration)
        info = reference_info(release, ref)
        full = adapter.encode(fixture, self.base_cfg)
        if isinstance(full, Dropped):
            return self.verdict(
                adapter,
                fixture,
                "fail",
                kind="bug",
                reference=info,
                declared=declaration,
                evidence={"dropped": full.reason, "at": "full"},
            )
        status = template_status(release, fixture, declaration, full.input_ids)
        if is_contradicted(status):
            return unaligned_verdict(
                self, adapter, release, fixture, declaration, full.input_ids, ref.input_ids, reference=info
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
                    "ok": framework_break is None or framework_break == template_break,
                    "template_breaks": template_break is not None,
                    "framework_breaks": framework_break is not None,
                    "same_break_as_template": framework_break == template_break,
                    "template_edits": describe_break(tokenizer, template_break),
                    "framework_edits": describe_break(tokenizer, framework_break),
                }
            )

        inherited = [c["boundary"] for c in checked if c.get("framework_breaks") and c["ok"]]
        conventions = {
            "boundaries": [c["boundary"] for c in checked],
            "template_breaks_at": [c["boundary"] for c in checked if c.get("template_breaks")],
            "framework_breaks_at": [c["boundary"] for c in checked if c.get("framework_breaks")],
            "prefix_preserved": not any(c.get("framework_breaks") for c in checked),
        }
        evidence: dict[str, Any] = {"boundaries": checked}
        properties = []
        if inherited:
            prop = prefix_break_property_id(release, template_sha256(template))
            properties.append(prop)
            evidence["inherited_from_template"] = (
                f"The prefix is NOT preserved at boundaries {inherited}. The framework breaks it exactly as its "
                f"declared template does, so this is attributed to the template: see template property {prop}."
            )
        result = "pass" if all(c["ok"] for c in checked) else "fail"
        return self.verdict(
            adapter,
            fixture,
            result,
            kind="bug" if result == "fail" else None,
            reference=info,
            conventions=conventions,
            declared=declaration,
            verified={"chat_template": status},
            template_properties=properties,
            evidence=evidence,
        )

    def plants(self, release: LoadedRelease, fixture: Fixture, declaration: Declaration) -> list[Plant]:
        if not boundaries(fixture):
            return []
        out = []
        has_boolean = any(_has_boolean_tool_argument(m) for m in fixture.messages[:-1] if m["role"] == "assistant")
        if has_boolean and _renders(release, _rewrite_history(fixture, _arguments_as_python_repr), declaration):
            # Skipped when the declared template refuses string arguments (Inkling's and Hy4's do):
            # then no framework using it can produce this fault, and the planted run would only crash.
            out.append(
                _history_plant(
                    "history_false_as_python",
                    "re-render earlier tool-call arguments as a Python repr (false -> False)",
                    _arguments_as_python_repr,
                    origin="real-world",
                    source="Prime Intellect, May 2026: history re-rendered from Python objects turned JSON false into False",
                )
            )
        out.append(
            _history_plant(
                "history_edit",
                "re-render earlier assistant turns with one character changed",
                _edit_one_char,
                origin="synthetic",
            )
        )
        return out


def _rewrite_history(conv: Conversation, rewrite: Any) -> Conversation:
    """The conversation with every assistant message that isn't the last one rewritten."""
    messages = copy.deepcopy(conv.messages)
    for i, message in enumerate(messages[:-1]):
        if message["role"] == "assistant":
            messages[i] = rewrite(message)
    return conv.with_messages(messages)


def _renders(release: LoadedRelease, conv: Conversation, declaration: Declaration) -> bool:
    """Can the declared template render this conversation at all? Decided from the template, not the framework."""
    try:
        template_token_ids(release.tokenizer, conv, declaration.chat_template)
    except Exception:  # noqa: BLE001 - a template that refuses the input is exactly what we're asking about
        return False
    return True


def _history_plant(name: str, description: str, rewrite: Any, *, origin: str, source: str | None = None) -> Plant:
    """A fault in how history is re-rendered: rewrite every assistant message that isn't the last one."""
    return Plant(
        name,
        description,
        lambda adapter: PlantedAdapter(adapter, name, before_encode=lambda conv: _rewrite_history(conv, rewrite)),
        origin=origin,
        source=source,
    )


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
