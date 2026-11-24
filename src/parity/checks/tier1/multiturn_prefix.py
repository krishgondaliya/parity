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

Declared property, verified by behavior: `tool_arguments`, what the framework
does with tool-call arguments given as a string. On fixtures with dict tool
arguments, the check also encodes the conversation with every argument given
as a compact JSON string ('{"celsius":false}'). A framework that passes it
through renders it verbatim; one that parses it as JSON re-serializes it like
the dict ('{"celsius": false}'). The result is recorded as
`verified.tool_arguments` and shown on the report card. A contradicted
declaration makes the verdict inconclusive (declared/actual mismatch).

Planted faults:
- history_false_as_python (synthetic): re-render the tool-call arguments of
  every assistant message that isn't the last as a Python repr (`false`
  becomes `False`). It shares a symptom with Prime Intellect's example, not
  its mechanism. Applies when a non-final assistant message calls a tool with
  a boolean argument (the tool_call fixture), the declared template can
  render string arguments, and the framework doesn't declare that it parses
  arguments as JSON (such a framework refuses a Python repr, so the fault
  can't arise in it and the planted run would only crash).
- history_false_as_python_tokens (synthetic): the same symptom, injected into
  the framework's encoded output, so it applies to every framework: in every
  assistant turn that isn't the conversation's last message, tokens spelling
  `false` are swapped for the matching `False` tokens. Applies when such a
  turn has a `false` token with a single-token `False` counterpart.
- history_edit: re-render every assistant message that isn't the last with
  one character changed (the first letter's case swapped, or "." for empty
  content). Synthetic, but applies to every applicable fixture.
"""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from typing import Any

from parity.adapters.base import Declaration, Dropped, Encoded, TrainCfg
from parity.checks.common import (
    CONSISTENT,
    CONTRADICTED,
    MISMATCH,
    Declare,
    Tier1Check,
    adapter_error,
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
from parity.references.template import AssistantSpan, TemplateReference, TemplateReferenceError, template_sha256, template_token_ids
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
        ref = declared_reference(release, fixture, declaration, self.mode(release))
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
        status = template_status(release, fixture, declaration, full.input_ids, self.mode(release))
        if is_contradicted(status):
            return unaligned_verdict(
                self, adapter, release, fixture, declaration, full.input_ids, ref.input_ids, reference=info
            )

        template_full = token_ids_on(release, fixture, template, self.mode(release))
        checked = []
        for j in boundaries(fixture):
            prefix = fixture.prefix(j)
            framework_prefix = adapter.encode(prefix, self.base_cfg)
            if isinstance(framework_prefix, Dropped):
                checked.append({"boundary": j, "ok": False, "dropped": framework_prefix.reason})
                continue
            template_break = break_signature(token_ids_on(release, prefix, template, self.mode(release)), template_full)
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
            prop = prefix_break_property_id(release, template_sha256(template), self.mode(release))
            properties.append(prop)
            evidence["inherited_from_template"] = (
                f"The prefix is NOT preserved at boundaries {inherited}. The framework breaks it exactly as its "
                f"declared template does, so this is attributed to the template: see template property {prop}."
            )
        verified = {"chat_template": status}
        probe = tool_arguments_probe(adapter, release, fixture, declaration, self.base_cfg, self.mode(release))
        if probe is not None:
            verified["tool_arguments"] = probe["status"]
            evidence["tool_arguments_probe"] = probe
        result = "pass" if all(c["ok"] for c in checked) else "fail"
        fields: dict[str, Any] = {"kind": "bug" if result == "fail" else None}
        if probe is not None and is_contradicted(probe["status"]):
            result, fields = "inconclusive", {"reason": adapter_error("tool_arguments", probe["status"])}
        return self.verdict(
            adapter,
            fixture,
            result,
            reference=info,
            conventions=conventions,
            declared=declaration,
            verified=verified,
            template_properties=properties,
            evidence=evidence,
            **fields,
        )

    def plants(self, release: LoadedRelease, fixture: Fixture, declare: Declare) -> list[Plant]:
        if not boundaries(fixture):
            return []
        declaration = declare(self.base_cfg)
        out = []
        has_boolean = any(_has_boolean_tool_argument(m) for m in fixture.messages[:-1] if m["role"] == "assistant")
        if (
            has_boolean
            and declaration.tool_arguments != "parsed_as_json"
            and _renders(release, _rewrite_history(fixture, _arguments_as_python_repr), declaration)
        ):
            # Skipped when the declared template refuses string arguments (Inkling's and Hy4's do), or the framework
            # declares it parses arguments as JSON (Axolotl): then this fault can't arise there, and the planted run
            # would only crash. The token-level variant below still guards the rule for every framework.
            out.append(
                _history_plant(
                    "history_false_as_python",
                    "re-render earlier tool-call arguments as a Python repr (false -> False)",
                    _arguments_as_python_repr,
                    origin="synthetic",
                    source=(
                        "symptom seen by Prime Intellect (May 2026): history rendered with Python False; their "
                        "mechanism (a parser round trip during RL rollouts) differs, so this is not a reproduction"
                    ),
                )
            )
        swaps = _false_token_swaps(release, fixture, declaration, self.mode(release)) if has_boolean else {}
        if swaps:
            out.append(_false_tokens_plant(release, declaration, self.mode(release), swaps))
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


# The tool-argument property and the token-level plant.


def _with_tool_arguments(conv: Conversation, convert: Any) -> Conversation:
    """The conversation with every dict tool-call argument replaced by convert(arguments)."""
    messages = copy.deepcopy(conv.messages)
    for message in messages:
        for call in message.get("tool_calls") or []:
            function = call.get("function", call)
            if isinstance(function.get("arguments"), dict):
                function["arguments"] = convert(function["arguments"])
    return conv.with_messages(messages)


def tool_arguments_probe(
    adapter: Any, release: LoadedRelease, fixture: Fixture, declaration: Declaration, cfg: TrainCfg, mode: str
) -> dict[str, Any] | None:
    """Verify `declaration.tool_arguments` by behavior, or None when the fixture has no dict tool arguments."""
    if declaration.tool_arguments is None:
        return None
    compact = _with_tool_arguments(fixture, lambda args: json.dumps(args, separators=(",", ":"), ensure_ascii=False))
    if compact.messages == fixture.messages:
        return None
    probe: dict[str, Any] = {"input": "every tool-call argument given as a compact JSON string"}
    if not _renders(release, compact, declaration):
        return {**probe, "status": "unverified: the declared template can't render string arguments"}
    verbatim = token_ids_on(release, compact, declaration.chat_template, mode)
    reparsed = token_ids_on(release, fixture, declaration.chat_template, mode)
    if verbatim == reparsed:
        return {**probe, "status": "unverified: the declared template renders both forms the same"}
    out = adapter.encode(compact, cfg)
    if isinstance(out, Dropped):
        return {**probe, "status": MISMATCH + f"the framework dropped the conversation ({out.reason})"}
    ids = tuple(out.input_ids)
    seen = [name for name, expected in (("passed_through", verbatim), ("parsed_as_json", reparsed)) if ids == expected]
    probe["behaves_as"] = seen[0] if seen else None
    if declaration.tool_arguments in seen:
        return {**probe, "status": CONSISTENT}
    if seen:
        return {**probe, "status": CONTRADICTED + f"the framework's tokens are exactly the {seen[0]!r} render"}
    return {**probe, "status": MISMATCH + "the tokens match neither the verbatim nor the re-serialized render"}


def _history_spans(conv: Conversation, ref: TemplateReference) -> list[AssistantSpan]:
    """Spans of assistant turns that aren't the conversation's last message (history, in this encoding)."""
    return [s for s in ref.spans if s.message_index != len(conv.messages) - 1]


def _false_token_swaps(release: LoadedRelease, conv: Conversation, declaration: Declaration, mode: str) -> dict[int, int]:
    """Token ids spelling `false` in this conversation's history turns -> the single token spelling `False`."""
    tokenizer = release.tokenizer
    try:
        ref = declared_reference(release, conv, declaration, mode)
    except TemplateReferenceError:
        return {}
    swaps = {}
    for span in _history_spans(conv, ref):
        for t in ref.input_ids[span.token_start : span.token_end]:
            text = tokenizer.decode([t])
            if "false" in text:
                ids = tokenizer(text.replace("false", "False"), add_special_tokens=False)["input_ids"]
                if len(ids) == 1:
                    swaps[t] = ids[0]
    return swaps


def _false_tokens_plant(release: LoadedRelease, declaration: Declaration, mode: str, swaps: dict[int, int]) -> Plant:
    def swap(out: Encoded | Dropped, conv: Conversation, cfg: TrainCfg) -> Encoded | Dropped:
        if not isinstance(out, Encoded):
            return out
        try:
            ref = declared_reference(release, conv, declaration, mode)
        except TemplateReferenceError:
            return out
        if tuple(out.input_ids) != tuple(ref.input_ids):
            return out  # can't place it; the plant then goes uncaught and the verdict is inconclusive
        ids, labels = list(out.input_ids), list(out.labels)
        for span in _history_spans(conv, ref):
            for i in range(span.token_start, span.token_end):
                if ids[i] in swaps:
                    new = swaps[ids[i]]
                    labels[i] = new if labels[i] == ids[i] else labels[i]
                    ids[i] = new
        return replace(out, input_ids=tuple(ids), labels=tuple(labels))

    name = "history_false_as_python_tokens"
    return Plant(
        name,
        "in the encoded output, swap `false` tokens for `False` in assistant turns that are history",
        lambda adapter: PlantedAdapter(adapter, name, after_encode=swap),
        origin="synthetic",
        source=(
            "token-level variant of history_false_as_python, so frameworks that parse tool arguments as JSON keep "
            "a guard on the rule; same symptom as Prime Intellect's example (May 2026), not its mechanism"
        ),
    )
