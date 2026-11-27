"""P-0008: does the suggested template change make the list form render like the official renderer, and change
nothing else?

Run from the repo root, in TRL's vlm environment (transformers 5.24.0):
    envs/trl-1.18.2/vlm/.venv/bin/python findings/P-0008/fix_check.py

The change, in the template's list branch: skip an empty text part on an assistant turn with tool calls or reasoning,
the same condition the string branch already applies (chat_template.jinja L75). Checked:
  1. on the four forms of the tool-call turn's content ("", null, missing, one empty text part), the patched
     template's text equals tml-renderers' output, for the fixture the renderer comparison used
     (docs/evidence/2026-11-26-p0008-renderer/results.json) and for the repro's own conversation (repro_results.json);
  2. on everything else tried, the patched template renders exactly what the original does: the 8 Phase 0 fixtures as
     given, each fixture with every string content turned into one text part (TRL's vision path), a non-empty text
     part on a tool-call turn, and an empty text part on a user turn and on an assistant turn without tool calls or
     reasoning;
  3. the one render the patch changes is TRL's vision-path row for the `tool_call` fixture (its tool-call turn's ""
     as one empty text part), and patched it equals the original template's render of the fixture as given.
Writes fix_check.json next to this file.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

from parity.fixtures import FIXTURE_IDS, load_fixture
from parity.release import load_release
from parity.targets import PHASE0_TARGETS

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
INKLING = next(t for t in PHASE0_TARGETS if t.repo_id == "thinkingmachines/Inkling-2-Mini")

OLD = """                {%- set text_part = (part.text if part.text is defined and part.text is string else "") -%}
                {{- rtok ~ "<|content_text|>" ~ text_part ~ "<|end_message|>" -}}"""
NEW = """                {%- set text_part = (part.text if part.text is defined and part.text is string else "") -%}
                {%- if text_part or not (message.role == "assistant" and (message.tool_calls or message.reasoning_content)) -%}
                    {{- rtok ~ "<|content_text|>" ~ text_part ~ "<|end_message|>" -}}
                {%- endif -%}"""


def as_parts(messages: list[dict]) -> list[dict]:
    """Every string content as one text part, as TRL's prepare_multimodal_messages does."""
    out = copy.deepcopy(messages)
    for m in out:
        if isinstance(m.get("content"), str):
            m["content"] = [{"type": "text", "text": m["content"]}]
    return out


def main() -> None:
    sys.path.insert(0, str(REPO / "docs/evidence/2026-11-26-p0008-renderer"))
    import compare  # the renderer comparison's own cases

    tok = load_release(INKLING).tokenizer
    original = tok.chat_template if isinstance(tok.chat_template, str) else tok.chat_template["default"]
    assert original.count(OLD) == 1
    patched = original.replace(OLD, NEW)

    def render(template, messages, tools):
        return tok.apply_chat_template(messages, tools=tools, chat_template=template, tokenize=False)

    renderer = json.loads((REPO / "docs/evidence/2026-11-26-p0008-renderer/results.json").read_text(encoding="utf-8"))
    out: dict = {"equals_renderer": {}, "equals_renderer_on_the_repro_conversation": {}, "unchanged": {}}
    for case in compare.cases():
        expected = renderer["cases"][case["name"]]["renderer"]["text"]
        out["equals_renderer"][case["name"]] = {
            "original": render(original, case["messages"], case["tools"]) == expected,
            "patched": render(patched, case["messages"], case["tools"]) == expected,
        }
    import repro_cases  # the repro's own conversation, and the renderer's output for it

    ns = repro_cases.repro_namespace()
    on_repro = json.loads((REPO / "docs/evidence/2026-11-26-p0008-renderer/repro_results.json").read_text(encoding="utf-8"))
    for name, content in {'""': "", "null": None, "missing": ..., '[{"type": "text", "text": ""}]': [{"type": "text", "text": ""}]}.items():
        messages = ns["conversation"]("")
        if content is ...:
            del messages[1]["content"]
        else:
            messages[1]["content"] = content
        expected = on_repro["cases"][name]["renderer_text"]
        out["equals_renderer_on_the_repro_conversation"][name] = {
            "original": render(original, messages, None) == expected,
            "patched": render(patched, messages, None) == expected,
        }
    others = {}
    for fid in FIXTURE_IDS:
        conv = load_fixture(fid)
        others[f"{fid} (as given)"] = (list(conv.messages), conv.tools)
        others[f"{fid} (string content as one text part)"] = (as_parts(list(conv.messages)), conv.tools)
    call = {"id": "c1", "type": "function", "function": {"name": "f", "arguments": {}}}
    others["non-empty text part on a tool-call turn"] = (
        [{"role": "user", "content": "q"}, {"role": "assistant", "content": [{"type": "text", "text": "Let me check."}], "tool_calls": [call]}], None)
    others["empty text part on a user turn"] = ([{"role": "user", "content": [{"type": "text", "text": ""}]}, {"role": "assistant", "content": "a"}], None)
    others["empty text part on a plain assistant turn"] = ([{"role": "user", "content": "q"}, {"role": "assistant", "content": [{"type": "text", "text": ""}]}], None)
    for name, (messages, tools) in others.items():
        a, b = render(original, messages, tools), render(patched, messages, tools)
        out["unchanged"][name] = a == b
    conv = load_fixture("tool_call")
    out["trl_row_for_tool_call_patched_equals_render_as_given"] = (
        render(patched, as_parts(list(conv.messages)), conv.tools) == render(original, list(conv.messages), conv.tools)
    )
    (HERE / "fix_check.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("equals_renderer", "equals_renderer_on_the_repro_conversation")}, indent=1))
    changed = [k for k, v in out["unchanged"].items() if not v]
    print(f"unchanged by the patch: {sum(out['unchanged'].values())} of {len(out['unchanged'])}; changed: {changed}")
    print("TRL's row for tool_call, patched, equals the template's render as given:",
          out["trl_row_for_tool_call_patched_equals_render_as_given"])


if __name__ == "__main__":
    main()
