"""P-0007's reach: which of Axolotl 0.21.1's bundled chat templates read a message field that Axolotl drops?

Run from the repo root, in the Axolotl environment:
    envs/axolotl-0.21.1/.venv/bin/python docs/evidence/2026-11-26-p0007-reach/scan_bundled.py

Axolotl passes a message field to the template only if it is mapped (`message_property_mappings`, validated to role
and content) or if its Jinja analyzer lists it (`ChatTemplatePrompter.chat_template_msg_variables`, from
`JinjaTemplateAnalyzer.get_message_vars`; prompt_strategies/chat_template.py `transform_message`). This script
doesn't trust the analyzer to say what a template reads. For each bundled template it renders probe conversations
twice, as given and as Axolotl keeps them (role, content and the analyzer's fields), and reports:
  - row differs: the two renders differ;
  - dropped and read: each dropped field the template reads. A field counts if removing it alone from the
    conversation as given changes the render, or if putting it back alone into the conversation as Axolotl keeps it
    changes that render. The second test catches fields a template reads as alternatives (`name`, else
    `tool_call_id`); the first catches fields hidden by another dropped field that makes the kept view raise.
The probes put reasoning on every assistant turn (templates differ in which turns render it: the last one, or
tool-call turns), under each of the names templates use (reasoning_content, reasoning, thinking), one name per
probe. Each probe is rendered with no template kwargs and with each documented thinking switch these templates read
(enable_thinking=True, skip_think=False); an outcome counts if it holds for any of them. Rendering uses
transformers' apply_chat_template with the template string (a tokenizer is needed only for its special-token
variables; Qwen3's, from the release cache, with a placeholder bos_token, since Qwen3 has none and some templates
need one). A render that raises is recorded with its error. Writes scan_bundled.json.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from parity.release import load_release
from parity.targets import CALIBRATION

HERE = Path(__file__).resolve().parent

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a city.",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        },
    }
]
REASONING_NAMES = ("reasoning_content", "reasoning", "thinking")


def tool_probe(reasoning: str) -> list[dict]:
    """A tool-call conversation: a call with empty content, its result, and a final answer with reasoning."""
    return [
        {"role": "user", "content": "What's the weather in Paris?"},
        {
            "role": "assistant",
            "content": "",
            reasoning: "PROBE_REASONING_BEFORE_CALL",
            "tool_calls": [
                {"id": "abc123XYZ", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "Paris"}}}
            ],
        },
        {"role": "tool", "tool_call_id": "abc123XYZ", "name": "get_weather", "content": "18C and sunny"},
        {"role": "assistant", "content": "It is 18C and sunny in Paris.", reasoning: "PROBE_REASONING_TEXT"},
    ]


def reasoning_probe(reasoning: str) -> list[dict]:
    """A plain two-turn conversation, reasoning on both assistant turns."""
    return [
        {"role": "user", "content": "Is 7 prime?"},
        {"role": "assistant", "content": "Yes.", reasoning: "PROBE_REASONING_HISTORY"},
        {"role": "user", "content": "Is 97 prime?"},
        {"role": "assistant", "content": "Yes, 97 is prime.", reasoning: "PROBE_REASONING_TEXT"},
    ]


KWARGS = ({}, {"enable_thinking": True}, {"skip_think": False})


def render(tok, template: str, messages: list[dict], tools, kwargs: dict) -> str:
    try:
        return tok.apply_chat_template(
            messages, tools=tools, chat_template=template, tokenize=False, bos_token="<BOS>", **kwargs
        )
    except Exception as e:  # a template may refuse a role or a shape; the error is part of the outcome
        return f"<raised {type(e).__name__}: {str(e)[:120]}>"


def without(messages: list[dict], field: str) -> list[dict]:
    return [{k: v for k, v in m.items() if k != field} for m in messages]


def as_axolotl_keeps(messages: list[dict], kept: set[str]) -> list[dict]:
    return [{k: v for k, v in m.items() if k in kept and v is not None} for m in messages]


def put_back(kept_view: list[dict], messages: list[dict], field: str) -> list[dict]:
    out = copy.deepcopy(kept_view)
    for mine, given in zip(out, messages):
        if field in given:
            mine[field] = given[field]
    return out


def scan(tok, template: str, kept: set[str]) -> dict:
    out = {"probes": {}}
    matters: set[str] = set()
    row_differs = False
    for probe_name, build, tools in (("tool_call", tool_probe, TOOLS), ("reasoning", reasoning_probe, None)):
        for reasoning in REASONING_NAMES:
            messages = build(reasoning)
            dropped = sorted({k for m in messages for k in m} - kept)
            probe = {"dropped": dropped, "dropped_and_read": [], "row_differs_with": []}
            view = as_axolotl_keeps(messages, kept)
            for kwargs in KWARGS:
                label = json.dumps(kwargs)
                full = render(tok, template, copy.deepcopy(messages), tools, kwargs)
                kept_render = render(tok, template, view, tools, kwargs)
                if kept_render != full:
                    probe["row_differs_with"].append(label)
                for field in dropped:
                    removed = render(tok, template, without(messages, field), tools, kwargs) != full
                    restored = render(tok, template, put_back(view, messages, field), tools, kwargs) != kept_render
                    if removed or restored:
                        probe["dropped_and_read"] = sorted(set(probe["dropped_and_read"]) | {field})
                probe.setdefault("raised", full.startswith("<raised"))
                if full.startswith("<raised"):
                    probe.setdefault("error", full)
                if kept_render.startswith("<raised") and not full.startswith("<raised"):
                    probe.setdefault("kept_view_raises", kept_render)
            out["probes"][f"{probe_name}/{reasoning}"] = probe
            matters |= set(probe["dropped_and_read"])
            row_differs |= bool(probe["row_differs_with"])
    out["dropped_and_read"] = sorted(matters)
    out["row_differs"] = row_differs
    return out


def main() -> None:
    import axolotl
    from axolotl.prompt_strategies.jinja_template_analyzer import JinjaTemplateAnalyzer
    from axolotl.utils.chat_templates import get_chat_template

    tok = load_release(CALIBRATION).tokenizer
    folder = Path(axolotl.__file__).parent / "utils" / "chat_templates" / "templates"
    result = {"axolotl": axolotl.__version__, "templates": {}}
    for path in sorted(folder.glob("*.jinja")):
        name = path.stem
        template = get_chat_template(name)
        analyzer = sorted(JinjaTemplateAnalyzer(template).get_message_vars("messages"))
        kept = {"role", "content", *analyzer}
        result["templates"][name] = {"analyzer_fields": analyzer, **scan(tok, template, kept)}
    (HERE / "scan_bundled.json").write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    affected = {n: r for n, r in result["templates"].items() if r["dropped_and_read"]}
    print(f"axolotl {result['axolotl']}: {len(result['templates'])} bundled templates, {len(affected)} read a field Axolotl drops")
    for name, r in result["templates"].items():
        flag = "AFFECTED" if r["dropped_and_read"] else "ok"
        raised = sorted({v["error"][:70] for v in r["probes"].values() if v["raised"]})
        kept_raises = sorted({v["kept_view_raises"][:70] for v in r["probes"].values() if "kept_view_raises" in v})
        print(f"  {flag:8} {name}: analyzer {r['analyzer_fields']}; dropped and read {r['dropped_and_read']}; "
              f"row differs {r['row_differs']}; given raises {raised}; kept view raises {kept_raises}")


if __name__ == "__main__":
    main()
