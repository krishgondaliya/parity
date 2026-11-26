"""Triage of M5 Axolotl x DeepSeek V4.1: which message fields reach the template?

Run from the repo root, as a file (Axolotl's dataset loading starts worker processes that re-import it):
    envs/axolotl-0.21.1/.venv/bin/python docs/evidence/2026-11-26-m5-axolotl-deepseek/triage_fields.py

Axolotl passes a message field to the template only if it is mapped (`message_property_mappings`, validated to
role and content) or if its Jinja analyzer finds it among the template's message variables
(`ChatTemplatePrompter.chat_template_msg_variables`; prompt_strategies/chat_template.py transform_message). It
prints what the analyzer finds in each Phase 0 template and in Qwen3's, then, through the adapter (the full
`axolotl preprocess` path), compares Axolotl's prepared row with the template's own render on the tool_call and
reasoning fixtures, and says whether the tool call and the reasoning text are in the row. This needs no reference
spans, so it works where our reference refuses (DeepSeek's tool_call). Writes triage_fields.json next to this file.
"""

from __future__ import annotations

import json
from pathlib import Path

from parity.adapters.axolotl import AxolotlAdapter  # imports Axolotl first, as the adapter does
from parity.adapters.base import TrainCfg
from parity.fixtures import load_fixture
from parity.release import load_release
from parity.targets import CALIBRATION, PHASE0_TARGETS

HERE = Path(__file__).resolve().parent


def first_assistant(text: str) -> str:
    """The first assistant turn of a DeepSeek or Qwen3 render: from its header to the next user turn."""
    for header, user in (("<｜Assistant｜>", "<｜User｜>"), ("<|im_start|>assistant", "<|im_start|>user")):
        if header in text:
            start = text.index(header)
            end = text.find(user, start)
            return text[start : end if end != -1 else len(text)]
    return ""


def main() -> None:
    from axolotl.prompt_strategies.jinja_template_analyzer import JinjaTemplateAnalyzer

    out: dict = {"message_variables": {}, "rows": {}}
    for pin in (CALIBRATION, *PHASE0_TARGETS):
        release = load_release(pin)
        template = release.tokenizer.chat_template
        template = template if isinstance(template, str) else template["default"]
        out["message_variables"][pin.repo_id] = sorted(JinjaTemplateAnalyzer(template).get_message_vars("messages"))

    for pin in (next(t for t in PHASE0_TARGETS if t.repo_id.startswith("deepseek-ai/")), CALIBRATION):
        release = load_release(pin)
        adapter = AxolotlAdapter(release)
        tok = release.tokenizer
        for fid, marker in (("tool_call", "get_weather"), ("reasoning", "divisible")):
            conv = load_fixture(fid)
            row = tok.decode(list(adapter.encode(conv, TrainCfg()).input_ids))
            render = tok.apply_chat_template(conv.messages, tools=conv.tools, tokenize=False)
            calls_in_render = render.count(marker)
            out["rows"][f"{pin.repo_id} / {fid}"] = {
                "row_equals_render": row == render,
                f"'{marker}' in the template's render": calls_in_render,
                f"'{marker}' in Axolotl's row": row.count(marker),
                "row_tail": row[-160:],
                "render_tail": render[-160:],
                "first_assistant_turn_row": first_assistant(row),
                "first_assistant_turn_render": first_assistant(render),
            }
    (HERE / "triage_fields.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for repo, names in out["message_variables"].items():
        print(f"analyzer, {repo}: {names}")
    for key, r in out["rows"].items():
        counts = {k: v for k, v in r.items() if k.startswith("'")}
        print(f"{key}: row == render: {r['row_equals_render']}; {counts}")


if __name__ == "__main__":
    main()
