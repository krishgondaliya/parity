"""Triage of M5 TRL x Inkling-2-Mini, t1.template_parity on tool_call: where does the extra empty text block appear?

Run from the repo root in TRL's vlm variant (Inkling's processor needs torchvision):
    envs/trl-1.18.2/vlm/.venv/bin/python docs/evidence/2026-11-26-m5-trl-inkling/triage_tool_call.py

For each conversation shape below, TRL's own SFTTrainer (assistant_only_loss=True, as the adapter builds it) prepares
the row two ways, and the row is compared with Inkling's own render of the same messages (the tokenizer's
apply_chat_template, string content as given):
- vision path: the release's Inkling2LoopForConditionalGeneration, so TRL loads Inkling2Processor (what a model id
  gives; the M5 run);
- text path: Inkling2LoopForCausalLM, so TRL loads the tokenizer (what a text model gives).
Nothing is re-implemented: the rows are trainer.train_dataset, as TRL built them.
Writes triage_tool_call.json next to this file.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import datasets
from transformers import AutoProcessor, AutoTokenizer
from trl import SFTConfig, SFTTrainer
from trl.trainer.utils import model_accepts_images

from parity.release import fetch_release_files
from parity.targets import PHASE0_TARGETS
from parity.tiny import build

HERE = Path(__file__).resolve().parent
INKLING = next(t for t in PHASE0_TARGETS if t.repo_id == "thinkingmachines/Inkling-2-Mini")
TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Weather for a city.",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]  # fmt: skip
CALL = [{"id": "call_0", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "Paris"}}}]
U = {"role": "user", "content": "Weather in Paris?"}
TOOL = {"role": "tool", "tool_call_id": "call_0", "name": "get_weather", "content": "{\"temp\": 18}"}
ANSWER = {"role": "assistant", "content": "It's 18 C."}
SHAPES = {
    "tool call, content '' (the fixture)": [U, {"role": "assistant", "content": "", "tool_calls": CALL}, TOOL, ANSWER],
    "tool call, content None": [U, {"role": "assistant", "content": None, "tool_calls": CALL}, TOOL, ANSWER],
    "tool call, content text": [U, {"role": "assistant", "content": "Let me check.", "tool_calls": CALL}, TOOL, ANSWER],
    "tool call, content '', with reasoning": [
        U, {"role": "assistant", "content": "", "reasoning_content": "I need the tool.", "tool_calls": CALL}, TOOL, ANSWER],
    "two tool-call turns, content ''": [
        U, {"role": "assistant", "content": "", "tool_calls": CALL}, TOOL,
        {"role": "assistant", "content": "", "tool_calls": [{**CALL[0], "id": "call_1"}]},
        {**TOOL, "tool_call_id": "call_1"}, ANSWER],
    "reasoning only, content '' (no tool call)": [
        U, {"role": "assistant", "content": "", "reasoning_content": "Thinking."}, {"role": "user", "content": "Ok?"}, ANSWER],
    "empty reply, content '' (no tool call, no reasoning)": [
        U, {"role": "assistant", "content": ""}, {"role": "user", "content": "Ok?"}, ANSWER],
}  # fmt: skip


def rows(local: Path, path: str, conversations: list[list[dict]]) -> tuple[str, list[list[int]]]:
    model = build(local, path).model
    processing = (AutoProcessor if model_accepts_images(model) else AutoTokenizer).from_pretrained(local)
    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(output_dir=tempfile.mkdtemp(), report_to="none", use_cpu=True, bf16=False,
                       assistant_only_loss=True, max_length=None),  # fmt: skip
        train_dataset=datasets.Dataset.from_list([{"messages": c, "tools": TOOLS} for c in conversations]),
        processing_class=processing,
    )
    return type(processing).__name__, [list(trainer.train_dataset[i]["input_ids"]) for i in range(len(conversations))]


def main() -> None:
    local = fetch_release_files(INKLING)
    tokenizer = AutoTokenizer.from_pretrained(local)
    names, conversations = list(SHAPES), list(SHAPES.values())
    reference = [tokenizer.apply_chat_template(c, tools=TOOLS, tokenize=True, return_dict=True)["input_ids"]
                 for c in conversations]  # fmt: skip
    out = {"release": INKLING.spec, "shapes": {}}
    for path in ("vision", "text"):
        processing, got = rows(local, path, conversations)
        for name, ref, row in zip(names, reference, got):
            extra = len(row) - len(ref)
            first = next((i for i, (a, b) in enumerate(zip(ref, row)) if a != b), None if extra == 0 else min(len(ref), len(row)))
            entry = out["shapes"].setdefault(name, {})
            entry[path] = {
                "processing_class": processing,
                "equal_to_the_template": row == ref,
                "extra_tokens": extra,
                "at": first,
                "inserted": tokenizer.convert_ids_to_tokens(row[first : first + extra]) if extra > 0 and first is not None else [],
            }
    (HERE / "triage_tool_call.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for name, entry in out["shapes"].items():
        v, t = entry["vision"], entry["text"]
        vision = "same" if v["equal_to_the_template"] else f"+{v['extra_tokens']} {v['inserted']}"
        text = "same" if t["equal_to_the_template"] else f"differs ({t['extra_tokens']:+d} tokens)"
        print(f"{name}: vision {vision} | text {text}")


if __name__ == "__main__":
    main()
