# axolotl==0.21.1, transformers==5.20.1, Python 3.12, CPU. Tokenizer and config files only. No Parity code.
# P-0002's reach (2026-11-25): on ordinary data shapes, is each assistant turn's <|im_end|> labeled? The last case is
# P-0002's own (a reasoning-only history turn renders empty) and is the control: it must print False for that turn.
import os
os.environ["AXOLOTL_DO_NOT_TRACK"] = os.environ["DO_NOT_TRACK"] = os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
from transformers import AutoTokenizer
from axolotl.prompt_strategies.chat_template import load
from axolotl.utils.dict import DictDefault

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", revision="c1899de289a04d12100db370d81485cdf75e47ca")
TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Weather for a city.",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
CALL = [{"id": "call_0", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "Paris"}}}]
R = "The user wants the weather, so I call the tool."
U1, U2 = {"role": "user", "content": "Weather in Paris?"}, {"role": "user", "content": "Thanks. And tomorrow?"}
TOOL = {"role": "tool", "tool_call_id": "call_0", "name": "get_weather", "content": "{\"temp\": 18}"}
ANS = {"role": "assistant", "content": "It's 18 C.", "reasoning_content": "The tool says 18."}
LAST = {"role": "assistant", "content": "I can only see today.", "reasoning_content": "No forecast tool."}
CASES = {
    "reasoning + tool call, content '', agentic (after last user)": [U1, {"role": "assistant", "content": "", "reasoning_content": R, "tool_calls": CALL}, TOOL, ANS],
    "reasoning + tool call, content '', history (before last user)": [U1, {"role": "assistant", "content": "", "reasoning_content": R, "tool_calls": CALL}, TOOL, ANS, U2, LAST],
    "reasoning + tool call, content None, history": [U1, {"role": "assistant", "content": None, "reasoning_content": R, "tool_calls": CALL}, TOOL, ANS, U2, LAST],
    "reasoning + answer, multi-turn (ordinary reasoning trace)": [U1, ANS, U2, LAST],
    "reasoning only, content '', history (P-0002 table row)": [U1, {"role": "assistant", "content": "", "reasoning_content": R}, U2, LAST],
}
strategy = load(tok, DictDefault({"train_on_inputs": False, "sequence_len": 4096}),
                DictDefault({"type": "chat_template", "roles_to_train": ["assistant"], "train_on_eos": "turn"}))
start, end = tok.convert_tokens_to_ids("<|im_start|>"), tok.convert_tokens_to_ids("<|im_end|>")
for name, messages in CASES.items():
    row = strategy.tokenize_prompt({"messages": messages, "tools": TOOLS})
    ids, labels = row["input_ids"], row["labels"]
    assert tok.decode(ids) == tok.apply_chat_template(messages, tools=TOOLS, tokenize=False), name
    out = []
    for i, t in enumerate(ids):
        if t == end:
            s = max(j for j in range(i) if ids[j] == start)
            if tok.decode(ids[s + 1:s + 2]) == "assistant":
                out.append((i, labels[i] != -100))
    print(f"{name}: assistant <|im_end|> (pos, labeled) {out}")
