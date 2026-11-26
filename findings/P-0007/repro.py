# axolotl==0.21.1 (also main@de8549a), transformers==5.20.1, torch==2.14.1, Python 3.12, CPU. Tokenizer files only.
# type: chat_template passes a message field to the template only if it is mapped or the Jinja analyzer finds it.
# EXAONE 4's template (bundled `exaone4`) reads msg.tool_calls and msg.reasoning_content: both are dropped. Exit 1.
import os, sys
os.environ["AXOLOTL_DO_NOT_TRACK"] = os.environ["DO_NOT_TRACK"] = "1"
from transformers import AutoTokenizer
from axolotl.prompt_strategies.chat_template import load
from axolotl.utils.chat_templates import get_chat_template
from axolotl.utils.dict import DictDefault
from axolotl.utils.schemas.datasets import SFTDataset

tok = AutoTokenizer.from_pretrained("LGAI-EXAONE/EXAONE-4.0-1.2B", revision="3abf2810673c7c0778df64a73c2d52eab32d91c4")
call = {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "Paris"}}}
tools = [{"type": "function", "function": {"name": "get_weather", "description": "Weather for a city.",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}]
messages = [{"role": "user", "content": "Weather in Paris?"},
            {"role": "assistant", "content": "", "tool_calls": [call]},
            {"role": "tool", "tool_call_id": "c1", "name": "get_weather", "content": "18C"},
            {"role": "assistant", "reasoning_content": "The tool says 18C.", "content": "It's 18C."}]
kwargs = {"skip_think": False}  # EXAONE 4 renders reasoning only with skip_think=False
ds = SFTDataset(path="data.jsonl", type="chat_template", chat_template="exaone4")  # validated as `axolotl preprocess` does
strategy = load(tok, DictDefault({"sequence_len": 4096, "chat_template_kwargs": kwargs}),
                DictDefault(ds.model_dump(exclude_none=True)))
row = tok.decode(strategy.tokenize_prompt({"messages": messages, "tools": tools})["input_ids"])
template = get_chat_template("exaone4")
expected = tok.apply_chat_template(messages, tools=tools, chat_template=template, tokenize=False, **kwargs)
print("mappings:", ds.message_property_mappings, "| analyzer found:", sorted(strategy.prompter.chat_template_msg_variables))
for what, mark in (("tool call", '<tool_call>{"name": "get_weather"'), ("reasoning", "The tool says 18C.")):
    print(f"{what}: in the template's render: {mark in expected}; in Axolotl's row: {mark in row}")
sys.exit(0 if row == expected else 1)
