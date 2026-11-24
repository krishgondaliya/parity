# axolotl==0.21.1, transformers==5.20.1, torch==2.14.1, Python 3.12, CPU. Downloads Qwen3-0.6B tokenizer and config files only.
# An empty assistant turn's <|im_end|> gets no label, despite train_on_eos: "turn". Exit 1 if that happens.
import os, sys
os.environ["AXOLOTL_DO_NOT_TRACK"] = os.environ["DO_NOT_TRACK"] = "1"
from transformers import AutoTokenizer
from axolotl.prompt_strategies.chat_template import load
from axolotl.utils.dict import DictDefault

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", revision="c1899de289a04d12100db370d81485cdf75e47ca")
ds_cfg = {"type": "chat_template", "roles_to_train": ["assistant"], "train_on_eos": "turn"}  # the documented defaults
strategy = load(tok, DictDefault({"train_on_inputs": False, "sequence_len": 2048}), DictDefault(ds_cfg))
messages = [
    {"role": "user", "content": "Say hi."},
    {"role": "assistant", "content": "Hi."},
    {"role": "user", "content": "Now reply with an empty message."},
    {"role": "assistant", "content": ""},  # the model should learn to end its turn right away
]
row = strategy.tokenize_prompt({"messages": messages})
im_end = tok.convert_tokens_to_ids("<|im_end|>")
ends = [i for i, t in enumerate(row["input_ids"]) if t == im_end]  # user, assistant, user, assistant
got = [row["labels"][i] != -100 for i in ends]
print(tok.decode(row["input_ids"]))
print("<|im_end|> at", ends)
print("trained, expected:", [False, True, False, True])
print("trained, actual:  ", got)
sys.exit(0 if got == [False, True, False, True] else 1)
