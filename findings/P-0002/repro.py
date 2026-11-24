# axolotl==0.21.1 (also main@de8549a), transformers==5.20.1, torch==2.14.1, Python 3.12, CPU. Tokenizer and config files only.
# With train_on_eos: "turn", the end-of-turn token of an empty assistant turn gets no label. Exit 1 if that happens.
import os, sys
os.environ["AXOLOTL_DO_NOT_TRACK"] = os.environ["DO_NOT_TRACK"] = "1"
from transformers import AutoTokenizer
from axolotl.prompt_strategies.chat_template import load
from axolotl.utils.dict import DictDefault

TOKENIZERS = {  # three chat-template families, pinned; each one's EOS token is its end-of-turn token
    "HuggingFaceTB/SmolLM2-135M-Instruct": "12fd25f77366fa6b3b4b768ec3050bf629380bac",
    "unsloth/Llama-3.2-1B-Instruct": "5a8abab4a5d6f164389b1079fb721cfab8d7126c",
    "mistralai/Mistral-7B-Instruct-v0.3": "c170c708c41dac9275d15a8fff4eca08d52bab71",
}
messages = [
    {"role": "user", "content": "Say hi."},
    {"role": "assistant", "content": "Hi."},
    {"role": "user", "content": "Now reply with an empty message."},
    {"role": "assistant", "content": ""},  # the model should learn to end its turn right away
]
ds_cfg = {"type": "chat_template", "roles_to_train": ["assistant"], "train_on_eos": "turn"}  # the documented defaults
failed = False
for repo, revision in TOKENIZERS.items():
    tok = AutoTokenizer.from_pretrained(repo, revision=revision)
    strategy = load(tok, DictDefault({"train_on_inputs": False, "sequence_len": 2048}), DictDefault(ds_cfg))
    row = strategy.tokenize_prompt({"messages": messages})
    ends = [i for i, t in enumerate(row["input_ids"]) if t == tok.eos_token_id]  # the last one closes the empty turn
    labeled = [row["labels"][i] != -100 for i in ends]
    print(f"{repo}: {tok.eos_token} at {ends}, labeled {labeled}; empty turn's {tok.eos_token} labeled: {labeled[-1]}")
    failed |= not labeled[-1]
sys.exit(1 if failed else 0)
