# P-0002 upstream issue draft. NOT POSTED.

Target: axolotl-ai-cloud/axolotl. Blank issues are disabled there, so this is laid out field by field for the **Bug report** form. First run the repro as `findings/P-0002/HOW-TO-RUN.md` says (it should exit 1). Then paste each section below into the form field it names, without this header, and attach `findings/P-0002/requirements.txt`.

---

**Title**

End-of-turn token of an empty assistant turn is never labeled with train_on_eos: turn

---

**Please check that this issue hasn't been reported before.** Tick it. Searched issues and PRs: "empty assistant turn", "empty content eos", "train_on_eos empty", "empty turn label", "turn_end_idx", "find_first_eos_token", "Content end boundary". Nearest: #3754 (closed, a different `eot_tokens` cause) and PR #4132 (doesn't touch the empty-turn path).

---

**Expected Behavior**

With `type: chat_template`, `roles_to_train: ["assistant"]` and `train_on_eos: turn` (the defaults), every assistant turn's end-of-turn token should be labeled. The `train_on_eos` docs say `turn` trains "the EOS token at the end of each trainable turn" ([`utils/schemas/datasets.py` L184](https://github.com/axolotl-ai-cloud/axolotl/blob/de8549a4406327ab5311f7187a8afaa232645487/src/axolotl/utils/schemas/datasets.py#L184)), and an assistant turn with `"content": ""` is trainable, so I expected its end-of-turn token to be labeled.

---

**Current behaviour**

An assistant turn whose `content` is `""` gets no label on its end-of-turn token, wherever it is in the conversation. If it is the last turn, no token of that turn is labeled. No warning is logged, and the row is kept whenever another turn is trainable.

It doesn't depend on the chat template. Output of the script below, on three template families:

```
HuggingFaceTB/SmolLM2-135M-Instruct: <|im_end|> at [19, 27, 35, 47, 53], labeled [False, False, True, False, False]; empty turn's <|im_end|> labeled: False
unsloth/Llama-3.2-1B-Instruct: <|eot_id|> at [25, 33, 40, 52, 57], labeled [False, False, True, False, False]; empty turn's <|eot_id|> labeled: False
mistralai/Mistral-7B-Instruct-v0.3: </s> at [8, 19], labeled [True, False]; empty turn's </s> labeled: False
```

In each row, the last end-of-turn token closes the empty assistant turn, and the row's `input_ids` equal `tokenizer.apply_chat_template(messages)`.

Scope (checked):
- Same result for an empty turn in the middle of a conversation, and with Gemma 3 (`unsloth/gemma-3-1b-it`) and Qwen3 (`Qwen/Qwen3-0.6B`).
- Not affected (checked with Qwen3): an assistant turn with a tool call and `content: ""` or `content: None`, in the middle or at the end. The tool call and the end-of-turn token are labeled.
- Not affected: a conversation whose only assistant turn is empty. It has no trainable tokens, so it is dropped with the existing "Dropped ... samples with no trainable tokens" warning.
- Also affected with Qwen3: a middle turn with `reasoning_content` and `content: ""`. Qwen3's template drops reasoning from earlier turns, so that turn renders empty.

---

**Steps to reproduce**

Run this script (CPU, a few seconds, downloads tokenizer and config files only; all three repos are ungated). It exits 1 when an empty turn's end-of-turn token is unlabeled.

```python
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
```

---

**Config yaml**

Not needed: the script calls the `chat_template` strategy directly with its defaults. `axolotl preprocess` with `type: chat_template`, `roles_to_train: ["assistant"]` and `train_on_eos: turn` gives the same labels (checked with Qwen3-0.6B).

---

**Possible solution**

Where it seems to come from (permalinks at main, `de8549a`):
- For an empty turn, `_find_turn_from_text` returns `(-1, -1)` ([L970-L971](https://github.com/axolotl-ai-cloud/axolotl/blob/de8549a4406327ab5311f7187a8afaa232645487/src/axolotl/prompt_strategies/chat_template.py#L970-L971)), and `find_turn` passes it on ([L1150-L1151](https://github.com/axolotl-ai-cloud/axolotl/blob/de8549a4406327ab5311f7187a8afaa232645487/src/axolotl/prompt_strategies/chat_template.py#L1150-L1151)). So content labeling is skipped ([L742](https://github.com/axolotl-ai-cloud/axolotl/blob/de8549a4406327ab5311f7187a8afaa232645487/src/axolotl/prompt_strategies/chat_template.py#L742)), which is fine for an empty span.
- The EOT and EOS searches then start from `turn_end_idx = -1` ([L800-L804](https://github.com/axolotl-ai-cloud/axolotl/blob/de8549a4406327ab5311f7187a8afaa232645487/src/axolotl/prompt_strategies/chat_template.py#L800-L804)). `find_first_eos_token` and `find_first_eot_token` begin at `input_ids[-1]` ([L847-L852](https://github.com/axolotl-ai-cloud/axolotl/blob/de8549a4406327ab5311f7187a8afaa232645487/src/axolotl/prompt_strategies/chat_template.py#L847-L852)):
  - When the row ends with the end-of-turn token (Llama 3.2, Mistral), the search returns `-1`, the index of that last token, and the check `token_idx != -1` treats it as "not found" ([L806-L808](https://github.com/axolotl-ai-cloud/axolotl/blob/de8549a4406327ab5311f7187a8afaa232645487/src/axolotl/prompt_strategies/chat_template.py#L806-L808)).
  - When the row ends with a newline (ChatML, Gemma 3, Qwen3), the search returns the first end-of-turn token in the row, which `abs(token_idx - turn_end_idx) <= 3` rejects.
  - Either way, the empty turn's own end-of-turn token is never labeled.

I may be missing a reason this is intended. If empty turns are meant to get no loss, a note in the docs or a warning would help.

---

**Which Operating Systems are you using?** Linux (WSL2). CPU only.

**Python Version:** 3.12.13

**axolotl branch-commit:** v0.21.1 (pip); also reproduces on main@de8549a4406327ab5311f7187a8afaa232645487

Environment details, to add under Current behaviour:
- axolotl 0.21.1, transformers 5.20.1, tokenizers 0.23.3, torch 2.14.1 (CUDA wheel, run on CPU). Reproduces in a fresh virtualenv.
- main at `de8549a` (0.21.2.dev0): transformers 5.22.0, torch 2.15.0. Same output.
- Pinned package list attached (`requirements.txt`).

---

**Acknowledgements**
- Title: concise and descriptive (sentence case).
- Searched existing issues: done (see above).
- "I am using the latest version of axolotl": 0.21.1 is the latest release, and main also reproduces.
- Enough information to reproduce: the script above.

---

Happy to open a PR with a fix and a test for this if that would help.
