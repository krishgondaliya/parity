# P-0002 upstream issue draft. NOT POSTED.

Target: axolotl-ai-cloud/axolotl. Blank issues are disabled there (`.github/ISSUE_TEMPLATE/config.yml`), so this is laid out field by field for the **Bug report** form (`bug-report.yaml`). Before posting, run `findings/P-0002/repro.py` yourself (it should exit 1), then paste each section below into the form field it names. Don't paste this header. Attach `findings/P-0002/requirements.txt`.

---

**Title**

[chat_template] End Token of an Empty Assistant Turn Is Not Labeled with `train_on_eos: turn` for Qwen3-0.6B (axolotl 0.21.1)

---

**Please check that this issue hasn't been reported before.** Tick it. Searched issues and PRs: "empty assistant turn", "empty content eos", "train_on_eos empty", "empty turn label", "turn_end_idx", "find_first_eos_token", "Content end boundary". Nearest: #3754 (closed, a different `eot_tokens` cause) and PR #4132 (doesn't touch the empty-turn path).

---

**Expected Behavior**

With `type: chat_template`, `roles_to_train: ["assistant"]` and `train_on_eos: turn` (the defaults), every assistant turn's end token should be labeled. The `train_on_eos` docs say `turn` trains "the EOS token at the end of each trainable turn" ([`utils/schemas/datasets.py` L184](https://github.com/axolotl-ai-cloud/axolotl/blob/v0.21.1/src/axolotl/utils/schemas/datasets.py#L184)), and an assistant turn with `"content": ""` is trainable, so I expected its `<|im_end|>` to be labeled.

---

**Current behaviour**

An assistant turn whose `content` is `""` gets no label on its end token (`<|im_end|>` with Qwen3's own template). If the empty turn is the last one, no token of that turn gets a label. No warning is logged, and the row is kept whenever another turn is trainable.

Output of the script below:

```
<|im_start|>user
Say hi.<|im_end|>
<|im_start|>assistant
Hi.<|im_end|>
<|im_start|>user
Now reply with an empty message.<|im_end|>
<|im_start|>assistant
<think>

</think>

<|im_end|>

<|im_end|> at [6, 13, 25, 34]
trained, expected: [False, True, False, True]
trained, actual:   [False, True, False, False]
```

Position 34, the empty turn's `<|im_end|>`, has label -100.

Scope (checked):
- Empty turn in the middle: with `[user "Say nothing.", assistant "", user "Now say hi.", assistant "Hi."]`, `<|im_end|>` is at `[6, 11, 20, 31]` and trained is `[False, False, False, True]`.
- Also affected with Qwen3: a middle turn with `reasoning_content` and `content: ""`. Qwen3's template drops reasoning from earlier turns, so that turn renders empty.
- Not affected: an assistant turn with a tool call and `content: ""` or `content: None`, in the middle or at the end. The tool call and `<|im_end|>` are labeled.
- Not affected: a conversation whose only assistant turn is empty. It has no trainable tokens, so it is dropped with the existing "Dropped ... samples with no trainable tokens" warning.
- The `<think>\n\n</think>\n\n` in the last turn is template text outside the content field. It is unlabeled for a non-empty last turn too (with a last turn of `"Hi."`, only `Hi.<|im_end|>` is labeled), so this issue is only about `<|im_end|>`.

Reference: the row's `input_ids` equal `tokenizer.apply_chat_template(messages)` with Qwen3-0.6B's own chat template.

---

**Steps to reproduce**

Run this script (CPU, a few seconds, downloads Qwen3-0.6B tokenizer and config files only). It exits 1 when the empty turn's `<|im_end|>` is unlabeled.

```python
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
```

---

**Config yaml**

Not needed: the script calls the `chat_template` strategy directly with its defaults. `axolotl preprocess` with `type: chat_template`, `roles_to_train: ["assistant"]` and `train_on_eos: turn` gives the same labels.

---

**Possible solution**

Where it seems to come from (v0.21.1; main at `de8549a` has the same code, lines shifted by 4):
- For an empty turn, `_find_turn_from_text` returns `(-1, -1)` ([L966-967](https://github.com/axolotl-ai-cloud/axolotl/blob/v0.21.1/src/axolotl/prompt_strategies/chat_template.py#L966-L967)), and `find_turn` passes it on ([L1146-1147](https://github.com/axolotl-ai-cloud/axolotl/blob/v0.21.1/src/axolotl/prompt_strategies/chat_template.py#L1146-L1147)).
- The EOT and EOS searches then start from `turn_end_idx = -1` ([L800](https://github.com/axolotl-ai-cloud/axolotl/blob/v0.21.1/src/axolotl/prompt_strategies/chat_template.py#L800)). Both look for `<|im_end|>` here, since `eot_tokens` defaults to the EOS token. They check `input_ids[-1]` (the trailing `\n`), then find the first `<|im_end|>` at position 6, and `abs(6 - (-1)) = 7 > 3` rejects it ([L803](https://github.com/axolotl-ai-cloud/axolotl/blob/v0.21.1/src/axolotl/prompt_strategies/chat_template.py#L803)). So the turn's own end token is never labeled.

I may be missing a reason this is intended. If empty turns are meant to get no loss, a note in the docs or a warning would help. The code path isn't specific to Qwen3, but Qwen3-0.6B is the only model I tested.

---

**Which Operating Systems are you using?** Linux (WSL2). CPU only.

**Python Version:** 3.12.13

**axolotl branch-commit:** v0.21.1 (pip); also reproduces on main@de8549a4406327ab5311f7187a8afaa232645487

Environment details, to add under Current behaviour or in a comment:
- axolotl 0.21.1, transformers 5.20.1, tokenizers 0.23.3, torch 2.14.1 (CUDA wheel, run on CPU). Reproduces in a fresh virtualenv.
- main at `de8549a` (0.21.2.dev0): transformers 5.22.0, torch 2.15.0. Same output.
- Pinned package list attached (`requirements.txt`).

---

**Acknowledgements**
- Title casing: done.
- Searched existing issues: done (see above).
- "I am using the latest version of axolotl": 0.21.1 is the latest release, and main also reproduces.
- Enough information to reproduce: the script above.
