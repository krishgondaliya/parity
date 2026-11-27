# P-0007 upstream issue draft. NOT POSTED.

Target: axolotl-ai-cloud/axolotl. Blank issues are disabled there, so this is laid out field by field for the **Bug report** form. First run the repro as `findings/P-0007/HOW-TO-RUN.md` says (it should exit 1 and print the two texts below). Then paste each section into the form field it names, without this header, and attach `findings/P-0007/requirements.txt`.

---

**Title**

[chat_template] reasoning_content and tool_calls dropped before rendering for EXAONE 4 and Gemma 4 templates

---

**Please check that this issue hasn't been reported before.** Tick it.

---

**Expected Behavior**

With `type: chat_template` and no `message_property_mappings`, I expected every message field the chat template reads (for example `reasoning_content` and `tool_calls`) to reach the template. The schema says: "If a property exists in the template but not in this mapping, the system will attempt to load it directly from the message using the property name as the key" ([`utils/schemas/datasets.py` L152](https://github.com/axolotl-ai-cloud/axolotl/blob/de8549a4406327ab5311f7187a8afaa232645487/src/axolotl/utils/schemas/datasets.py#L152)), and the tool-use section of the conversation docs passes `tool_calls` with no mapping ([`docs/dataset-formats/conversation.qmd` L172-L247](https://github.com/axolotl-ai-cloud/axolotl/blob/de8549a4406327ab5311f7187a8afaa232645487/docs/dataset-formats/conversation.qmd#L172-L247)).

---

**Current behaviour**

For some templates, Axolotl drops fields the template reads before rendering, with no warning. With EXAONE-4.0 and the bundled `exaone4` template, the tool call and the reasoning are in the template's own render but not in the tokenized row. Output of the script below (from the first user turn on; the system prompt with the tool list is the same in both):

```
mappings: {'role': 'role', 'content': 'content'} | analyzer found: ['content', 'role']
--- the template's render ---
[|user|]
Weather in Paris?[|endofturn|]
[|assistant|]
<think>

</think>

<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>[|endofturn|]
[|tool|]
<tool_result>{"result": "18C"}</tool_result>[|endofturn|]
[|assistant|]
<think>
The tool says 18C.
</think>

It's 18C.[|endofturn|]

--- Axolotl's row ---
[|user|]
Weather in Paris?[|endofturn|]
[|assistant|]
[|endofturn|]
[|tool|]
<tool_result>{"result": "18C"}</tool_result>[|endofturn|]
[|assistant|]
<think>

</think>

It's 18C.[|endofturn|]
```

Checked through `axolotl preprocess` (`load_cfg` + `load_datasets`), on plain OpenAI-format jsonl, tokenizer and config files only:
- `LGAI-EXAONE/EXAONE-4.0-1.2B`, with `chat_template: exaone4` or `tokenizer_default` (`skip_think: false`): the reasoning and the tool call are missing from the rows.
- `google/gemma-4-12B-it` with `chat_template: tokenizer_default`: the reasoning and the tool call are missing from the rows.
- `google/gemma-4-31B` with `chat_template: gemma4`, and `google/gemma-4-12B-it` with `gemma4_unified` (as in the v0.21.1 examples): a conversation with an OpenAI-format tool result fails in preprocessing with `TypeError: can only concatenate str (not "NoneType") to str`. The template takes the tool's name from the tool message's `name`, or by matching its `tool_call_id` ([`gemma4.jinja` L280-L283](https://github.com/axolotl-ai-cloud/axolotl/blob/de8549a4406327ab5311f7187a8afaa232645487/src/axolotl/utils/chat_templates/templates/gemma4.jinja#L280-L283)), and both are dropped; with either one kept, it renders. With the result in `tool_responses` on the assistant message instead, the reasoning is missing from the row.

In a template-level scan of the 36 bundled templates, only these three (`exaone4`, `gemma4`, `gemma4_unified`) render differently without the fields Axolotl drops. Ten others raise on my tool-call conversation either way, so for them only the reasoning conversation was compared.

Listing the fields explicitly fixes every case above, for example `message_property_mappings: {role: role, content: content, reasoning_content: reasoning_content, tool_calls: tool_calls}` (plus `tool_call_id: tool_call_id, name: name` for the Gemma 4 templates). With that, the rows equal the template's render.

Environment:
- axolotl 0.21.1, transformers 5.20.1, tokenizers 0.23.3, torch 2.14.1 (CUDA wheel, run on CPU), Linux (WSL2). Reproduces in a fresh virtualenv.
- main at `de8549a` (0.21.2.dev0): transformers 5.22.0, torch 2.15.0. Same output.
- Pinned package list attached (`requirements.txt`).

---

**Steps to reproduce**

`pip install -r requirements.txt` (attached: axolotl 0.21.1, transformers 5.20.1, torch 2.14.1), then `python repro.py`. It runs on CPU in a few seconds, downloads tokenizer files only (the repo is ungated), and exits 1 when the row differs from the template's render.

```python
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
start = expected.index("[|user|]")  # from the first user turn on; the system prompt (the tool list) is the same in both
print("--- the template's render ---\n" + expected[start:] + "\n--- Axolotl's row ---\n" + row[row.index("[|user|]"):])
sys.exit(0 if row == expected else 1)
```

---

**Config yaml** (paste without the fences; the field renders as yaml)

```yaml
# axolotl preprocess with this config, on the script's conversation, gives the same input_ids and labels as the script.
base_model: LGAI-EXAONE/EXAONE-4.0-1.2B
chat_template: exaone4
chat_template_kwargs:
  skip_think: false
datasets:
  - path: data.jsonl
    type: chat_template
sequence_len: 4096
val_set_size: 0
dataset_prepared_path: prepared
output_dir: out
micro_batch_size: 1
gradient_accumulation_steps: 1
num_epochs: 1
learning_rate: 1e-5
```

---

**Possible solution**

Where it seems to come from (permalinks at main, `de8549a`):
- Config validation fills `message_property_mappings` with `{role: role, content: content}` when it isn't set ([`utils/schemas/utils.py` L48-L92](https://github.com/axolotl-ai-cloud/axolotl/blob/de8549a4406327ab5311f7187a8afaa232645487/src/axolotl/utils/schemas/utils.py#L48-L92)). So the prompter's fallback that would add `reasoning_content` for empty mappings ([`prompt_strategies/chat_template.py` L84-L90](https://github.com/axolotl-ai-cloud/axolotl/blob/de8549a4406327ab5311f7187a8afaa232645487/src/axolotl/prompt_strategies/chat_template.py#L84-L90)) isn't reached with a validated config.
- `transform_message` then keeps an unmapped field only if it is in `chat_template_msg_variables` ([L1398-L1404](https://github.com/axolotl-ai-cloud/axolotl/blob/de8549a4406327ab5311f7187a8afaa232645487/src/axolotl/prompt_strategies/chat_template.py#L1398-L1404)), which comes from `JinjaTemplateAnalyzer.get_message_vars("messages")` ([L345-L349](https://github.com/axolotl-ai-cloud/axolotl/blob/de8549a4406327ab5311f7187a8afaa232645487/src/axolotl/prompt_strategies/chat_template.py#L345-L349)).
- What `get_message_vars("messages")` returns on four one-line templates (0.21.1):
  ```
  {% for message in messages %}{{ message.x }}{% endfor %}                                 -> ['x']
  {% for message in messages %}{{ message.get("x") }}{% endfor %}                          -> ['get']
  {% for msg in messages %}{{ msg.x }}{% endfor %}                                         -> []
  {% for i in range(messages|length) %}{% set msg = messages[i] %}{{ msg.x }}{% endfor %}  -> []
  ```
  - `exaone4` reads its fields through `{%- set msg = messages[i] %}` ([L39](https://github.com/axolotl-ai-cloud/axolotl/blob/de8549a4406327ab5311f7187a8afaa232645487/src/axolotl/utils/chat_templates/templates/exaone4.jinja#L39)), so none of those is found. The `content` and `role` it lists come from an earlier `for message in messages` loop ([L32-L33](https://github.com/axolotl-ai-cloud/axolotl/blob/de8549a4406327ab5311f7187a8afaa232645487/src/axolotl/utils/chat_templates/templates/exaone4.jinja#L32-L33)).
  - `gemma4` reads `message.get('reasoning_content')` ([L238](https://github.com/axolotl-ai-cloud/axolotl/blob/de8549a4406327ab5311f7187a8afaa232645487/src/axolotl/utils/chat_templates/templates/gemma4.jinja#L238)), which is recorded as `get`.
- The only trace is a debug message ([L399-L401](https://github.com/axolotl-ai-cloud/axolotl/blob/de8549a4406327ab5311f7187a8afaa232645487/src/axolotl/prompt_strategies/chat_template.py#L399-L401)), and the module's logger is set to INFO ([L30](https://github.com/axolotl-ai-cloud/axolotl/blob/de8549a4406327ab5311f7187a8afaa232645487/src/axolotl/prompt_strategies/chat_template.py#L30)).

Some directions, if any fits the design:
- pass every unmapped message key through to the template. For these conversations that is the same as the explicit mapping above, which gives row = render;
- add the common OpenAI message keys (`tool_calls`, `tool_call_id`, `name`, `reasoning_content`, `reasoning`) to the fields `transform_message` keeps;
- or at least warn when a message has keys that are neither mapped nor found by the analyzer.

I may be missing a reason the current filtering is intended; if so, a note in the docs would help.

Related prior art on the DPO path: #3824, and its open PR #3825, which adds the same OpenAI message keys to the analyzer's fields in the DPO and preference strategies, with a comment that the analysis misses `message.get(...)` reads (e.g. `gemma4`). It doesn't change the SFT `chat_template` strategy above. PR #3228 adds a `tool` key to the empty-mapping default, which validated configs don't reach.

---

**Which Operating Systems are you using?** Tick Linux.

**Python Version:** 3.12.13

**axolotl branch-commit:** v0.21.1 (pip); also reproduces on main@de8549a4406327ab5311f7187a8afaa232645487

---

**Acknowledgements**
- Title: concise and descriptive (sentence case).
- Searched existing issues: done.
- "I am using the latest version of axolotl": 0.21.1 is the latest release, and main also reproduces.
- Enough information to reproduce: the script above.

---

Happy to open a PR with a fix and tests covering the templates above, or to help extend #3825 if that's the preferred direction.
