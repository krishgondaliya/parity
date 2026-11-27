# P-0008 discussion draft. NOT POSTED.

Target: the Hugging Face model repo `thinkingmachines/Inkling-2-Mini`, as a Community discussion (https://huggingface.co/thinkingmachines/Inkling-2-Mini/discussions, "New discussion"). There is no form: paste the title, then the body below the line, as Markdown. First run the repro as `findings/P-0008/HOW-TO-RUN.md` says (it should exit 1).

---

**Title**

[chat_template] Empty text part on a tool-call turn renders an empty text message, unlike "content": ""

---

**Body**

`chat_template.jinja` renders an assistant turn's empty content in two different ways, depending on its form. On a turn with `tool_calls`:

- `"content": ""`, `"content": null`, or no `content`: no text message. The string branch skips empty content on an assistant turn with tool calls or reasoning ([L75](https://huggingface.co/thinkingmachines/Inkling-2-Mini/blob/7a67ae2314b44ef05c87fd1d4d311c58aaeb18f6/chat_template.jinja#L75)).
- `"content": [{"type": "text", "text": ""}]`: an empty text message, plus a second `<|message_model|>`, before the tool call. The list branch renders every text part, empty or not ([L77-L83](https://huggingface.co/thinkingmachines/Inkling-2-Mini/blob/7a67ae2314b44ef05c87fd1d4d311c58aaeb18f6/chat_template.jinja#L77)).

The same `chat_template.jinja` ships in Inkling-2-Small and Inkling-2-Small-MXFP8. For the conversation in the script below, the tool-call turn renders as:

```
"content": ""                                ...<|message_model|>get_weather<|content_invoke_tool_json|>...
"content": [{"type": "text", "text": ""}]    ...<|message_model|><|content_text|><|end_message|><|message_model|>get_weather<|content_invoke_tool_json|>...
```

**What tml-renderers does with the same conversation**

I gave tml-renderers 0.5.0 that conversation four ways, with the tool-call turn's `content` as `""`, `null`, missing, and `[{"type": "text", "text": ""}]`. Each went through `chat.OpenAIMessage.from_openai_chat_dicts(messages)` and `renderers.Inkling2(effort=0.9).render_for_sft(...)`. I used effort 0.9 because the template adds "Thinking effort level: 0.9" by default. It accepted all four forms and returned the same output for each, with no empty text message:

```
<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>Weather in Paris?<|end_message|><|message_model|>get_weather<|content_invoke_tool_json|>{"name":"get_weather","args":{"city":"Paris"}}<|end_message|><|content_model_end_sampling|><|message_tool|>get_weather<|content_text|>18C<|end_message|><|message_model|><|content_text|>It's 18C.<|end_message|><|content_model_end_sampling|>
```

For `""`, `null` and missing content, that is exactly what `chat_template.jinja` renders, text and token ids. Only the list form differs, by `<|content_text|><|end_message|><|message_model|>`.

**Why the list form shows up in training data**

Given this model's id, TRL's `SFTTrainer` loads `Inkling2Processor` ([`sft_trainer.py` L715-L719](https://github.com/huggingface/trl/blob/v1.18.2/trl/trainer/sft_trainer.py#L715-L719)). It then wraps every string `content`, `""` included, as one text part before applying the template ([`data_utils.py` L88-L90](https://github.com/huggingface/trl/blob/v1.18.2/trl/data_utils.py#L88-L90), called at [L391-L394](https://github.com/huggingface/trl/blob/v1.18.2/trl/data_utils.py#L391-L394)). This is deliberate: several VLM processors reject `"content": ""` ([TRL PR #5496](https://github.com/huggingface/trl/pull/5496)). So with `assistant_only_loss=True`, a tool-call turn written with `"content": ""` gets those three tokens inside its `{% generation %}` block, and they are trained: a fine-tune is trained on an empty text message before each such tool call, which tml-renderers doesn't produce for any of these forms. `null` or missing content is unaffected.

**A possible change**

Skip an empty text part in the list branch under the same condition the string branch uses:

```diff
             {%- elif part.type is not defined or part.type in ("text", "input_text") -%}
                 {%- set text_part = (part.text if part.text is defined and part.text is string else "") -%}
-                {{- rtok ~ "<|content_text|>" ~ text_part ~ "<|end_message|>" -}}
+                {%- if text_part or not (message.role == "assistant" and (message.tool_calls or message.reasoning_content)) -%}
+                    {{- rtok ~ "<|content_text|>" ~ text_part ~ "<|end_message|>" -}}
+                {%- endif -%}
```

With this change, the template matches tml-renderers on all four forms, for this conversation and for a longer tool-call conversation. Otherwise it renders exactly as before on everything I tried: 8 test conversations, each as given and with every string content as one text part, a non-empty text part on a tool-call turn, and an empty text part on a user turn and on a plain assistant turn. The one render that changes is the tool-call conversation with every string content as one text part (what TRL passes), which then equals the render of the conversation as given.

**Repro** (CPU, a few seconds; downloads tokenizer and template files only)

`pip install transformers==5.24.0 jinja2==3.1.6` runs it; with `trl==1.18.2` installed too, the last line also shows TRL's conversion. It exits 1 while any form renders an empty text message.

```python
# transformers==5.24.0, jinja2==3.1.6 (trl==1.18.2 optional), Python 3.12, CPU. Tokenizer and template files only.
# Inkling-2-Mini's chat template renders a tool-call turn's empty content differently as "" and as one empty text
# part: only the list form gets an empty text message. Thinking Machines' tml-renderers 0.5.0 renders none. Exit 1.
import sys
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("thinkingmachines/Inkling-2-Mini", revision="7a67ae2314b44ef05c87fd1d4d311c58aaeb18f6")
call = {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "Paris"}}}


def conversation(content):
    return [{"role": "user", "content": "Weather in Paris?"},
            {"role": "assistant", "content": content, "tool_calls": [call]},
            {"role": "tool", "tool_call_id": "c1", "name": "get_weather", "content": "18C"},
            {"role": "assistant", "content": "It's 18C."}]

EMPTY = "<|content_text|><|end_message|>"  # an empty text message
forms = {'""': "", "null": None, '[{"type": "text", "text": ""}]': [{"type": "text", "text": ""}]}
renders = {form: tok.apply_chat_template(conversation(c), tokenize=False) for form, c in forms.items()}
print("expected (tml-renderers 0.5.0): no empty text message for any form")
for form, text in renders.items():
    print(f"content {form}: {text.count(EMPTY)} empty text message(s)")
try:  # optional: TRL's SFTTrainer with a processor converts string content to parts before rendering
    from trl.data_utils import prepare_multimodal_messages
    wrapped = prepare_multimodal_messages(conversation(""))  # conversation() builds fresh dicts
    print(f'TRL turns content "" into {wrapped[1]["content"]}: '
          f"{tok.apply_chat_template(wrapped, tokenize=False).count(EMPTY)} empty text message(s)")
except ImportError:
    pass
sys.exit(1 if any(text.count(EMPTY) for text in renders.values()) else 0)
```

Output:

```
expected (tml-renderers 0.5.0): no empty text message for any form
content "": 0 empty text message(s)
content null: 0 empty text message(s)
content [{"type": "text", "text": ""}]: 1 empty text message(s)
TRL turns content "" into [{'type': 'text', 'text': ''}]: 1 empty text message(s)
```

Environment: transformers 5.24.0, tokenizers 0.23.3, jinja2 3.1.6, trl 1.18.2, Python 3.12.13, Linux, CPU. Model revision `7a67ae2`; `chat_template.jinja` is unchanged at the current main (`c9d0283`).

I may be missing a reason the two forms are meant to render differently. If so, it would help to know which form training data should use for tool-call turns with no text.
