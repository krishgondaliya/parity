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
