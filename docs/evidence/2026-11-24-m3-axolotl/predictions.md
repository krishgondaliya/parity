# Predictions: Axolotl 0.21.1 on Qwen3-0.6B, from Axolotl's docs alone

Written and committed before the Axolotl adapter existed and before any check ran on Axolotl. Sources, all at v0.21.1:

- `docs/dataset-formats/conversation.qmd`
- `docs/installation.qmd`
- The config reference. It is generated from the field descriptions in `axolotl/utils/schemas/*.py`, so those descriptions are the documentation.

I had also read some of Axolotl's code while looking for where `axolotl preprocess` and the collator live: the default `excess_length_strategy` handling, and the collator builder. Nothing below relies on code that the docs don't also state.

## The config these predictions assume

The documented recipe for "train on assistant turns only":

```yaml
base_model: <the release's local snapshot: tokenizer, template and config files only>
datasets:
  - path: <a jsonl of the conversation(s)>
    type: chat_template
    roles_to_train: ["assistant"]   # documented default ("# default value"), set explicitly
    train_on_eos: turn              # documented default; train_on_eot follows it
sequence_len: <the check's limit, or 8192 when the check wants no limit>
```

- `chat_template` is left unset. Documented default: `tokenizer_default`, Qwen3's own template.
- `excess_length_strategy` is left unset. Documented default: `drop`.
- `sample_packing` is left unset (off), so batches are padded. `train_on_inputs` defaults to false. `generation_tags` is off.
- `roles_to_train` and `train_on_eos` are set explicitly because the docs' legacy example ("Training on last message") leaves both empty and trains only the last message.

## What the docs say that matters

1. **Template:** `tokenizer_default` uses the model's own chat template, so the tokens should be Qwen3's official render.
2. **Which tokens get loss:**
   - `generation_tags` is described as labeling "assistant turns from the chat template's {% generation %} blocks, including the template structure inside them, instead of locating each turn's fields".
   - So the default (no `generation_tags`) locates each turn's fields (content, reasoning_content, tool calls) and trains those, not the template structure inside the turn.
   - Qwen3's own template renders an empty think block, `<think>\n\n</think>\n\n`, at the start of the last assistant turn, and wraps reasoning in `<think>\n…\n</think>\n\n`. Our reference counts that structure as must-train: in thinking mode the model generates it, because the generation prompt ends at `<|im_start|>assistant\n`.
3. **End of turn:** `train_on_eos: turn` trains "the EOS token at the end of each trainable turn". Qwen3-0.6B's EOS is `<|im_end|>`, the token that closes every assistant turn.
4. **Over-length rows:** `excess_length_strategy` defaults to `drop`, which "removes the row" when a tokenized row "exceeds sequence_len". So a row of exactly `sequence_len` is kept.
5. **Batching:** `sample_packing` is off by default, so batches are padded. `pad_to_multiple_of`: "Pad each batch to a multiple of this value."

## Predicted verdicts (48 cells)

| Check | single_turn | multi_turn | system_prompt | tool_call | reasoning | empty_assistant | long_answer | non_ascii |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| t1.template_parity | pass | pass | pass | pass | pass | pass | pass | pass |
| t1.assistant_mask | **fail** | **fail** | **fail** | **fail** | **fail** | **fail** | **fail** | **fail** |
| t1.end_of_turn | pass | pass | pass | pass | pass | pass | pass | pass |
| t1.special_tokens | pass | pass | pass | pass | pass | pass | pass | pass |
| t1.truncation | **fail** | **fail** | **fail** | **fail** | **fail** | **fail** | pass | **fail** |
| t1.multiturn_prefix | n/a | pass | n/a | pass | pass | pass | n/a | n/a |

Totals: 29 pass, 15 fail, 4 not applicable. Every applicable self-test caught.

### Why

- **Template parity, all pass:** `tokenizer_default` is Qwen3's own template, so no training template is swapped in. Risk: the `datasets` round trip on `tool_call`. The docs warn only about tool arguments with mixed types, which this fixture doesn't have.
- **Assistant mask, all fail:** every fixture's last assistant turn starts with think-block structure the default labeling doesn't train (point 2):
  - the empty block on 7 fixtures;
  - the wrappers around the reasoning on `reasoning`.
  - On `tool_call`, the `<tool_call>` wrapper of the first turn may also be untrained.
  - The verdict would list must-train tokens without loss: `<think>`, `\n\n`, `</think>`, `\n\n` at the start of the last turn.
  - Confidence: medium-high. It rests on one sentence of the `generation_tags` description.
- **End of turn, all pass:** point 3 trains `<|im_end|>` at the end of every assistant turn.
- **Special tokens, all pass:**
  - Tokens equal the official render, and `<|im_end|>` keeps its loss.
  - The padded batch uses the tokenizer's pad token (`<|endoftext|>`) with no loss and doesn't change the row.
  - Not predicted from the docs, so a risk: which side Axolotl pads on (either side passes), and whether it pads to exactly `pad_to`.
- **Truncation:**
  - `long_answer` passes: 537 tokens > 128, and the declared mode is `drop`.
  - The 7 boundary fixtures run at exactly their length, so they are kept (point 4). But their loss is judged with the same zone rule as the assistant mask, so they fail for the same root cause.
  - These 7 fails duplicate the assistant-mask root cause; the "blocked" rule only covers template parity.
- **Multi-turn prefix, 4 pass:** Axolotl renders with Qwen3's own template, so it breaks the prefix exactly where the template does. That is inherited and passes, naming Qwen3's prefix-break property. The 4 single-assistant fixtures are not applicable.

## What a fail here would mean

If the assistant-mask prediction holds, it is a triage question, not yet a finding:
- If Axolotl's docs make leaving template structure untrained deliberate and documented (the `generation_tags` sentence suggests so), it becomes a known-differences entry with that doc link.
- If they don't, it's a candidate finding. A Qwen3 model fine-tuned this way in thinking mode is never trained to produce its own think tags.
