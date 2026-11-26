# Predictions: TRL 1.18.2 (vlm variant) on thinkingmachines/Inkling-2-Mini@7a67ae2314b44ef05c87fd1d4d311c58aaeb18f6 (both serving modes)

Written from TRL's docs and source, Inkling's model card and template, and our reference alone. Committed before TRL runs on Inkling-2-Mini. All 96 cells are graded: 48 primary (thinking) and 48 secondary (non-thinking).

**Not blind, stated plainly.** These were known when this file was written, and they are in the repo:
- **The coverage preview** (`docs/evidence/2026-11-25-m5-coverage-preview/`): on our side, every primary cell can be judged, and Inkling's generation markers agree with our reference on all 8 fixtures.
- **The support check** (`docs/evidence/2026-11-25-m5-env-support/`):
  - TRL 1.18.2's environment builds `Inkling2LoopForConditionalGeneration`.
  - `Inkling2Processor` builds only in the `vlm` variant; without `torchvision` its image processor fails.
- **The tiny model factory** builds the release's own vision model in the `vlm` variant (`tests/test_tiny.py`).
- **Qwen3 calibration in the `vlm` variant:** 96 of 96 identical to M4 (`docs/evidence/2026-11-26-m4-recalibration/`).

Nothing of TRL's data path has run on Inkling. The template-parity prediction below comes from reading TRL's source and Inkling's template side by side.

## Sources
- **TRL 1.18.2** (installed files byte-identical to the tag):
  - `create_model_from_path` builds `config.architectures[0]` (`trl/trainer/utils.py` L1362-L1375).
  - `SFTTrainer` loads `AutoProcessor` when the model accepts images (`sft_trainer.py` L714-L718).
  - `get_training_chat_template` returns `None` (no patch) for a template with `{% generation %}` markers that isn't in TRL's stored list (`chat_template_utils.py`).
  - A conversational row goes through `_tokenize` (`data_utils.py` L367). For a processor it first calls `prepare_multimodal_messages`, which wraps every string `content` as `[{"type": "text", "text": ...}]`, the empty string included, then calls the processor's `apply_chat_template(..., return_assistant_tokens_mask=True)`.
  - A text-only dataset gets `DataCollatorForLanguageModeling` with the tokenizer's pad id and `pad_to_multiple_of` (`sft_trainer.py` L1039-L1060).
- **The `vlm` extra:** TRL's package metadata (`Provides-Extra: vlm`: Pillow, torchvision, num2words). The human's decision, 2026-11-26.
- **Inkling's card at the pinned revision:** "Setting reasoning effort to its minimum does not fully disable reasoning" (L202). So there's no documented non-thinking mode (D3).
- **Inkling's `chat_template.jinja`** at the pinned revision:
  - each assistant message is rendered inside `{% generation %}` (L127-L131), header and closing `<|content_model_end_sampling|>` included;
  - a string `content` renders a text block unless it is empty on an assistant turn with `tool_calls` or reasoning (L75-L76);
  - a list `content` renders one text block per part, even an empty one (L77-L83).

## The recipe
- `src/parity/adapters/trl.py`, in `envs/trl-1.18.2/vlm`.
- The model is a tiny `Inkling2LoopForConditionalGeneration`, the release's own architecture on the vision path. The processing class is `Inkling2Processor`, chosen by TRL's own `model_accepts_images`.
- `SFTConfig(assistant_only_loss=True)`, with the other settings as on Qwen3.

## What the sources say about each part of the data path
1. **Template:** Inkling's own (`template_source` "model"). It has generation markers and isn't in TRL's stored list, so TRL patches nothing.
2. **Rows:** the processor's render of TRL's wrapped messages. The wrapping changes one thing on our fixtures:
   - **`tool_call`:** its first assistant turn has `content: ""` plus a tool call. As a string, Inkling's template skips that turn's text block. As TRL's `[{"type": "text", "text": ""}]`, it emits `<|message_model|><|content_text|><|end_message|>` before the tool call: three tokens the model's own render doesn't have.
   - **Unchanged:** every other content is non-empty, or is an empty turn without tool calls or reasoning (`empty_assistant`), which renders the same either way.
3. **Labels:** the template's generation markers, which our preview found agree with our reference on all 8 fixtures. That covers each assistant message from its header through `<|end_message|>` and `<|content_model_end_sampling|>`.
4. **Padding:** right, with `<|endoftext|>` (Inkling's pad token, not its stop token), honoring `pad_to_multiple_of`, as on Qwen3.
5. **Truncation:** `max_length`, keeping the start, as on Qwen3.

## How our checks read that
- **Thinking mode (primary):**
  - **`tool_call`:** template parity fails (kind bug: the extra empty text block, an interaction between TRL's VLM wrapping and Inkling's template; rule 4 first). The other five checks on that fixture can't line its row up with the declared template's render, so they're inconclusive (framework-side, explained by the template parity fail).
  - **Every other fixture:** pass. The multi-turn prefix check passes on its 3 other multi-assistant fixtures and is n/a on the 4 single-assistant ones.
- **Non-thinking mode (secondary):** every cell is not applicable (D3: the card documents no non-thinking mode), decided before TRL is asked anything.

## Predicted cells: thinking (primary)

| Check | single_turn | multi_turn | system_prompt | tool_call | reasoning | empty_assistant | long_answer | non_ascii |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| t1.template_parity | pass | pass | pass | **fail** | pass | pass | pass | pass |
| t1.assistant_mask | pass | pass | pass | inconclusive | pass | pass | pass | pass |
| t1.end_of_turn | pass | pass | pass | inconclusive | pass | pass | pass | pass |
| t1.special_tokens | pass | pass | pass | inconclusive | pass | pass | pass | pass |
| t1.truncation | pass | pass | pass | inconclusive | pass | pass | pass | pass |
| t1.multiturn_prefix | n/a | pass | n/a | inconclusive | pass | pass | n/a | n/a |

Counts: 38 pass, 1 fail, 5 inconclusive, 4 n/a.

## Predicted cells: non_thinking (secondary)

All 48 cells: n/a (D3).

## Also predicted (not graded cells)
- **Declaration:** `template_source` "model"; truncation right; batching padded; pad token `<|endoftext|>`; tool arguments passed through.
- **Model build:** `model_build` records `Inkling2LoopForConditionalGeneration`, path vision, no stand-in.
- **Lockfile:** every verdict records `env_lock` `envs/trl-1.18.2/vlm/uv.lock`.

## Where this could be wrong (named before the run)
- **If `model_accepts_images` is False for the tiny model,** TRL loads a tokenizer instead of the processor and nothing is wrapped. Then `tool_call` passes, all 44 judged cells pass, and the declaration's processing class says which happened.
- **If the processor's `apply_chat_template` normalizes content differently** (e.g. drops an empty text part), `tool_call` passes too.
- **If `return_assistant_tokens_mask` behaves differently through a processor** than through a tokenizer, assistant mask and end of turn would show it on every fixture.
