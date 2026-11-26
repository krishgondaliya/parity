# Predictions: Axolotl 0.21.1 on deepseek-ai/DeepSeek-V4.1-Flash-1017@08c8657f02291e11b6f6b63e0e16014f8e748518 (both serving modes)

Written from Axolotl's docs and source, the model's template and our reference alone. Committed before Axolotl runs on DeepSeek V4.1 in the M5 suite. All 96 cells are graded: 48 primary (thinking) and 48 secondary (non-thinking).

**Not blind, stated plainly.** These were known when this file was written, and they are in the repo:
- **The support check** (`docs/evidence/2026-11-25-m5-env-support/`): Axolotl 0.21.1 can't build `deepseek_v41`, so the model is the `deepseek_v4` stand-in (D1) and `base_model` is a folder holding the stand-in's tiny config next to the release's own tokenizer and template.
- **The coverage preview** (`docs/evidence/2026-11-25-m5-coverage-preview/`): our reference refuses `tool_call` and `empty_assistant` on this model.
- **P-0002 on the targets** (`findings/P-0002/target_reach.py`, the human's request, 2026-11-26): Axolotl's `chat_template` strategy already ran on this tokenizer. It leaves an empty turn's `<｜end▁of▁sentence｜>` unlabeled and labels the control turns' end token. That is the one piece of Axolotl's data path that has run here, and it touches only `empty_assistant`, which our reference refuses anyway.
- **Qwen3 calibration with this adapter:** 96 of 96 identical to M4 (`docs/evidence/2026-11-26-m4-recalibration/`), including P-0001's pattern (the think opener untrained; KD-0002/KD-0003).

## Sources
- **Axolotl 0.21.1** (files cited at the tag were checked byte-identical to the installed ones):
  - `type: chat_template` with `chat_template` unset renders with the tokenizer's own template (`docs/dataset-formats/conversation.qmd`);
  - `roles_to_train: ["assistant"]` and `train_on_eos: turn` train each assistant turn and its EOS;
  - `eot_tokens` defaults to the tokenizer's EOS, `<｜end▁of▁sentence｜>`, which the template contains literally, so no warning (`prompt_strategies/chat_template.py` L422-L432);
  - `chat_template_kwargs` sets the serving mode (`utils/schemas/config.py` L1497-L1502).
- **How Axolotl finds a turn** (`chat_template.py`, `_find_turn_from_text`): it renders the conversation with the turn's content swapped for sentinel strings and diffs the renders. The trained span is the characters that change, the content, plus the EOS within three tokens after it. Anything the template emits around the content whatever the content is stays outside the span.
- **Our reference for DeepSeek V4.1:**
  - **Thinking mode:** the last turn renders `<｜Assistant｜><think></think>{content}<｜end▁of▁sentence｜>`. The generation prompt ends at `<think>`, so `</think>` is must-train, and on `reasoning` so are the reasoning text and its `</think>`. History turns render `<｜Assistant｜></think>{content}<eos>`, and that `</think>` is a pre-fill, so prompt side.
  - **Non-thinking mode:** every turn's `</think>` is the generation prompt's pre-fill, so prompt side.

## What the sources say about each part of the data path
1. **Template and rows:** DeepSeek's own template, rendered over the whole conversation. Rows equal the template's render (P-0002's target check saw exactly that on this tokenizer).
2. **Labels, thinking mode:** content plus EOS per assistant turn. The last turn's `</think>`, must-train for our reference, sits before the content in every render, so it stays outside the span: untrained.
   - On `reasoning`, the empty-versus-filled reasoning diff covers the reasoning text, but the `</think>` after it is in neither span.
   - This is P-0001's mechanism (the think opener on Qwen3). KD-0002 and KD-0003 are scoped to Qwen3's template, so on DeepSeek these are fails of kind bug until triage.
3. **Labels, non-thinking mode:** content plus EOS, exactly the reference's must-train tokens.
4. **Padding:** Axolotl's collator pads input ids with the pad token, which here equals the EOS, and labels with -100. Real EOS labels are set before collation, so they keep their loss. `pad_to_multiple_of` is honored, as on Qwen3.
5. **Truncation:** `excess_length_strategy` unset, so Axolotl drops a row longer than `sequence_len`. On Qwen3, `long_answer` was dropped, and the truncation check passed it as declared.

## How our checks read that
- **Thinking (primary):**
  - Assistant mask fails on all 6 fixtures our reference computes: the last turn's `</think>` is must-train but untrained.
  - Truncation fails on the same fixtures except `long_answer`, which Axolotl drops (as on Qwen3).
  - Template parity, end of turn, special tokens and multi-turn prefix pass.
  - `tool_call` and `empty_assistant` are inconclusive in every check (our reference refuses).
- **Non-thinking (secondary):** every computed cell passes. `reasoning` is n/a (mode rule), and `tool_call` and `empty_assistant` are inconclusive (our reference refuses).

## Predicted cells: thinking (primary)

| Check | single_turn | multi_turn | system_prompt | tool_call | reasoning | empty_assistant | long_answer | non_ascii |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| t1.template_parity | pass | pass | pass | inconclusive | pass | inconclusive | pass | pass |
| t1.assistant_mask | **fail** | **fail** | **fail** | inconclusive | **fail** | inconclusive | **fail** | **fail** |
| t1.end_of_turn | pass | pass | pass | inconclusive | pass | inconclusive | pass | pass |
| t1.special_tokens | pass | pass | pass | inconclusive | pass | inconclusive | pass | pass |
| t1.truncation | **fail** | **fail** | **fail** | inconclusive | **fail** | inconclusive | pass | **fail** |
| t1.multiturn_prefix | n/a | pass | n/a | inconclusive | pass | inconclusive | n/a | n/a |

Counts: 21 pass, 11 fail, 12 inconclusive, 4 n/a.

## Predicted cells: non_thinking (secondary)

| Check | single_turn | multi_turn | system_prompt | tool_call | reasoning | empty_assistant | long_answer | non_ascii |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| t1.template_parity | pass | pass | pass | inconclusive | n/a | inconclusive | pass | pass |
| t1.assistant_mask | pass | pass | pass | inconclusive | n/a | inconclusive | pass | pass |
| t1.end_of_turn | pass | pass | pass | inconclusive | n/a | inconclusive | pass | pass |
| t1.special_tokens | pass | pass | pass | inconclusive | n/a | inconclusive | pass | pass |
| t1.truncation | pass | pass | pass | inconclusive | n/a | inconclusive | pass | pass |
| t1.multiturn_prefix | n/a | pass | n/a | inconclusive | n/a | inconclusive | n/a | n/a |

Counts: 26 pass, 12 inconclusive, 10 n/a.

## Also predicted (not graded cells)
- **Declaration:** `template_source` "model"; truncation drop; batching padded; pad token `<｜end▁of▁sentence｜>`; tool arguments parsed as JSON.
- **Model build:** `model_build` records the `deepseek_v4` stand-in (`DeepseekV4ForCausalLM`) with the deviation and its reason.
- **Rule for any fail here:** it is a stand-in run, so it leaves the repo only with a standalone repro that doesn't need the release's architecture (D1). Axolotl's span logic uses only the tokenizer and template, so such a repro is possible, as P-0002's was.

## Where this could be wrong (named before the run)
- **If Axolotl's reasoning handling also spans the template's think markers** on this template (it splits reasoning from content), the `</think>` fails disappear, and assistant mask and truncation pass in thinking mode.
- **If the stand-in's `deepseek_v4` model type switches on an Axolotl model-specific branch** (its model-support registry reads the type), the data path could differ from what a future `deepseek_v41` build would get. The declaration's sources show which branch ran.
