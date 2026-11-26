# Predictions: Unsloth 2026.11.10 on deepseek-ai/DeepSeek-V4.1-Flash-1017@08c8657f02291e11b6f6b63e0e16014f8e748518, GPU backend (both serving modes)

Written from Unsloth's docs, its source, the model's own files and our reference alone. Committed before the GPU kit that runs this pair is built. All 96 cells are graded: 48 primary (thinking) and 48 secondary (non-thinking).

**Not blind, stated plainly.** These were known when this file was written, and they are in the repo:
- **The support check** (`docs/evidence/2026-11-25-m5-env-support/`): Unsloth 2026.11.10 builds the release's own architecture (`DeepseekV41ForCausalLM`, its own implementation, registered at import; text only).
- **The tiny model factory** built that architecture on CPU (`tests/test_tiny.py`). 4 of the 7 documented default LoRA module names exist in it.
- **The role-marker check on our reference** (`docs/evidence/2026-11-26-unsloth-deepseek-parts/`): it found which reference zones fall inside the region the parts mark as response. That check is the main input to the assistant-mask prediction below.
- **Qwen3 on Unsloth's CPU route:** 96 of 96 identical before and after the adapter change. Nothing of Unsloth's data path has run on this model, on any backend.

## Citation level
**General docs; training this model is not documented** (the human's decision, 2026-11-26). Unsloth's DeepSeek-V4 guide documents running V4 models, not training V4.1. A finding here leaves the repo only if it reproduces with documented choices.

## Sources
- **Unsloth's general docs:**
  - the fine-tuning guide (https://unsloth.ai/docs/get-started/fine-tuning-llms-guide): "`max_seq_length = 2048`", "`load_in_4bit = True`", batch size 2, gradient accumulation 4, 60 steps, learning rate 2e-4;
  - the LoRA hyperparameters guide: the default `target_modules` list, and "you will need to define the instruction and assistant parts" for `train_on_responses_only`, with examples using each template's user and assistant headers;
  - the chat-templates page: `tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)` into a `text` column.
- **The model's own `encoding/README.md`** at the pinned revision: `<｜User｜>` "User turn prefix", `<｜Assistant｜>` "Assistant turn prefix". Those are the parts.
- **Unsloth 2026.11.10's source:**
  - `FastLanguageModel.from_pretrained` hands a model type it has no fast path for to `FastModel` (`models/loader.py` L1991).
  - `FastModel` has DeepSeek V4.1 handling (L2958) and falls back to its causal LM because no image-text class is registered (L3490-L3509).
  - `train_on_responses_only` masks everything up to and including each response part, and trains from there up to the next instruction part (`unsloth_zoo/dataset_utils.py`).

## The recipe
`src/parity/adapters/unsloth.py`, `RECIPES["deepseek-ai/DeepSeek-V4.1-Flash-1017"]`:
- `FastLanguageModel.from_pretrained(max_seq_length=2048, load_in_4bit=True)` on the tiny release-architecture folder;
- `get_peft_model` with the default list;
- the guide's trainer settings, with `padding_free` left to Unsloth;
- the parts `<｜User｜>` and `<｜Assistant｜>`;
- non-default serving modes `not_configurable`.

## What the sources say about each part of the data path
1. **Rows:** the template's render, tokenized as text. It already starts with `<｜begin▁of▁sentence｜>` and ends each turn with `<｜end▁of▁sentence｜>`, the EOS, so TRL appends nothing, and the tokenizer adds no BOS (`add_bos_token: false`). On the GPU route Unsloth's text rows equal the render (P-0004's GPU check, on Qwen3).
2. **Labels:**
   - Everything after each `<｜Assistant｜>` is trained, up to the next `<｜User｜>`. That includes every token our reference must train (the role-marker check found none outside).
   - It also includes tokens our reference keeps on the prompt side: the generation prompt's `<think>` on the last turn in thinking mode, and the pre-filled `</think>` on history turns.
   - A single response part can't separate them: the generation prompt is `<｜Assistant｜><think>` on the last turn and `<｜Assistant｜></think>` on history turns.
3. **Padding:** the recipe leaves `padding_free` unset, and Unsloth then turns padding-free batching on (measured on GPU for the Qwen3 Thinking-2507 recipe, which leaves it unset too; `docs/STATUS.md`). A padding-free batch is declared as packed.
4. **Truncation:** `SFTConfig.max_length`, keeping the start (TRL), as on Qwen3.

## How our checks read that
- **Thinking (primary):**
  - **Assistant mask** fails on all 6 fixtures our reference computes: prompt-side `<think>` and `</think>` tokens are trained.
  - **Truncation** fails on the same 6: it judges the kept labels against the same zones.
  - **Special tokens** is inconclusive on the same 6: packing is declared for the padded batch, which Phase 0 can't judge.
  - **Template parity, end of turn and multi-turn prefix** pass.
  - **`tool_call` and `empty_assistant`** are inconclusive everywhere (our reference refuses).
- **Non-thinking (secondary):** 38 `not_configurable` (Unsloth documents no training recipe for this model, so no way to set its mode in training data), 10 n/a.

## Predicted cells: thinking (primary)

| Check | single_turn | multi_turn | system_prompt | tool_call | reasoning | empty_assistant | long_answer | non_ascii |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| t1.template_parity | pass | pass | pass | inconclusive | pass | inconclusive | pass | pass |
| t1.assistant_mask | **fail** | **fail** | **fail** | inconclusive | **fail** | inconclusive | **fail** | **fail** |
| t1.end_of_turn | pass | pass | pass | inconclusive | pass | inconclusive | pass | pass |
| t1.special_tokens | inconclusive | inconclusive | inconclusive | inconclusive | inconclusive | inconclusive | inconclusive | inconclusive |
| t1.truncation | **fail** | **fail** | **fail** | inconclusive | **fail** | inconclusive | **fail** | **fail** |
| t1.multiturn_prefix | n/a | pass | n/a | inconclusive | pass | inconclusive | n/a | n/a |

Counts: 14 pass, 12 fail, 18 inconclusive, 4 n/a.

## Predicted cells: non_thinking (secondary)

| Check | single_turn | multi_turn | system_prompt | tool_call | reasoning | empty_assistant | long_answer | non_ascii |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| five checks | not_configurable | not_configurable | not_configurable | not_configurable | n/a | not_configurable | not_configurable | not_configurable |
| t1.multiturn_prefix | n/a | not_configurable | n/a | not_configurable | n/a | not_configurable | n/a | n/a |

Counts: 38 not_configurable, 10 n/a.

## Also predicted (not graded cells)
- **Declaration:** `template_source` "model"; truncation right; batching packed (padding-free); tool arguments passed through.
- **Model build:** `model_build` records `DeepseekV41ForCausalLM`, path text, no stand-in, citation level "general docs; training this model is not documented".
- **Backend:** every verdict records `gpu`.

## Where this could be wrong (named before the run)
- **If Unsloth keeps padded batches** (`padding_free` not turned on), special tokens is judged. Then P-0005 applies: `train_on_responses_only` drops `pad_to_multiple_of` on the padded path, the width rule fails on the 6 fixtures (root cause P-0005), and pad handling passes. The pad token is the EOS here, and labels are set before collation.
- **If the GPU loader can't 4-bit-load the tiny DeepSeek V4.1 model** (its packed experts, or the dimensions), every cell is a harness error (inconclusive), and the adapter, not Unsloth, is the first suspect (rule 4).
- **If Unsloth's GPU loader edits the chat template** (its post-load template repairs), the declaration's `template_source` stops being "model", and template parity judges the edited template. It left Qwen3's unchanged.
- **Any assistant-mask fail goes to triage first:** whether a single response part is what Unsloth's docs imply for a template whose generation prompt differs between the last turn and history turns.
