# Predictions: Unsloth 2026.11.10 on Qwen3-0.6B, GPU backend (both serving modes)

Written from Unsloth's docs and notebooks alone, and committed before the GPU run kit exists and before anything runs on a GPU. All 96 cells are graded: 48 primary (thinking) and 48 secondary (non-thinking).

**Not blind, stated plainly.** Before writing these, I already knew three things, and they are in the repo:
- the CPU-backend run (`docs/evidence/2026-11-24-m3-unsloth/`);
- P-0004's reading of Unsloth's GPU-route code;
- P-0005's reading of `train_on_responses_only`.

I didn't use them to choose any cell. Each cell below follows from the docs, plus how our checks treat what the docs describe. The "Where the earlier evidence disagrees" section names, in advance, the cells where the repo's code-based evidence points elsewhere. The headline score is the docs-alone score. A miss on one of those named cells means the docs don't tell a user what they get.

## Sources (docs only)
- **Unsloth docs**, all fetched 2026-11-25 as Markdown. Pages aren't versioned, so each is pinned by the first 12 hex digits of its sha256.
  - The Qwen3 page, https://unsloth.ai/docs/models/tutorials/qwen3-how-to-run-and-fine-tune (`9535fcea8c7e`):
    - "By default, Qwen3 has thinking enabled. When you call `tokenizer.apply_chat_template`, you don't need to set anything manually";
    - `enable_thinking=False` "Disables thinking mode";
    - "Qwen3 (14B) fits comfortably in a Google Colab 16GB VRAM Tesla T4 GPU";
    - it marks the Reasoning + Conversational notebook as recommended.
  - The requirements page, https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements (`436ed3047de9`): "Minimum CUDA Capability 7.0 (V100, T4, …)".
  - The LoRA hyperparameters guide, "Training on completions only, masking out inputs", https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide (`8a383a869906`):
    - the user turns are struck out and the assistant answers are highlighted, through the final ".";
    - "you will need to define the instruction and assistant parts", passed to `train_on_responses_only(trainer, instruction_part=..., response_part=...)`.
  - The packing page, https://unsloth.ai/docs/blog/3x-faster-training-packing (`ade433d93f0d`):
    - "we will **automatically use padding-free batching**";
    - "simply set `packing=False` and we will enable auto padding-free";
    - padding-free gives "the exact same loss".
  - The datasets guide, https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide (`5ff4c86d6b9a`), and the fine-tuning guide, https://unsloth.ai/docs/get-started/fine-tuning-llms-guide (`bd37bc85a28f`):
    - Unsloth fixes chat templates "when we upload the quantized versions to our repositories";
    - its no-suffix uploads "sometimes include … chat template or tokenizer fixes".
    - So the fixes ship in Unsloth's own model repos, not in the loader.
  - The fine-tuning guide: "`max_seq_length = 2048` – Controls context length".
- **The recommended notebook**, "Qwen3 (14B) Reasoning + Conversational", `unslothai/notebooks@ebe88117` (`nb/Qwen3_(14B)-Reasoning-Conversational.ipynb`):
  - `FastLanguageModel.from_pretrained(max_seq_length=2048, load_in_4bit=True, ...)`;
  - `tokenizer.apply_chat_template(conversations, tokenize=False)`;
  - `SFTConfig(dataset_text_field="text", ..., padding_free = False, # Set to True if > 17 GB VRAM)`;
  - `enable_thinking=False` at inference, for non-thinking mode.
- **The assistant-only step**, from the Qwen3-4B-Thinking-2507 notebook (`unslothai/notebooks@0c6268b8`, `nb/Qwen3_(4B)-Thinking.ipynb`):
  - `train_on_responses_only(trainer, instruction_part="<|im_start|>user\n", response_part="<|im_start|>assistant\n")`;
  - the next cells print `tokenizer.decode(trainer.train_dataset[100]["input_ids"])` and the labels decoded with -100 shown as spaces. The saved output, from a T4 run of Unsloth 2025.8.2 (old, so not evidence about 2026.11.10), shows the user turn blank and the label text starting at `<think>\n`. Colab truncated the output, so the row's end isn't visible.
- **TRL 1.15.0's SFT docs**: `max_length` truncates, `truncation_mode` defaults to `keep_start`, and `pad_to_multiple_of` is an `SFTConfig` field. Unsloth's recipe runs TRL's `SFTTrainer`.

## The recipe
It is unchanged from the CPU run (`src/parity/adapters/unsloth.py`, `docs/DECISIONS.md` 2026-11-24), and only the backend differs:
- the recommended notebook's loader, formatting and trainer, including its explicit `padding_free=False`;
- plus the explicit-parts `train_on_responses_only` call;
- non-thinking mode: `enable_thinking=False` passed to the formatting `apply_chat_template`.

## What the docs say about each part of the data path
1. **Backend.** A T4 (capability 7.5) is supported, so `FastLanguageModel` takes its GPU route. The kit records the backend and refuses to run on CPU.
2. **Template.**
   - The notebook formats with the tokenizer's own template; no `get_chat_template`.
   - Unsloth's template fixes ship in its uploaded repos, and we load Qwen's official files.
   - So the declared template is the model's own (`template_source` "model").
3. **Rows.**
   - The notebooks present the formatted text as what is trained (they decode `train_dataset[i]["input_ids"]` as the check).
   - No Unsloth doc and no TRL doc says an EOS is appended to a `text` row.
   - So the row is exactly the template's render: no extra `<|im_end|>`.
4. **Labels.**
   - The completions-only docs: user turns get no loss, and assistant answers get loss through their end.
   - The parts are plain strings. Every span that starts with `<|im_start|>user\n` is masked; that includes Qwen3's tool results, which its template renders as user turns.
   - Everything after `<|im_start|>assistant\n` is trained. That is the think block (the notebook's labels start at `<think>\n`), the content and the turn's `<|im_end|>`, and for an empty turn just its `<|im_end|>`.
   - The parts don't depend on the mode, and Qwen3's full render (no generation prompt) is the same in both modes. So the labels are the same in both modes.
5. **Padding.**
   - Padding-free is Unsloth's default, but the notebook sets `padding_free = False` explicitly, and its comment ("Set to True if > 17 GB VRAM") says that setting is honored.
   - So batches are padded, by the tokenizer's pad token.
   - Nothing in the docs says `train_on_responses_only` touches the collator, so `pad_to_multiple_of` (a documented `SFTConfig` field) is honored.
6. **Truncation.** The loader's `max_seq_length` and `SFTConfig.max_length` get the same value (the adapter sets both), and TRL keeps the start.

## How our checks read that
- **Thinking mode (primary):**
  - the reference trains the think block, the content and the closing `<|im_end|>`, and masks the rest;
  - so 3 to 6 above give pass everywhere, and multi-turn prefix inherits the template's own breaks (still a pass, labeled with the template property).
- **Non-thinking mode (secondary):**
  - the reference treats the last turn's empty think block (`<think>\n\n</think>\n\n`) as the generation prompt's pre-fill, so prompt side, must not train;
  - Unsloth trains it (point 4), so assistant mask and truncation fail on every fixture whose last turn has the pre-fill. That is all of them except `reasoning`.
  - No known-differences entry covers it, so these are fails of kind bug, untriaged until /triage.
  - End of turn, special tokens and template parity don't depend on that zone, so they pass.
  - `reasoning` is inconclusive in every check: our reference refuses that fixture in non-thinking mode.
- **Not applicable:** multi-turn prefix on the four fixtures with fewer than two assistant turns, decided from the fixture alone.

## Predicted cells: thinking (primary)

| Check | single_turn | multi_turn | system_prompt | tool_call | reasoning | empty_assistant | long_answer | non_ascii |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| t1.template_parity | pass | pass | pass | pass | pass | pass | pass | pass |
| t1.assistant_mask | pass | pass | pass | pass | pass | pass | pass | pass |
| t1.end_of_turn | pass | pass | pass | pass | pass | pass | pass | pass |
| t1.special_tokens | pass | pass | pass | pass | pass | pass | pass | pass |
| t1.truncation | pass | pass | pass | pass | pass | pass | pass | pass |
| t1.multiturn_prefix | n/a | pass | n/a | pass | pass | pass | n/a | n/a |

## Predicted cells: non_thinking (secondary)

| Check | single_turn | multi_turn | system_prompt | tool_call | reasoning | empty_assistant | long_answer | non_ascii |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| t1.template_parity | pass | pass | pass | pass | inconclusive | pass | pass | pass |
| t1.assistant_mask | **fail** | **fail** | **fail** | **fail** | inconclusive | **fail** | **fail** | **fail** |
| t1.end_of_turn | pass | pass | pass | pass | inconclusive | pass | pass | pass |
| t1.special_tokens | pass | pass | pass | pass | inconclusive | pass | pass | pass |
| t1.truncation | **fail** | **fail** | **fail** | **fail** | inconclusive | **fail** | **fail** | **fail** |
| t1.multiturn_prefix | n/a | pass | n/a | pass | inconclusive | pass | n/a | n/a |

## Also predicted (not graded cells)
- Declaration:
  - `template_source` "model";
  - truncation right;
  - batching padded;
  - pad token `<|endoftext|>`, which is not Qwen3's stop token;
  - `pad_to_multiple_of` honored.
- Tool-call arguments given as a string: rendered verbatim, verified consistent by multi-turn prefix on `tool_call`.
- Every verdict records backend `gpu`.

## Where the earlier evidence disagrees (named before the run)
- **Special tokens, every cell that runs:** the docs say pass. P-0005 (CPU run plus code reading) says `train_on_responses_only` replaces the collator and drops `pad_to_multiple_of`. Then the padding part's batch isn't padded to `pad_to`, which our check makes inconclusive. If these come back inconclusive, P-0005 holds on GPU, and that counts as a docs-alone miss.
- **Template parity:** the docs and P-0004's GPU-route code reading agree: no extra `<|im_end|>`. P-0004 also notes that a failed GPU patch only logs a warning and falls back to stock TRL, which appends one. If template parity fails with an extra `<|im_end|>` at the end, the first thing to check is whether the patch applied.
- **Padding-free:** the docs and P-0005's code reading agree that the explicit `padding_free=False` keeps batches padded. If the GPU route turned on padding-free anyway, the declaration would say packed, and special tokens' padding part would be inconclusive for that reason instead.
- **Truncation:** no earlier evidence on GPU. P-0004 left open whether the GPU trainer patch changes `max_length`.

## Not predicted
- **Root causes** for the predicted non-thinking fails: they are new until triaged. Rule 4 applies before anything else: our reference's pre-fill zone and the mode setting are the first suspects.
- **The M5 target models:** the kit takes a model list, but these predictions cover only the calibration model. Each target needs its own predictions before it runs.
