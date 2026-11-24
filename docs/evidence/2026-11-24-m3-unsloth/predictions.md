# Predictions: Unsloth 2026.11.10 on Qwen3-0.6B (primary, thinking mode)

Committed before any check ran on Unsloth and before the adapter existed.

**Not docs alone, stated plainly.** Before writing these, I ran two probes in Unsloth's environment:
- a loader probe, needed to find out whether Unsloth accepts the local tiny-model folder;
- a data-path probe on 4 fixtures (`single_turn`, `reasoning`, `empty_assistant`, `tool_call`).

The data-path probe showed behavior these predictions were meant to call blind. So each cell group below names its basis: **docs** (Unsloth's docs and notebook), **code** (read, not run), or **probe** (observed). The cells that rest on the probe are not independent predictions.

## Sources
- Unsloth's Qwen3 page, https://unsloth.ai/docs/basics/qwen3-how-to-run-and-fine-tune: it names the recommended notebook, says to mix reasoning and non-reasoning data, and shows `enable_thinking=False` for non-thinking mode.
- The recommended notebook, "Qwen3 (14B) Reasoning + Conversational", `unslothai/notebooks@ebe88117` (`nb/Qwen3_(14B)-Reasoning-Conversational.ipynb`):
  - it loads with `FastLanguageModel.from_pretrained(max_seq_length=2048, load_in_4bit=True)` and `get_peft_model(...)`;
  - it formats with `tokenizer.apply_chat_template(conversations, tokenize=False)`;
  - it builds `SFTTrainer(args=SFTConfig(dataset_text_field="text", ...))`.
- The assistant-only step, from the Qwen3-4B-Thinking-2507 notebook (`unslothai/notebooks@0c6268b8`): `train_on_responses_only(trainer, instruction_part="<|im_start|>user\n", response_part="<|im_start|>assistant\n")`.
- `train_on_responses_only`'s docstring and unsloth-zoo's tests for it.
- TRL 1.15.0's SFT docs. Unsloth's recipe runs TRL's `SFTTrainer`.

## The recipe, and one choice that had to be made
- **No assistant-only recipe exists for the original Qwen3 models.** Unsloth's Qwen3 page shows none, and neither do the recommended notebook, the plain Qwen3 (14B) one, the 32B one or the 0.6B ExecuTorch one. All of them train on every token, prompt included.
- **`train_on_responses_only` appears only in the Qwen3-2507 notebooks and Qwen3.8.**
  - Thinking-2507 and Tool-Calling pass the explicit ChatML parts above.
  - Instruct-2507, QAT and Qwen3.8 call it with auto-detection.
- **The choice (logged in `docs/DECISIONS.md`):** the recommended notebook's loader, formatting and trainer, plus the explicit-parts call from the Thinking-2507 notebook. Qwen3-0.6B is a hybrid thinking model, closest to the Thinking notebook.
  - Probe: auto-detection picks the same parts and gives identical labels on 4 fixtures, so the choice between the two calls doesn't change the results.
- **Mode setting for the secondary table:** `enable_thinking=False` passed to `tokenizer.apply_chat_template` when formatting (Unsloth's Qwen3 page).
  - Not applied: the recommended notebook also appends " /no_think" to the user turns of its non-reasoning data. That is a data change, not a mode setting.

## What matters for the cells
1. **Template.**
   - *Docs:* the notebook uses the tokenizer's own template, with no `get_chat_template`.
   - *Probe:* Unsloth's CPU loader leaves Qwen3's template unchanged.
2. **An extra EOS on every row.**
   - *Probe:* every row is the template's render plus one appended `<|im_end|>` after the final `\n`. TRL's `SFTTrainer` appends the EOS to a `text` row that doesn't end with it ("Adding EOS to train dataset"; `trl/trainer/sft_trainer.py` L1284-1305 @ 1.15.0).
   - *Docs:* TRL's SFT docs only say to align `eos_token` with the chat template for Qwen; they never mention the append. Unsloth's notebook doesn't mention it either.
3. **Which tokens get loss.**
   - *Docs and code:* `train_on_responses_only` masks everything except the spans after each response marker, ending at message boundaries (unsloth-zoo's `test_train_on_responses_message_boundaries`).
   - *Probe:* the think block, the content and the turn's `<|im_end|>` are trained, including an empty turn's `<|im_end|>`.
4. **Padding.**
   - *Probe:* the collator becomes `DataCollatorForSeq2Seq`, padding on the right with `<|endoftext|>` and label -100.
5. **Truncation.**
   - *Probe:* TRL's `SFTConfig.max_length` truncates (TRL default `truncation_mode="keep_start"`). The loader's `max_seq_length` does not reach it on CPU: `args.max_length` stays 1024.

## Predicted cells (primary, thinking mode)

Because of point 2, the framework's tokens are the declared template's render plus one trailing `<|im_end|>`. How each check handles that:
- **Template parity fails:** an extra token, and no known-differences entry.
- **Assistant mask and truncation are inconclusive:** they first require the framework's row to equal the reference ("unaligned").
- **End of turn passes:** it judges each assistant turn's closing token, all of which come before the divergence at the very end. The probe shows them trained, including the empty turn's.
- **Special tokens is inconclusive:** its count rule sees one more `<|im_end|>` than the reference and fails, and that fail is blocked by template parity on the same fixture.
- **Multi-turn prefix is inconclusive:** every prefix encoding also ends in the extra `<|im_end|>`, so the framework's breaks differ from the template's. That fail is blocked by template parity.

| Check | single_turn | multi_turn | system_prompt | tool_call | reasoning | empty_assistant | long_answer | non_ascii |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| t1.template_parity | **fail** | **fail** | **fail** | **fail** | **fail** | **fail** | **fail** | **fail** |
| t1.assistant_mask | inconclusive | inconclusive | inconclusive | inconclusive | inconclusive | inconclusive | inconclusive | inconclusive |
| t1.end_of_turn | pass | pass | pass | pass | pass | pass | pass | pass |
| t1.special_tokens | inconclusive | inconclusive | inconclusive | inconclusive | inconclusive | inconclusive | inconclusive | inconclusive |
| t1.truncation | inconclusive | inconclusive | inconclusive | inconclusive | inconclusive | inconclusive | inconclusive | inconclusive |
| t1.multiturn_prefix | n/a | inconclusive | n/a | inconclusive | inconclusive | inconclusive | n/a | n/a |

Basis:
- template parity: probe on 4 fixtures, plus code;
- end of turn: probe on 4 fixtures, plus docs and code on the rest;
- the inconclusive cells: how our checks treat the extra token;
- not applicable: decided from the fixture alone.

## Expected labels underneath (not graded; for triage)

If the appended EOS were not there, docs, code and probe suggest:
- template parity, assistant mask and special tokens would pass on all 8, as end of turn already does;
- truncation would pass;
- multi-turn prefix would pass, with the template's own breaks.

In non-thinking mode (secondary), `train_on_responses_only` would still train the empty think block, because it can't see the mode and the rendered text is the same. So assistant mask and truncation would fail there, as TRL's cells did before it was told the mode. Here, telling Unsloth the mode can't change the labels.
