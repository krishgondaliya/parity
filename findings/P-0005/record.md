# P-0005: Unsloth's `train_on_responses_only` silently drops `SFTConfig.pad_to_multiple_of`

- Status: candidate. Class: **config value silently ignored**. To be confirmed on the GPU run before any draft (the human's decision, 2026-11-24).
- Date found: 2026-11-24
- Model: Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca (the behavior is in the collator swap, not model-specific)
- Framework: unsloth@2026.11.10 (unsloth-zoo 2026.11.10), with trl 1.15.0, on the CPU backend   (lockfile: envs/unsloth-2026.11.10/uv.lock)
- Check: t1.special_tokens, padding part: inconclusive in 15 of 16 cells. In non-thinking `reasoning` it never ran, because our reference refuses there. Not a fail, so no verdict is pinned to it; the cells are also blocked by P-0004, which replaced their reason text. The "538 tokens, not pad_to=545" figure comes from re-running the adapter (`encode_batch` on single_turn + long_answer with pad_to=545: 538 with assistant_only, 545 with objective=all), not from a pinned verdict.
- Verdict files: docs/evidence/2026-11-24-m3-unsloth/verdicts/*/t1.special_tokens__*.json
- Predicted: no

## Evidence
- The adapter maps `TrainCfg.pad_to` to `SFTConfig(pad_to_multiple_of=...)`, TRL's documented option. TRL's own collator honors it: before `train_on_responses_only`, a batch asked to pad to 545 (or 64) is padded to a multiple of it.
- After `train_on_responses_only(trainer, ...)`, the recipe's assistant-only step:
  - the trainer's collator is a fresh `DataCollatorForSeq2Seq(pad_to_multiple_of=None)`;
  - batches are padded only to the longest row (538 instead of 545 in the check; 26 instead of 64 in the repro).
- **Silent:** nothing is printed or logged about the collator swap or the dropped setting. The skeptic checked at DEBUG for every logger, with warnings routed into logging, during a real training step.
- **Nothing puts it back:** in a real `trainer.train()` step (max_steps=1), the dataloader's collate function, the training step, the model forward and the embedding all see `(2, 26)`. Without `train_on_responses_only` they see `(2, 64)`.
- **Mechanism** (`unsloth_zoo/dataset_utils.py` @ 2026.11.10): `train_on_responses_only` rebuilds the collator as `DataCollatorForSeq2Seq(tokenizer=tokenizer, **_kept)` (L2482).
  - `_kept` copies `padding`, `max_length`, `pad_to_multiple_of` and `label_pad_token_id` only when the old collator pads through a processor (`_processor_backed`, L2446; `_same_class` and `_padding_class`, L2458-L2477).
  - TRL's `DataCollatorForLanguageModeling` doesn't, so its `pad_to_multiple_of` is dropped.
  - The code's own comment, about another collator class, calls this same effect a bug: an earlier version "dropped its settings and silently turned the run into a dynamically padded one" (L2463-L2464). The fix it describes covers only collators that pad through a processor.
- **Impact:** padding length only. Labels and attention masks stay right; only the batch width changes.
  - It matters to users who set `pad_to_multiple_of` for tensor-core alignment or fixed-shape compilation: they get dynamic padding without knowing.
  - It reaches further than TRL's setting (skeptic). User-supplied collators that pad through the tokenizer lose their settings too: `DataCollatorForLanguageModeling(pad_to_multiple_of=64)` and `DataCollatorForTokenClassification(padding="max_length", max_length=128)` both become dynamically padded.
  - Unsloth itself sets `pad_to_multiple_of` to the flex-attention block size when `UNSLOTH_ENABLE_FLEX_ATTENTION=1` (rl.py L3730-3738). A padded run would lose that too. Not verified.
  - Workaround (checked at collator level only): pass your own `DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=...)`. It is kept unchanged.
- **When it happens:** whenever batches are padded, i.e. packing and padding-free are both off. That is the CPU route's default, and what the recommended Qwen3 notebook sets (`padding_free = False`). On the GPU route, Unsloth's defaults may choose packing or padding-free on their own, and then the collator is never replaced and the setting survives. See "Backend".

## Triage
- **Calibration green on the known-good model:** yes for this part. The padding part of `t1.special_tokens` passes on Qwen3 through TRL 1.18.2 (8 of 8) and Axolotl 0.21.1 (7 of 8, plus one fail from a different root cause, P-0002), with `pad_to` honored. So the check logic is not the suspect.
- **Adapter uses the documented public path:** yes.
  - `pad_to_multiple_of` is TRL's documented `SFTConfig` field.
  - `padding_free=False` and `max_steps=30` are the recommended Qwen3 notebook's own settings (`unslothai/notebooks@ebe88117`, the SFTConfig cell: `padding_free = False, # Set to True if > 17 GB VRAM`).
    - The skeptic read them as an adapter deviation, because the Qwen3-4B-Thinking-2507 notebook (where the `train_on_responses_only` call comes from) leaves `padding_free` unset. They aren't: the adapter's SFTConfig follows the recommended notebook. But the two notebooks differ here, and on GPU that difference decides whether batches are padded.
  - `train_on_responses_only` is called after building the trainer, exactly as Unsloth's notebooks do.
  - The adapter's live declaration reads the swapped collator and reports `pad_to_multiple_of=None`, which agrees with the behavior.
- **Fixture valid:** yes. Any two rows of different lengths show it.
- **Reference:** not involved (padding only).
- **Documented as intended:** no, but the code makes a deliberate choice a maintainer may point to.
  - `train_on_responses_only`'s docstring (L860) says it masks instruction labels. Neither it nor any doc says the collator is replaced.
  - **The counter-argument, quoted:** for collators that don't pad through a processor, the code says "for any other class this is a replacement, not a swap, and its same-named attributes need not mean the same thing" (L2455-L2457). So not copying attributes is deliberate.
  - **The answer:**
    - the user's value also lives in `trainer.args.pad_to_multiple_of`, which has one meaning;
    - Unsloth's own GPU trainer patch reads `args.pad_to_multiple_of` every time it rebuilds a collator (rl.py L3367-3404), so keeping the setting is Unsloth's own convention;
    - the L2463-L2464 comment calls the same effect a bug, and collators that pad through the tokenizer still hit it (see Impact).
- **Duplicates:** searched issues and PRs in unslothai/unsloth and unslothai/unsloth-zoo, read-only, for "train_on_responses_only pad_to_multiple_of", "train_on_responses_only collator", "pad_to_multiple_of ignored" and "DataCollatorForSeq2Seq train_on_responses_only". None found; the skeptics searched further and found none either.
  - Related: unsloth-zoo PR #1279 (merged 2026-09-21) fixed the padding-free case of this same replacement and kept "replacement of unrelated collators" as before; unsloth-zoo PR #836 says text collators are "replaced exactly as before"; unsloth #2703 shows a user passing `DataCollatorForLanguageModeling(pad_to_multiple_of=8)` before `train_on_responses_only`, hitting this without noticing (that issue is about something else).
  - Not fixed: unchanged in 2026.11.11 and on unsloth-zoo main at `8420b55e8fd6` (the block is now at L2559-2583).
- **Backend:** `train_on_responses_only` is the same shared function on both backends, so with padded batches the drop happens on GPU too. But it only happens when batches are padded.
  - On GPU, the Unsloth SFT config defaults `packing=None` and `padding_free=None` (rl.py L3574, L3618), and a GPU-only `SFTTrainer.__init__` patch then chooses: packing when there is no `max_steps`, padding-free when there is, unless its CUDA cost model picks padded batches.
  - With packing or padding-free, the collator is never replaced and the setting survives.
  - So on GPU it happens when the user turns padding-free off (as the recommended Qwen3 notebook does) or when Unsloth's own choice is padded batches.
  - For the GPU run: the repro now sets `packing=False, padding_free=False` explicitly, so the check can't come back a false negative. Also run once with the notebook-default `padding_free=None` to measure how far it reaches in practice.
  - Not yet run on a GPU, which is the confirmation the human asked for.

## Skeptic verdict (2026-11-24)
- **Verdict: real candidate.** Confidence: high that the behavior is real, framework-side and silent; medium on how much it matters to users, since on the GPU route it is narrower than first written.
- **Checked:**
  - the repro, 3/3;
  - a real `trainer.train()` step;
  - DEBUG logging for every logger;
  - the docs and the notebook;
  - the adapter's own `encode_batch` (`all` 545 vs `assistant_only` 538);
  - user-supplied collators;
  - packing and padding-free on CPU (setting kept);
  - unsloth-zoo main;
  - duplicates and related PRs;
  - the GPU route, by code reading.
- **Corrections applied above:**
  - the Check line (15 of 16; the 538/545 figure's source);
  - the counter-argument at L2455-L2457, quoted and answered;
  - the L2463-L2464 range;
  - the backend scope (padded batches only), and the repro's explicit `packing=False, padding_free=False`;
  - the wider impact and the workaround;
  - the silence check, the related PRs and the not-fixed-on-main line;
  - the calibration counts.
- **One correction not applied:** the skeptic called the adapter's `padding_free=False` a deviation from the notebook. It is the recommended notebook's own setting (see Triage, Adapter). The difference from the Thinking-2507 notebook is recorded instead.

## Repro
- `findings/P-0005/repro.py` (triage repro, not shrunk; CPU; tokenizer and config files only; padded batches set explicitly): exits 1 (the setting is dropped), 3/3 on unsloth 2026.11.10, before and after the `packing=False, padding_free=False` change.
- On a GPU: not tested. Required before any draft.

## Upstream
- None. No draft until the GPU run confirms it (the human's decision).
- For a future draft (skeptic): the title should say "silently drops" or "ignores", not "bug", and should name the padded-batch condition.
