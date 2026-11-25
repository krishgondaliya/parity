# P-0005: Unsloth's `train_on_responses_only` silently drops `SFTConfig.pad_to_multiple_of`

- Status: **candidate, posted 2026-11-25**: https://github.com/unslothai/unsloth/issues/19567, by the human. Reproduced on GPU (a Tesla T4; rule 4 cleared); repro shrunk. Class: **config value silently ignored**. "Candidate" until a maintainer confirms (rule 8).
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
  - Workaround: **not verified** (correction, 2026-11-25). The first triage said a user-supplied `DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=...)` is kept. Re-checked in the fresh pinned environment, calling the collator on the dataset's rows the way the repro does, it crashes: the rows still carry the raw `text` column. A real training step may drop that column first; not checked. The issue suggests no workaround.
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

## GPU run (2026-11-25, the first trip): consistent with P-0005; rule 4 still open then (settled by the second trip, below)
- **What the GPU run showed** (Tesla T4, Unsloth's CUDA route, `docs/evidence/2026-11-25-m3-unsloth-gpu/`):
  - `t1.special_tokens` is inconclusive in all 15 cells where its reference runs (8 thinking, 7 non-thinking):
    - part 1 (the special tokens themselves) passes;
    - part 2 (padding) saw a batch 537 wide when asked for `pad_to`=545: "the padded row has 537 tokens, not pad_to=545; check the adapter's mapping of pad_to";
    - so the `pad_is_closing_token_masked` plant had no padded batch to show in, and the self-test reads "missed".
  - The predictions (docs alone) said pass, and named these cells in advance as the ones where this record's code reading disagrees. These are the run's only 15 misses (81 of 96).
  - The adapter's live declaration for that config, on GPU: `DataCollatorForSeq2Seq(pad_to_multiple_of=None, label_pad_token_id=-100)`, the collator `train_on_responses_only` installs. That's the same as on CPU.
- **Rule 4, our adapter's `pad_to` mapping first.** What rules it out, and what doesn't yet:
  - Code: the adapter passes `TrainCfg.pad_to` straight to `SFTConfig(pad_to_multiple_of=...)`, with no backend branch. The GPU declaration's config shows `pad_to=545` reached the adapter. That is Parity's `TrainCfg`, not `SFTConfig` (the skeptic's correction).
  - Code, Unsloth's GPU route (skeptic, read-only):
    - its `SFTConfig` replacements leave the field alone (`unsloth/models/rl.py` L3549-3575; L3730-3738 replaces only None/'auto', and only with flex attention on);
    - TRL builds `DataCollatorForLanguageModeling(pad_to_multiple_of=args.pad_to_multiple_of)` (`trl/trainer/sft_trainer.py` L975-979);
    - Unsloth's collator rebuild does nothing when no collator is passed (`rl.py` L3354-3384);
    - the zoo's swap keeps nothing from TRL's collator (`dataset_utils.py` L2450-2482).
  - Behavior on CPU, same adapter code: objective `all` pads to the requested width, and `assistant_only` doesn't. The skeptic saw 545 vs 538 in this record's first triage; `findings/P-0005/adapter_check.py` shows 546 vs 538 on CPU, with `trainer.args.pad_to_multiple_of` holding the value in both cases.
  - **Not shown on GPU:** that `trainer.args.pad_to_multiple_of` and the collator before the swap held 545 there. The GPU route replaces `SFTConfig` and `SFTTrainer` with Unsloth's patched versions, and the verdicts recorded only the final collator.
  - So the drop happens after our adapter on CPU, and is consistent with that on GPU, but not shown there. These 15 cells are not attributed to P-0005 until `adapter_check.py` runs on GPU. The adapter's declaration now also records `trainer.args.pad_to_multiple_of`, so future verdicts carry that fact.
- **The optional repro didn't run** (harness error H-10, `findings/LEDGER.md`). Both GPU runs crashed at `SFTTrainer` with "You cannot perform fine-tuning on purely quantized models": the repro attached no LoRA. Fixed: `get_peft_model` before the trainer, as the recommended notebook does. On CPU the results are unchanged (width 64 to 26, exit 1, for both variants).
- **The `padding_side='left'` the GPU declaration showed was ours** (harness error H-11, found by the skeptic; `findings/LEDGER.md`):
  - Unsloth's patched `SFTTrainer.__init__` ends with `model.for_inference()` (`rl.py` L754-757), which sets the tokenizer's padding side to left (`llama.py` L6026-6030).
  - `train()` is wrapped (`rl.py` `prepare_for_training_mode`) to call `model.for_training(...)` first, which sets it back to right.
  - The adapter collated in between, in a state training never uses. It now makes the same `for_training` call first (`_enter_training_mode`).
  - On CPU the re-run is identical: 96 of 96 verdicts unchanged.
  - No verdict was affected: the check accepts padding at either end, and the padding part never got that far.
- **Next GPU trip:** the kit's `RUN = "p0005"`, a few minutes. It runs:
  - the fixed repro, padded;
  - the repro with `padding_free` left to Unsloth;
  - the repro with one real `trainer.train()` step (`--train-step`): the batch training actually draws, its width and padding side;
  - `adapter_check.py`. Expected if P-0005 holds: objective `all` 545 and `assistant_only` 537, with `trainer.args.pad_to_multiple_of=545` in both.
  - Then the 15 cells can be attributed.

## Skeptic verdict on the GPU run (2026-11-25)
- **Verdict: partly holds.** P-0005 is the likely cause, but it isn't shown on GPU. Confidence medium-high, from code reading; nothing on GPU has measured it.
- **What the evidence does show:** the final collator, and a width equal to the longest row (537).
- **What it doesn't:** that fits a dropped setting, and it equally fits a value that never reached `SFTConfig`.
- **The minimal measurement:** `adapter_check.py` on GPU, plus one real training step with a recording collator. Both are in the P-0005-only mode.
- **Its other finding:** H-11, above.

## Confirmed on GPU (2026-11-25, the P-0005-only trip; `docs/evidence/2026-11-25-p0005-gpu/`)
- **The run:** Colab Tesla T4 (driver 580.82.07, CUDA 13.0), Unsloth on `cuda`, `envs/unsloth-2026.11.10/uv.lock`, kit commit `25c3dc8`.
  - Verified by its ingest script: 12 files against their manifests, `git archive` of the commit rebuilt here with the recorded sha256, every step passed, every installed version as locked.
  - All four runs completed.
- **Rule 4 cleared, our adapter's mapping:** `adapter_check.py`, Parity's own adapter at the check's `pad_to` (545):
  - objective `all` (no `train_on_responses_only`) pads to 545;
  - `assistant_only` pads to 537, with the collator `DataCollatorForSeq2Seq(pad_to_multiple_of=None)`;
  - `trainer.args.pad_to_multiple_of=545` in both;
  - so the value reaches the trainer, and the swap drops it. Exit 0.
- **The recipe's padded path:** `packing=False, padding_free=False`, as the recommended Qwen3 notebook sets.
  - The collator goes from `DataCollatorForLanguageModeling(pad_to_multiple_of=64)`, batch width 64, to `DataCollatorForSeq2Seq(pad_to_multiple_of=None)`, width 25.
  - **In real training:** one `trainer.train()` step draws a batch 25 wide, padded on the right. The drop reaches training. (After construction and before `train()`, the tokenizer pads on the left: Unsloth's inference state, H-11.)
- **Unsloth's defaults** (`padding_free` unset, `max_steps=60`): Unsloth chose `padding_free=True`. The collator was not replaced, and the batch kept the setting (width 64, exit 0).
  - So the scope is the explicit padded path the recommended notebook uses.
- **The 15 inconclusive `t1.special_tokens` cells of the GPU run** (`docs/evidence/2026-11-25-m3-unsloth-gpu/`) are explained by P-0005 (537, not 545; rule 4 cleared above).
  - Root-cause pins link fail verdicts only, so no pin is added; this is the link.
  - Since 2026-11-25 the check judges a dropped width as its own fail (rule 2a, `pad_to_honored`) and pad handling separately (rule 2b). The next full GPU run's width fails get pinned here after /triage.

## Cells linked to this root cause (2026-11-25, M4 calibration GPU run, `docs/evidence/2026-11-25-m4-calibration/unsloth/`)
- **The run:** Colab Tesla T4 (driver 580.82.07), Unsloth on `cuda`, torch 2.14.1+cu130, `envs/unsloth-2026.11.10/uv.lock`, kit commit `b73da05`, RUN = "full". Verified by `report.py --unsloth-zip`: 107 files against their manifests, `git archive` rebuilt here with the kit's sha256 (`e7e6c894…`), all 96 verdicts unsloth@2026.11.10 on the GPU backend with the run's lockfile.
- **8 fail verdicts, pinned in `findings/root_causes.json`** (new P-0005 entry, backend `gpu`): `t1.special_tokens` on all 8 fixtures, thinking mode. Non-thinking mode is not configurable for Unsloth (P-0006), so it has no cells here.
- **/triage, cell by cell, before pinning:**
  - **The one failing rule is `pad_to_honored`** (rule 2a): asked for 545, the batch is 537 wide, which is the longer row's reference length (`pad_to` = longer row + 8). That is the batch cut back to its longest row.
  - **Everything else passes:** `special_tokens` and `pad_handling` (rule 2b, both rows judged at the produced width), and all 4 planted faults are caught (including `pad_to_ignored`, P-0005's own mechanism).
  - **Rule 4, from each verdict's own declaration:** the batch call (pad_to=545) records `trainer.args.pad_to_multiple_of=545`. Our adapter's value reaches the trainer. The collator that `train_on_responses_only` installed records `DataCollatorForSeq2Seq(pad_to_multiple_of=None)`, on the padded path (`packing=False, padding_free=False`). So the drop is after our adapter, as the P-0005-only trip showed.
  - **Check logic:** `padding_width` passes on the same fixtures through TRL 1.18.2 and Axolotl 0.21.1 (M4 calibration), so the rule itself is not the suspect.
- **Calibration:** P-0005 explains these 8 fails and nothing else. They are allowed under the green definition (a fail linked to a triaged candidate) and listed in `runs/calibration-2026-11-25.md`.

## Labels not affected; loss the same up to float rounding (verified on CPU, 2026-11-25)
- `findings/P-0005/impact.py`, run 8 times in the fresh pinned environment: 3 by me, 5 by the issue reviewer. It compares, on the same two rows with the model in training mode, the swapped collator's batch (width 26) with the same collator class keeping `pad_to_multiple_of=64` (width 64).
- Results:
  - the real tokens are identical, and so are the labels on them;
  - every pad position is unsupervised;
  - 18 supervised tokens in both;
  - the loss is the same up to float rounding: identical in 6 of 8 runs, one float32 ulp (9.5e-7) apart in 2. (Correction from the issue reviewer: "bit for bit" was wrong, since my 3 runs happened to match.)
- Only the tensor width changes. Not measured on GPU, where attention kernels at different widths may differ in floating point.

## Repro
- **Shrunk:** `findings/P-0005/repro.py`, 30 lines, no Parity code. It runs the recommended notebook's path (`FastLanguageModel.from_pretrained`, `get_peft_model`, `SFTTrainer`, `train_on_responses_only`) on a tiny random model with Qwen3-0.6B's tokenizer and config.
  - It prints the collator and batch width before and after, and exits 1 if the width isn't a multiple of 64.
  - Pinned environment: `findings/P-0005/requirements.txt` (the lock's exact versions, without Parity) and `findings/P-0005/uv.lock`.
  - How to run it: `findings/P-0005/HOW-TO-RUN.md` (CPU only, with the telemetry opt-outs). Those exact commands were run from a scratch folder on 2026-11-25: exit 1 on 2026.11.10, on 2026.11.12 and on unsloth-zoo `cad81ef2`.
- **Fresh environment** (`uv venv --python 3.12` + `uv pip install -r requirements.txt`, Parity not importable): fails 3/3 (64 to 26, exit 1). unsloth and unsloth-zoo 2026.11.10, trl 1.15.0, transformers 5.17.0, torch 2.14.1.
- **Newest release** (unsloth and unsloth-zoo 2026.11.12, fresh venv): fails 3/3, same output.
- **unsloth-zoo main** at `cad81ef2dc55` (installed from git, checked via `direct_url.json`; fresh venv): fails 3/3. The collator-swap block is byte-identical to 2026.11.10, now at L2547-L2583.
- On GPU: the shrunk script wasn't run there. The GPU evidence is the triage repro's, the same mechanism and calls.
- **Triage repro**, kept as `findings/P-0005/triage_repro.py` (the kit runs it): padded, `--padding-free-unset`, `--train-step`.
- 2026-11-25: added `--padding-free-unset` (opt-in; without it the repro behaves as before, checked on CPU: width 64, then 26, exit 1).
  - It leaves packing and padding_free to Unsloth's defaults, with `max_steps=60` as the Thinking-2507 notebook sets.
  - It prints a `batching:` line showing what Unsloth chose. Under packing or padding-free the exit code is not the verdict; read the lines.
  - On CPU, Unsloth's defaults choose padded batches too, so the result is the same (exit 1).
  - The GPU kit's optional cell 6 runs both variants and puts the output in the results zip's own `P-0005/` folder (`docs/gpu-run/GPU-RUN.md`).
  - The kit counts a run as complete only if it printed its "after" line, since a crash also exits 1.

## Upstream
- **Posted 2026-11-25** (by the human): https://github.com/unslothai/unsloth/issues/19567. It was open when recorded (created 2026-11-25T20:15:48Z, checked read-only).
  - **Repo:** unslothai/unsloth, with that repo's "Bug / Issue" template, not unsloth-zoo where the code lives. unsloth-zoo#698 ("[Notice] Please use main Unsloth repo for filing issues", open) asks for issues to go to the main repo, which the maintainers monitor more closely.
  - **Title as posted:** "[Bug] `SFTConfig.pad_to_multiple_of` is silently dropped with padded batches (unsloth-zoo 2026.11.12 and main)". It starts with the template's default "[Bug]", where the draft had "[train_on_responses_only]".
  - **Maintainer response:** none yet.
  - **Fixed in:** not yet.
- **Drafted first:** `findings/P-0005/issue.md`, written for https://github.com/unslothai/unsloth-zoo/issues. The code is `unsloth_zoo/dataset_utils.py`; `unsloth.chat_templates` re-exports it.
  - Uses unsloth-zoo's "Bug / Issue" template, with its six questions answered. Its default "[Bug]" title is replaced (rule 8).
  - States the scope, says labels are not affected and the loss is the same up to float rounding (verified on CPU), and ends with a one-line offer to send a fix with a test.
- Duplicates searched again, read-only, in both repos, issues and PRs: "train_on_responses_only pad_to_multiple_of", "pad_to_multiple_of", "train_on_responses_only collator", "DataCollatorForSeq2Seq train_on_responses_only", "train_on_responses_only padding" and "pad_to_multiple_of ignored". No duplicate. #2703 stays related.
- **Issue reviewer (2026-11-25): not ready as first drafted; every fix applied.**
  - Three claims were wrong:
    - "the same loss bit for bit": it differs by 1 ulp in 2 of its 5 runs;
    - "settings are kept only for collators that pad through the tokenizer": `_kept` is filled only for a pad-delegating collator holding a processor with no `.pad` (`_pads_through_a_processor`);
    - the inlined script was presented as run on GPU, but only the triage variant ran there.
  - Also applied: the title names 2026.11.12 and main; the scope names `max_steps=60` and the CPU route; notebook and comment permalinks; the template's six questions verbatim; Python 3.12.13; "Nothing is printed about the swap".
  - Confirmed by the reviewer: the repro exits 1 in the pinned venv, on 2026.11.12 and on main; the permalinks; the target repo (unsloth-zoo, which handles `train_on_responses_only` issues, e.g. #2363); no duplicate.
