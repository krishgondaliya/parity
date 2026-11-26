# P-0004: On its CPU backend, Unsloth's Qwen3 recipe trains on rows with a second `<|im_end|>`; its GPU route doesn't add it

- Status: candidate (CPU backend only). Decided by the human, 2026-11-24: on the report card as Unsloth CPU-backend behavior, labeled CPU-only, low impact (`card_label` in `findings/root_causes.json`). Not shrunk. Revisit after the GPU run.
- Severity: low (CPU backend only)
- Date found: 2026-11-24
- Model: Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca
- Framework: unsloth@2026.11.10 on its CPU fallback, running trl 1.15.0   (lockfile: envs/unsloth-2026.11.10/uv.lock)
- Check: t1.template_parity, all 8 fixtures in thinking mode (primary) and 7 in non-thinking mode (secondary; `reasoning` is inconclusive there by design). 15 fails, one mechanism.   Kind: bug (as run)
- Verdict files: docs/evidence/2026-11-24-m3-unsloth/verdicts/{thinking,non_thinking}/t1.template_parity__*.json
- Predicted: yes, but from a probe run before the predictions, not from docs (`predictions.md` says so).

## Evidence
- Each of the 15 rows is exactly the declared template's render followed by one extra `<|im_end|>`. The first divergence is at the render's length; expected: nothing more; got: `<|im_end|>`. Checked on all 15.
- **Mechanism.** The recipe formats each conversation with `tokenizer.apply_chat_template(..., tokenize=False)`, and Qwen3's render ends `<|im_end|>\n`.
  - **CPU route.** `unsloth_zoo/cpu/trainer.py::_make_cpu_sft_trainer` installs a CPU SFT trainer that overrides `__init__` and `compute_loss` but not `_prepare_dataset`. So stock TRL's `_prepare_dataset` runs, and it appends the EOS to any `text` row that doesn't end with it (`trl/trainer/sft_trainer.py#L1284-L1306` @ 1.15.0). The row ends in `\n`, so a second `<|im_end|>` is added.
  - **GPU route.** `unsloth/models/rl_replacements.py::sft_trainer_prepare_dataset`, registered for `sft_trainer` and applied by `rl.py`'s `patch_functions` through `PatchFastRL` at import, replaces TRL's `_prepare_dataset` body with `unsloth_zoo.dataset_utils.sft_prepare_dataset`. The swap happens only when a signature regex matches. That function leaves chat-formatted text alone: `text.rstrip().endswith(<EOS or any special token>)`.
  - **Unsloth's stated intent:** unsloth-zoo PR #2558, "SFT: plain-text rows end in one EOS", merged 2026-11-16, says chat text ending in `<|im_end|>\n` is "unchanged. TRL's own check only looks at the EOS string, so it would add a second token after `<|im_end|>\n`; this does not." Its test covers only the zoo function, which the CPU backend never calls.
  - **GPU-route evidence (skeptic).** The GPU rewriter, pulled out and run on TRL 1.15.0's real `_prepare_dataset` source, matches its regex and installs the zoo body with no `add_eos` left. On the adapter's own trainers, its output equals the render on 16/16 fixture-and-mode cells; the CPU route gives the render plus exactly `<|im_end|>` on the same 16/16.
    - Caveat: neither the full GPU patch pipeline nor a real GPU run was exercised. `patch_trl_rl_trainers` only logs a warning on failure, so a failed GPU patch would quietly fall back to stock TRL and append too.
  - **Version drift:** unchanged in 2026.11.11 (the CPU trainer is the same; `_cpu_init.py` only adds DPO/KTO no-ops) and on unsloth-zoo main.
- **Does it get loss?**
  - With the assistant-only recipe (`train_on_responses_only`): no. The appended `<|im_end|>` and the `\n` before it are untrained. It is the last token, so it can't change any trained prediction.
  - With the recommended Qwen3 notebook as written (no `train_on_responses_only`, every token trained): yes. The model is trained to emit a second `<|im_end|>` after `<|im_end|>\n`. At inference, generation stops at the first `<|im_end|>`, so that position is never reached.
- Mode-independent: Qwen3's full render is identical in both modes, so the secondary cells show the same thing.

## Triage
- **Calibration green on the known-good model:** this run is the calibration model. `t1.template_parity` is green on Qwen3 through TRL 1.18.2 (KD-0001 only) and Axolotl 0.21.1, and on both stand-ins. So the check logic is not the suspect.
- **Adapter uses the documented public path:** yes for the recipe, no for the backend.
  - The adapter follows Unsloth's recommended Qwen3 notebook (`unslothai/notebooks@ebe88117`): `FastLanguageModel.from_pretrained`, `get_peft_model`, `apply_chat_template(tokenize=False)`, `SFTTrainer(dataset_text_field="text")`. It adds `train_on_responses_only` from the Qwen3-4B-Thinking-2507 notebook.
  - It runs on Unsloth's CPU backend, a deviation we chose (`envs/unsloth-2026.11.10/environment.json`). That backend is officially supported for SFT: the README says "We support … CPUs", the CPU startup message says "supported: text LoRA / full finetuning with SFTTrainer", and CPU SFT is actively maintained (unsloth-zoo #2234, #2519, #2560). So the behavior is Unsloth's own, on a backend it supports; it is not our error. These cells describe Unsloth's CPU backend, not its GPU recipe, and the report must say so.
  - The notebook pins `trl==0.22.2` and `transformers==4.57.6`; our environment has TRL 1.15.0 and transformers 5.17.0. TRL 0.22.2 has the same text-row EOS append, so the mechanism holds under the notebook's pins too (skeptic). Added to `environment.json`.
- **Fixture valid:** yes. The same fixtures pass template parity on TRL and Axolotl.
- **Reference consistent:** yes. The rows minus the extra token equal `apply_chat_template` with Qwen3's own template, and Unsloth's GPU-route preparation agrees with the reference.
- **Documented as intended:**
  - TRL: appending EOS to plain text is deliberate ("Add EOS token if needed: non-conversational only"). The Qwen double-EOS was reported as TRL #3318 (closed 2025-04-23 as completed); TRL's fix covers conversational data, not pre-rendered text. TRL's SFT docs don't mention the append.
  - Unsloth: its GPU-route code says chat-formatted text must be "left alone". Unsloth #3456 shows Unsloth knows TRL #3318 and keeps its own preparation patch on purpose. Its CPU backend doesn't apply that patch. Unsloth markets CPU training ("supported: text LoRA / full finetuning with SFTTrainer").
  - So: intended in TRL. In Unsloth, a CPU/GPU inconsistency against its own stated intent; no doc mentions the difference.
- **Duplicates:** searched Unsloth and TRL issues, read-only, for "double eos", "extra eos", "eos twice", "duplicate eos", "two eos tokens", "Adding EOS", "cpu fallback", "no GPU found training on CPU" and more.
  - Related: TRL #3318 (closed; same append, conversational path fixed), Unsloth #3456 (closed; Unsloth's preparation patch kept), Unsloth #3228 (closed; multiple EOS in a DPO notebook, a different path).
  - No report of the CPU backend skipping Unsloth's preparation.

## Consequences of the same extra token (not separate problems)
- The "blocked by t1.template_parity" inconclusives on `t1.special_tokens` and `t1.multiturn_prefix`.
- The summary's "tool-call arguments … mismatch (t1.multiturn_prefix on tool_call)". The tool_call row minus the appended token equals the verbatim render; the verdict shows `tokens_from_end` 32 for the framework against 31 for the template.

## Also seen (not a fail): now P-0005
- `t1.special_tokens`'s padding part is inconclusive on every fixture: asked to pad to 545, the batch came back at 538, the longest row.
  - `train_on_responses_only` replaces TRL's collator with a fresh `DataCollatorForSeq2Seq` and doesn't carry `SFTConfig.pad_to_multiple_of` over.
  - The adapter's live declaration already reports `pad_to_multiple_of=None`.
  - It affects padding length only, not labels.
  - The skeptic confirmed it: with `pad_to=64`, `objective="all"` keeps TRL's collator and pads to 128; `assistant_only` pads to 68.
  - The cause is zoo `train_on_responses_only` (`dataset_utils.py` ~L2450-2482): it keeps settings only for collators that pad through a processor.
  - The function is shared by both backends, so by code reading this isn't CPU-specific. Unchanged in 2026.11.11; no duplicate found.
  - Triaged as its own finding: `findings/P-0005/record.md` (the human's decision, 2026-11-24).

## Skeptic verdict (2026-11-24)
- **Verdict: real candidate, CPU backend only.** Not our error, not on both backends, not a duplicate, not fixed upstream.
- **Confidence:** high for the CPU mechanism, the absence of any adapter error, version drift and duplicates; medium for the GPU contrast (Unsloth's GPU rewriter was run, but not the full patch or a real GPU).
- **Checked:**
  - the repro;
  - the GPU swap path, and that nothing else on the GPU path appends an EOS (formatting_func, packing and IterableDataset all go through the zoo's guard);
  - default packing doesn't apply (`padding_free=False`, `max_steps=30`);
  - the notebook's own data ends `<|im_end|>\n` the same way;
  - official CPU support;
  - the loss claims on all 8 fixtures (generation stops at the first `<|im_end|>`: `generation_config.json` has `eos_token_id: [151645, 151643]`).
- **Corrections applied above:**
  - "a harness-side cause" was wrong: running on CPU was our choice, but the behavior is Unsloth's own on a supported backend;
  - PR #2558 added as the statement of intent;
  - mechanism citations corrected;
  - the GPU-route evidence is now 16/16 from the rewriter;
  - the version-drift line, the notebook pins, the downstream consequences and the TRL line range added.

## Decision (human, 2026-11-24)
- On the report card as Unsloth CPU-backend behavior, labeled CPU-only, low impact.
- Don't shrink the repro.
- Not a new report to TRL either. The underlying append is TRL's text-dataset behavior from TRL #3318: TRL fixed the conversational path, and appending EOS to plain text rows is intended.
- Revisit after the GPU run. Unsloth's GPU route becomes its primary result there, and these CPU results stay as a labeled secondary "CPU backend" table.

## Proposed handling (before the decision)
1. Present the 15 fails as Unsloth's CPU-backend behavior, labeled CPU-only, not as the GPU recipe's behavior. Don't suppress them.
2. As a framework observation: Unsloth's CPU backend doesn't apply its own SFT dataset preparation, so CPU and GPU runs train on different rows. Low severity: the extra token gets no loss under assistant-only, and is never reached at inference under all-token training. Possibly worth a short upstream note once confirmed on a GPU.
3. Not verifiable here: that a real GPU run produces the render. The evidence is Unsloth's GPU-route function run on CPU, not a GPU run.

## Repro
- `findings/P-0004/repro.py` (triage repro, not shrunk; CPU; tokenizer and config files only): exits 1 (the two routes differ), 3/3 on unsloth 2026.11.10, and 2/2 in the skeptic's runs.
- Not shrunk (the human's decision). The triage repro stays as the evidence.
- On a GPU: not tested (no GPU).

## GPU run (2026-11-25): CPU-only confirmed
- Unsloth's GPU route (Tesla T4, `envs/unsloth-2026.11.10/uv.lock`, the integrity checks are in `docs/evidence/2026-11-25-m3-unsloth-gpu/provenance.json`), the same adapter and recipe:
  - `t1.template_parity` passes on all 15 cells that run (8 thinking, 7 non-thinking): every row equals the declared template's render, with no second `<|im_end|>`;
  - the tool-call property is consistent there, where on CPU the extra token made it a mismatch.
- So the GPU route's dataset preparation leaves chat-formatted text alone, as its code and unsloth-zoo PR #2558 say. The GPU patch did apply: a failed patch would have fallen back to stock TRL and appended the token.
- The CPU fails stay linked here (15 as run; 8 shown since the non-thinking cells became `not_configurable`). P-0004's root cause carries `"backend": "cpu"`, so no GPU verdict can link to it.
- Decision (human, 2026-11-25): **card only for now.** No upstream note. Revisit after the kill test, once it's clear which Unsloth findings matter most.
- Since the same day's decision on P-0006, Unsloth's non-thinking cells are `not_configurable`. So P-0004 now explains the 8 thinking-mode template-parity fails. Its 7 non-thinking pins stay, and still match the as-run verdicts in git history.

## Upstream
- Issue URL (posted by human): none (card only)
- Maintainer response: none yet
- Fixed in: not yet
- Card only (the human's decision, 2026-11-25); revisit after the kill test. Not a TRL report either: TRL #3318 already covers the append, and TRL treats it as intended for text rows.
