# P-0006: In non-thinking mode, Unsloth's assistant-only recipe trains the last turn's empty think block, which our reference puts on the prompt side

- Status: **exploratory** (non-thinking table only; not a candidate). Decided by the human, 2026-11-25: the non-thinking table is exploratory. This record exists so the 14 fails are triaged and linked (Gate A, A4), not to put anything on Unsloth.
- Date found: 2026-11-25 (predicted before the run: `docs/evidence/2026-11-25-m3-unsloth-gpu/predictions.md`, `290eff2`)
- Model: Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca
- Framework: unsloth@2026.11.10 on its GPU route (Tesla T4), running trl 1.15.0   (lockfile: envs/unsloth-2026.11.10/uv.lock)
- Check: t1.assistant_mask and t1.truncation, 7 fixtures each (all but `reasoning`, which our reference refuses in non-thinking mode), non-thinking (secondary) only. 14 fails, one mechanism. The truncation fails restate the mask fails: at these limits nothing is cut, and the check judges zones only (`truncation.py` L208-212).
- Kind: `bug` is only the engine's label for a fail with no known-differences entry. This record makes no claim that Unsloth has a bug; the skeptic calls it a difference of serving convention (below).
- Verdict files: docs/evidence/2026-11-25-m3-unsloth-gpu/verdicts/non_thinking/t1.{assistant_mask,truncation}__*.json
- Predicted: yes, from Unsloth's docs alone (all 14 cells).

## Evidence
- Every one of the 14 fails has exactly four violations and nothing else:
  - the tokens are `<think>`, `\n\n`, `</think>`, `\n\n`, contiguous, labels equal to the tokens, all in the reference's `must_not_train` zone;
  - they start at the first token after the last `<|im_start|>assistant\n` (checked against the render: positions 13, 23, 30, 31, 36, 53 and 251 by fixture);
  - no must-train token lacks loss, and no label is wrong;
  - every self-test caught its planted faults.
- **The mechanism:**
  - `train_on_responses_only(instruction_part="<|im_start|>user\n", response_part="<|im_start|>assistant\n")` trains every token after each response marker, up to the message boundary. Its parts are plain strings, so it can't see the serving mode.
  - Qwen3's template renders the full conversation identically in both modes (checked on all 8 fixtures). `enable_thinking` only changes the generation prompt.
  - So Unsloth's labels are the same in both modes. The mode setting reached it (the declaration records `{'enable_thinking': False} passed to tokenizer.apply_chat_template`), but on training text, with no generation prompt, it changes nothing. Its only citation is an inference example.
- **Why only the non-thinking table fails:**
  - In non-thinking mode, the generation prompt ends with the empty think block (`<think>\n\n</think>\n\n`). So our reference treats it as a pre-fill, on the prompt side, must not train (`src/parity/serving.py`; `docs/DECISIONS.md`, 2026-11-24).
  - In thinking mode, the same four tokens are model output, must train. All 8 thinking-mode cells of both checks pass on the same labels.
- **Impact, if the reference's zone is right:**
  - a model trained this way on non-thinking data learns to emit an empty think block after the assistant header;
  - at serving time in non-thinking mode the template pre-fills that block, so the model is never asked to produce it;
  - the trained-but-pre-filled tokens are teacher-forced and harmless to stopping. Whether they matter is a Tier 3 (train/serve) question, not a Tier 1 one.

## Triage (from /triage)
- **Calibration green on the known-good model:** yes. These checks pass in non-thinking mode on Qwen3 through TRL 1.18.2 (7 of 7, with `reasoning` inconclusive) and Axolotl 0.21.1 (6 of 7, plus `empty_assistant` from P-0002). So the check logic is not the suspect.
- **Adapter uses the documented public path:** yes.
  - It uses `train_on_responses_only` with the explicit ChatML parts from the Qwen3-4B-Thinking-2507 notebook (`unslothai/notebooks@0c6268b8`).
  - The mode reaches Unsloth as `enable_thinking=False` in the formatting `apply_chat_template`, as Unsloth's Qwen3 page shows. Unsloth documents no other mode setting for training data.
  - The recommended notebook also appends " /no_think" to non-reasoning user turns. That is a data change, not a mode setting, and the adapter doesn't apply it. It wouldn't change the labels either: the response parts are the same.
- **Fixture valid:** yes. The same 7 fixtures pass on TRL and Axolotl in this mode.
- **Reference consistent:** yes, and it models one of two documented conventions.
  - The template facts are consistent: the generation prompt pre-fills the block in non-thinking mode, and history turns render without it (the `prefill_missing_from_history` property in the summary).
  - **Hard switch** (`enable_thinking=False`), which our non-thinking reference models.
    - Qwen3-0.6B's model card (README @c1899de, the note after the soft-switch example) says that with `enable_thinking=False` the model "will not include a `<think>...</think>` block": the block is pre-filled, not generated.
    - TRL makes the same choice on purpose: its Qwen3 training template "emitted the part of the thinking block that the generation prompt pre-fills before {% generation %}, so that assistant-only loss doesn't train it" (`trl/chat_templates/qwen3_training.jinja`, header comment).
  - **Soft switch** (`/no_think` in the user turn, with thinking on): the model itself outputs an empty block (the same model-card note). That is how Unsloth documents non-thinking training.
- **Documented as intended:** yes, for the soft switch (correction from the skeptic).
  - The recommended notebook (`notebooks@ebe88117`, cells 19-20) appends `/no_think` to every non-reasoning user turn. It says: "The chat template gives these examples an empty `<think></think>` block, so without the switch the model learns to skip thinking even when `enable_thinking = True`".
  - So in Unsloth's documented recipe, the empty block is model output the model should learn to emit. That is our thinking-mode zone, where all 8 cells pass.
  - Unsloth's completions-only docs train "assistant messages" whole (`unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide`), and the Thinking-2507 notebook's saved labels start at `<think>\n`.
  - Unsloth documents no way to prepare training data for the hard switch. Under the 2026-11-24 rule (a framework that documents no way to set the mode gets `not_configurable` cells), these cells arguably shouldn't be judged at all. That is a question for the human; see "Open".
- **Duplicates:** searched unslothai/unsloth and unslothai/unsloth-zoo, read-only, for "train_on_responses_only enable_thinking", "train_on_responses_only think", "enable_thinking False empty think", "non-thinking think tags trained", "train_on_responses_only qwen3 think block", "mask thinking", "empty think", "enable_thinking", "no_think train_on_responses_only" and "thinking tags loss". No duplicate. Related:
  - #6695 (closed as completed): a helper to mask `</think>` in truncated reasoning traces, as Nemotron Ultra did. It shipped as `train_on_responses_only(mask_out_tokens=...)` (`unsloth_zoo/dataset_utils.py` L869-879). It can't express this zone: it masks the given tags everywhere in kept spans, and leaves the `\n\n` tokens alone.
  - #4158 (closed as completed): the vision collator ignored chat-template arguments, which caused a train/infer think-mode mismatch on Qwen3.5. Closest in kind.
  - #3005 (closed): Qwen3 loses its reasoning after training on the recommended notebook. That notebook never calls `train_on_responses_only`: the same data effect, by another code path.

## Skeptic verdict (2026-11-25)
- **Verdict: partly holds.** The mechanism holds (high confidence). The framing doesn't: this is intended behavior under a different non-thinking convention (medium confidence on how to classify it).
- **Checked:**
  - all 14 verdicts: exactly the 4 tokens, zeros elsewhere, self-tests caught;
  - per fixture, the supervised-token count is identical in both modes, and the must-train count drops by exactly 4;
  - the declaration (`enable_thinking` acts only inside `add_generation_prompt`);
  - the calibration counts;
  - the predictions (14 of 14);
  - duplicates.
- **Corrections, all applied above and each verified:**
  - the notebook's soft-switch cells, so "no doc" was wrong;
  - Qwen's model card and TRL's training template, so "nothing in Qwen's docs" was wrong;
  - the mode setting has no effect on training text;
  - #6695 shipped as `mask_out_tokens`;
  - #3005 is another code path; #4158 added;
  - the truncation fails restate the mask fails.
  - Also: the summary's secondary-table header still said "the same encoded rows… No extra framework runs", stale since 2026-11-24. Fixed in `src/parity/engine/report.py`, and every summary re-rendered.
- **The skeptic's proposal:** reclassify the 14 cells as a known difference, or as `not_configurable` (Unsloth documents no hard-switch training), not `bug`. Not done: it changes how the harness models Unsloth's non-thinking mode, and the human decided the table is exploratory.

## Open (for the human)
- How should the non-thinking table judge a framework whose documented non-thinking training is the soft switch?
  - (a) `not_configurable` for Unsloth, since it documents no hard-switch training;
  - (b) a known-differences entry for this zone signature;
  - (c) a second, soft-switch reference (`/no_think` in the user turn).
- Until then P-0006 stays exploratory, pinned, and never counted against Unsloth.

## Repro
- No separate repro: the 14 verdicts are the evidence, and `docs/evidence/2026-11-25-m3-unsloth-gpu/ingest.py --relink` re-renders them.
- Not shrunk, and not to be (exploratory).

## Upstream
- None. Exploratory, and the open question is ours (the pre-fill zone), not Unsloth's.
