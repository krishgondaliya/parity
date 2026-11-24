# P-0003: In non-thinking mode (secondary), TRL's default encoding trains Qwen3's empty think block, which that mode treats as prompt (TRL is never told the mode)

- Status: **ours: evaluation design** (skeptic: high confidence). Not a TRL difference and not an upstream issue: TRL labels correctly for whatever `enable_thinking` it is given, and Parity never gives it the mode. The human decides the fix (see "Proposed handling").
- Date found: 2026-11-24
- Model: Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca
- Framework: trl@1.18.2   (lockfile: envs/trl-1.18.2/uv.lock)
- Check: t1.assistant_mask (7 fixtures), t1.truncation (7 fixtures), **secondary results only (non-thinking mode)**   Kind: bug (as run; no registry entry)
- Verdict files: docs/evidence/2026-11-24-m3-rerun/trl/verdicts/non_thinking/t1.assistant_mask__*.json, t1.truncation__*.json
- Predicted: no. This is the first run with a secondary mode.

## Evidence
- Every one of the 14 fails is the same tokens: each assistant turn's empty think block (`<think>`, `\n\n`, `</think>`, `\n\n`) is trained, and the non-thinking reference says it is prompt side.
  - Example, `single_turn`: positions 13-16 trained.
  - `multi_turn`: positions 15-18, 35-38 and 61-64, one block per assistant turn. The evidence list shows the first 8.
- Nothing else differs: no must-train token untrained, no label mismatch, and the rows align token for token with TRL's declared template (`qwen3_training.jinja`) rendered in non-thinking mode.
- In thinking mode (primary), the same rows pass: TRL's only primary differences are KD-0001.
- `reasoning` is inconclusive in non-thinking mode on every check (TRL and Axolotl, this re-run), because our reference refuses: the prefix diff fails when a last turn that reasons meets a generation prompt that closes the think block empty. TRL's own template does give it a non-thinking reading: with `enable_thinking=False`, the whole reasoning trace falls outside the loss (94 tokens; skeptic).

## Triage
- **Our side first (rule 4).** The secondary results judge the *same encoded rows* against a non-thinking reference. That is the human's decision of 2026-11-24 ("same encoded rows, a second reference, no extra framework runs"). The adapter therefore never tells TRL the mode, and TRL encodes with its default chat-template arguments, i.e. thinking mode. So the mismatch comes from our evaluation design: TRL was never asked for non-thinking labels.
- **Adapter:** uses TRL's documented assistant-only path (`SFTConfig(assistant_only_loss=True)`). Unchanged since M3.
- **Fixture:** valid; the same 7 fixtures pass in thinking mode.
- **Reference:** the non-thinking reference on TRL's own template gives the block to the prompt, because the template's non-thinking generation prompt pre-fills it. TRL's template agrees, if asked:
  - `qwen3_training.jinja` @ v1.18.2, L53-60: with `enable_thinking` false, the block is emitted before `{% generation %}`, so it is not trained. Otherwise it is inside, so it is trained.
  - Its header comment (L7-8) says so: "Emitted the part of the thinking block that the generation prompt pre-fills before {% generation %}, so that assistant-only loss doesn't train it."
- **Documented, tested and intended.**
  - A user who serves in non-thinking mode passes `enable_thinking: false` per example through a `chat_template_kwargs` column. `tokenize_fn` merges it into `apply_chat_template` (`trl/trainer/sft_trainer.py#L1360-L1366` @ v1.18.2), and TRL tests the column (`test_train_with_chat_template_kwargs`, `tests/test_sft_trainer.py#L1366` @ v1.18.2).
  - TRL documents the behavior in `docs/source/chat_templates.md` L115 @ v1.18.2: "The part of the thinking block that the generation prompt pre-fills (`<think>\n\n</think>\n\n` with `enable_thinking=False`) stays outside the markers, since the prompt already ends with it."
  - TRL tests it too (`test_assistant_masks_exclude_generation_prompt`, `tests/test_chat_template_utils.py#L1813`, over `enable_thinking` None/False/True, Qwen3 included).
  - **What is undocumented is narrower:** how to pass chat-template arguments to SFT. `sft_trainer.md` and `dataset_formats.md` never mention the per-row column, and `SFTConfig` has no `chat_template_kwargs` field: it raises `TypeError` (checked), unlike `GRPOConfig` (`grpo_trainer.md` L813). At most a docs or feature request, not a finding.
- **Run evidence (skeptic):**
  - With the column set to `{"enable_thinking": false}` on every row, through TRL's public path, all 14 cells pass in non-thinking mode, with every planted fault caught. The mirror image happens in thinking mode, which confirms the mode is the whole story.
  - One real `SFTTrainer.train()` step works with the column.
- **Duplicates and upstream history:** this exact class was reported and fixed upstream.
  - #8071 (closed 2026-11-06), "SFT with assistant_only_loss trains the pre-filled thinking block on Qwen3 checkpoints", was fixed by PR #8072 (merged 2026-11-06). The template lines cited above are that fix.
  - #8109 (open) tracks TRL's rule that the mask must start where the generation prompt ends "for every value of the flags the generation prompt branches on".
  - v1.18.2 is still the latest release.

## Skeptic verdict (2026-11-24)
- **Verdict: our error, a design artifact of the evaluation. TRL's behavior is intended, documented and tested, and is itself an upstream fix. Confidence: high.**
- Re-checked independently:
  - All 14 cells violate only on exactly the empty-think-block runs `[<think>, \n\n, </think>, \n\n]`, with no other violation (full lists, not the capped evidence).
  - TRL's own generation markers rendered with `enable_thinking=False` equal our non-thinking zones.
  - The per-row column makes all 14 cells pass.
- Corrections applied above:
  - the docs claim (TRL does document it, in `chat_templates.md`);
  - duplicates (#8071, #8072, #8109);
  - the `reasoning` scope;
  - "mirror image of KD-0002/0003" was misleading. KD-0002/0003 describe Axolotl's own labeling, which no setting changes; here TRL labels correctly when given the documented setting.
- Out of scope, ours: `t1.truncation` recorded "the row matches no truncation mode" when only the loss was wrong. Fixed in `001bfbe`.
- Out of scope, TRL, not a finding: if only some rows carry `chat_template_kwargs`, preparation crashes with a `TypeError` (loud, not silent).

## Proposed handling (human decides)
1. **Preferred (skeptic and me): in secondary runs, give each framework its documented setting for the mode.**
   - For TRL that is a per-row `chat_template_kwargs: {"enable_thinking": false}` column, its public, tested path, so rule 2 holds. Axolotl has its own documented `chat_template_kwargs`.
   - The cost is small: about 2 minutes for all six checks with planted faults.
   - But it changes the encoded rows, so it reverses part of the 2026-11-24 decision ("same encoded rows, no extra framework runs"). It needs your decision and a provenance note.
2. **If "same rows" stays:** label these cells "framework not configured for this mode" in the secondary table, rather than a bug or a TRL known difference. A KD entry would publish a difference TRL doesn't have.
3. **Until then:** all 14 cells are pinned to P-0003 in `findings/root_causes.json`, so they count once and are visibly ours, not TRL's.

## Repro
- `docs/evidence/2026-11-24-m3-rerun/run.py trl`, the non-thinking table. No standalone repro: nothing goes upstream.

## Upstream
- None.
