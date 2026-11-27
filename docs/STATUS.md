# Status

Keep this file short and true. `/wrap-session` updates it at the end of every session.

- **Current phase:** Phase 0, kill test. M0-M4 done (calibration green for all three). **M5:** the three local pairs are run and triaged; Unsloth x DeepSeek V4.1 waits on the Colab run (kit built). Milestone numbers follow `docs/phases/phase-0-kill-test.md`.
- **Last updated:** 2026-11-27 (P-0008 posted as a Community discussion on the model repo; H-13 fixed; the DeepSeek-only kit rebuilt)
- **Gates:** A in progress (2026-11-27 review, not `/gate-check`): A1, A2, A5, A7 met; A3 not met (Unsloth × DeepSeek V4.1 not run, no `runs/phase0/summary.md`); A4 met for every run so far, but KD-0001 has no skeptic verdict; A6 not met (no confirmation recorded; P-0002's collaborator reply is the human's call) · B not started · C not started

## Done

- Plan, starter kit, setup. The guard hook is tested on Windows and WSL2 (`tests/test_guard.py`).
- M0: targets and the Qwen3 calibration model pinned. M1: 8 fixtures and the Tier 1 template reference (`docs/evidence/golden-review.md`). M2: six Tier 1 checks with their planted faults (22, 3 real-world, since the special-tokens split) on an engine that enforces the self-test.
- Engine and reference:
  - **serving modes:** primary thinking mode, plus a secondary non-thinking table in which each framework gets its documented mode setting, or `not_configurable` cells;
  - **root-cause linking:** exact evidence pins; the report and Gate A count root causes, not cells;
  - **tool arguments:** a token-level false→False plant, and `tool_arguments` declared and verified by behavior;
  - **DeepSeek:** added-token structural headers.
- M3 on Qwen3-0.6B, both modes:
  - **TRL 1.18.2** (`docs/evidence/2026-11-24-m3-rerun/trl/`): primary 40 pass, 4 KD-0001, 4 n/a. Its only root cause is KD-0001.
  - **Axolotl 0.21.1** (`docs/evidence/2026-11-24-m3-rerun/axolotl/`): primary 27 pass, 13 KD-0002/0003 (P-0001), 4 P-0002, 4 n/a. 21 fail verdicts from 2 root causes.
  - **Unsloth 2026.11.10, GPU route, primary** (`docs/evidence/2026-11-25-m3-unsloth-gpu/`; Colab T4, verified against commit `15ed6df`):
    - thinking: 36 pass, 8 inconclusive (`t1.special_tokens` padding part), 4 n/a, no fails;
    - non-thinking: not configurable (44 cells, 4 n/a). Unsloth documents non-thinking training via the `/no_think` soft switch; hard-switch training is not configurable (P-0006, closed);
    - 81 of 96 cells as the docs-alone predictions said; the 15 misses are the special-tokens cells named in advance.
  - **Unsloth CPU backend, secondary** (`docs/evidence/2026-11-24-m3-unsloth/`): thinking 8 pass, 8 fail (P-0004), 28 inconclusive, 4 n/a; non-thinking not configurable.
- **M4 calibration, green for all three** (`runs/calibration-2026-11-25.md`, evidence in `docs/evidence/2026-11-25-m4-calibration/`), under the human's green definition (`parity.engine.calibration`): no untriaged fail, no harness-caused inconclusive, every planted fault caught. Allowed and listed:
  - **TRL 1.18.2:** 7 KD-0001;
  - **Axolotl 0.21.1:** 13 P-0001 and 8 P-0002;
  - **Unsloth 2026.11.10 (GPU, Colab T4):** 8 P-0005, the `t1.special_tokens` width rule on all 8 fixtures. Non-thinking mode is not configurable for Unsloth.
- **Harness errors caught: 13** (`findings/LEDGER.md`). They are never counted against a framework.

## M5 coverage, the honest math (primary table, 432 cells = 9 pairs x 48)

From the coverage preview (`docs/evidence/2026-11-25-m5-coverage-preview/`), the Day-0 support table (`docs/evidence/2026-11-25-m5-env-support/summary.md`) and the human's decisions (`docs/DECISIONS.md`, 2026-11-26). Nothing has run on a target yet, so "judgeable" means "at most": the predictions put some of these at inconclusive.

| Pair | Model | Judgeable at most | Not judgeable |
| --- | --- | --- | --- |
| TRL × Inkling (vlm variant) | the release's own (vision path) | 44 | 4 n/a |
| TRL × DeepSeek V4.1 | `deepseek_v4` stand-in | 0 | 44 `not_supported` (TRL's refusal, predicted), 4 n/a |
| Axolotl × DeepSeek V4.1 | `deepseek_v4` stand-in | 32 | 12 reference refuses, 4 n/a |
| Unsloth × DeepSeek V4.1 (GPU) | the release's own (text path) | 32 | 12 reference refuses, 4 n/a |
| TRL × Hy4, Axolotl × Inkling, Axolotl × Hy4, Unsloth × Inkling, Unsloth × Hy4 | not judged: can't build the release's processor | 0 | 240 |

- **Judgeable at most: 108 of 432.** 76 on the release's own architecture (TRL × Inkling 44, Unsloth × DeepSeek 32) and 32 on a stand-in (Axolotl × DeepSeek).
- **Not judgeable: 324.**
  - 240 in the five pairs whose release processor can't be built (their cells include 20 that would be n/a anyway);
  - 44 predicted `not_supported`;
  - 24 our reference refuses (DeepSeek's `tool_call` and `empty_assistant`);
  - 16 n/a.
- **Predicted, not yet run:** of the 108, the predictions expect 73 pass, 24 fail (DeepSeek V4.1's think markers, in Axolotl and Unsloth; one TRL × Inkling template-parity fail) and 11 inconclusive. These are the files in `docs/evidence/2026-11-26-m5-*/predictions.md`.
- Secondary (non-thinking) table: Inkling's 48 cells are n/a (D3), Unsloth × DeepSeek's are `not_configurable`, and TRL × DeepSeek's are `not_supported`.

## This session (2026-11-25 to 26)

- **The human's decisions:** D1 (release architecture where the framework's documented path builds it, otherwise a same-family stand-in recorded as a deviation; a support table per pair; stand-in fails leave the repo only with a standalone repro; the vision path kept when a framework's docs use it), D2 (`not_supported`, pinned to the exact exception), D3 (no documented mode, n/a), D4 (secondary table in M5, not counted in the 432). The M0 gap is logged as a process lesson in Phase 5.
- **Modes cited** at each target's pinned revision (`MODE_SPECS`, tested). Hy4 and Inkling from the card; DeepSeek V4.1 is "template only": its card says the release has no Jinja template and points to `encoding.py`, whose `thinking_mode` is required.
- **DeepSeek V4.1:** `encoding.py` is the model's own format (committed by the model team in the release; the Jinja was added 2026-11-10 as a port). Run sandboxed under the human's conditions: the Jinja agrees on 42 of 42 renders (strings and token ids). No model-level root cause. Observation only: `encoding.py`'s own loader drops top-level tools when a conversation starts with a user turn.
- **Support check:** the release's architecture builds in 2 of 9 pairs (TRL × Inkling, Unsloth × DeepSeek V4.1, via Unsloth's own implementation). Hy4's transformers support is an unmerged PR.
- **Built:** `parity.tiny` (Qwen3 unchanged, tested), D2/D3 in the engine, `model_build` in every declaration, the adapters on the factory with a path table per framework, TRL's pinned refusal, `processor_config.json` allowed.
- **Re-calibration** (`docs/evidence/2026-11-26-m4-recalibration/`): every verdict identical to M4 apart from the new `model_build` (TRL 96/96, Axolotl 96/96); Unsloth's CPU backend 96/96 before and after. The GPU route is re-checked on the next Colab trip.
- **2026-11-26, after the first checkpoint:**
  - P-0002 posted by the human (axolotl-ai-cloud/axolotl#4318); a collaborator replied the same day. Its mechanism reproduces on all three target tokenizers.
  - The human's decisions:
    - TRL's `vlm` variant for TRL × Inkling;
    - the vision-path pairs whose processor can't be built are not judged (applied to TRL × Hy4 too, flagged);
    - Unsloth × DeepSeek V4.1 follows the general docs with the template's role markers;
    - `encoding.py`'s loader observation is out of scope;
    - the sandbox write slip is harness error H-12.
  - Built: `envs/trl-1.18.2/vlm` (Qwen3 calibration there 96/96 identical to M4), the processor step and the per-pair Day-0 support table, the Unsloth recipe table (CPU calibration still 96/96). Predictions committed for the 4 runnable pairs, each on its own.
- Tests: core 590 (after KD-0004/KD-0005; the framework environments were last run in full before them: TRL 590, `vlm` 590, Axolotl 589, Unsloth 592 with 1 expected failure; the KD tests pass in the Axolotl environment).

## M5 so far (primary and secondary, 96 cells per pair)

| Pair | Cells as predicted | Primary result | Evidence |
| --- | --- | --- | --- |
| TRL 1.18.2 × DeepSeek V4.1 | 96 of 96 | 44 `not_supported` (TRL's pinned refusal: no generation markers), 4 n/a | `docs/evidence/2026-11-26-m5-trl-deepseek/` |
| TRL 1.18.2 (vlm) × Inkling-2-Mini | 96 of 96 | 38 pass, 1 fail (P-0008, `tool_call`), 5 inconclusive, 4 n/a | `docs/evidence/2026-11-26-m5-trl-inkling/` |
| Axolotl 0.21.1 × DeepSeek V4.1 | 81 of 96 (90 at the first run) | 17 pass, 1 fail (P-0007), 9 known differences (KD-0004: `</think>` untrained without reasoning, the human's decision after the run), 17 inconclusive, 4 n/a. Misses: the 6 `reasoning` cells, and the 9 cells predicted `fail` that are now differences (their behavior is unchanged) | `docs/evidence/2026-11-26-m5-axolotl-deepseek/` |
| Unsloth 2026.11.10 × DeepSeek V4.1 (GPU) | no results yet | the 2026-11-26 Colab run's DeepSeek zip never came back (no per-model zip, no combined zip) | `docs/evidence/2026-11-26-m5-unsloth-deepseek/predictions.md` |

- **Built for M5:** the runner refuses a pair without committed predictions (`parity.engine.predictions`, `docs/evidence/2026-11-26-m5-runs/run.py`); the GPU kit applies the same guard to its archive and zips each model's results as it finishes.
- **The 2026-11-26 Colab run** (kit `4b4de273e093`): only Qwen3's per-model zip came back; Chrome blocked the other downloads. Qwen3, verified and ingested on its own (`docs/evidence/2026-11-27-unsloth-gpu-per-model/`): 90 of 96 verdicts identical to M4's GPU run apart from `model_build`; the other 6 (non-thinking `reasoning`) moved from not_configurable to not_applicable by 8a00284, which came after M4's kit commit, as the CPU backend already shows. Calibration green (8 P-0005 fails, pinned).
- **The kit now makes one download** (the combined zip, cell 7) and lists every zip to keep, to take by hand from Colab's Files panel if the browser blocks it. It needs rebuilding at the new commit before the next trip.

## Next actions

1. **Human: watch [axolotl-ai-cloud/axolotl#4320](https://github.com/axolotl-ai-cloud/axolotl/issues/4320)** (P-0007, posted 2026-11-27) for a maintainer's reply.
2. **Human: watch [Inkling-2-Mini discussion 62](https://huggingface.co/thinkingmachines/Inkling-2-Mini/discussions/62)** (P-0008, posted 2026-11-27) for a reply. The post lost its Markdown formatting (links, headings, bullets; the code blocks are intact), and the closing line offering a PR isn't in it yet: editing the post with `findings/P-0008/issue.md`'s body, plus that line, would restore both.
3. **Human: the DeepSeek-only Colab run** with `dist/gpu-kit/54e700602b7f/` (built from `54e7006`, with H-13's fix). The first attempt (kit `55be6c81deb3`) stopped in the model load: our tiny checkpoint had no engram table rows (H-13, `docs/evidence/2026-11-27-unsloth-deepseek-engram/`); fixed and checked on CPU. Then: an M5 ingest for the target (the compare against predictions is still to write), the DeepSeek triage, and Unsloth's documented auto-detection on the same rows.
4. **After M5 (not yet):** whether `AutoProcessor`'s silent fallback to the tokenizer changes what TRL trains on for Inkling under a plain `pip install trl` (the human's item 4); predictions first.
5. **Human:** decide whether the collaborator's reply on axolotl-ai-cloud/axolotl#4318 (P-0002) confirms the finding; watch it for a fix. Watch unslothai/unsloth#19567 (P-0005).

## Open questions

- The structural method's added-header widening is exercised only on DeepSeek (8 agreeing turns); the method as a whole covers 3 families. Accept, or ask for more coverage?
- Unsloth's recipe follows the recommended notebook's `padding_free=False` (padded batches, so P-0005 applies). Thinking-2507 leaves it unset, and then Unsloth chooses `True` and keeps the width (measured on GPU). Should the report card say which recipe it judged? It currently says so only in the adapter's sources.
- Possible docs gaps, not findings: Axolotl never says think tags go untrained by default; TRL's SFT docs never say how to pass `chat_template_kwargs`, and `SFTConfig` has no field for it.
- The guard reads command text: a shell command that only writes text naming an environment with no committed predictions is refused too. Use the Edit/Write tools for such files.
- Deferred to Phase 2 (the kill test is frozen at 8 fixtures): gpt-oss-20b and the `<|end|>` question; a truncation fixture.
- TRL drops rows left with no trainable token after truncation: the rule is logged, and it will be implemented when a fixture hits it.
- Unsloth's tiny-model folder is still named `model` (as calibrated). Naming it after the release could switch on Unsloth's name-keyed code; check on the GPU run before changing it.
- Follow-up: Unsloth's git main (what the Inkling notebook installs) as a framework version of its own, with its own calibration.
- Human: review the Hy4 license. Before Phase 3: close the `docker` group and WSL interop routes; edit the vault-path lines in `vault/README.md`, SETUP.md and Phase 3 M3.

## Findings so far

- **P-0001** (Axolotl 0.21.1): only the think opener is untrained. Known difference KD-0002/KD-0003, thinking mode only; possible docs gap.
- **P-0002** (Axolotl 0.21.1): an empty assistant turn's end-of-turn token is never labeled, silently. Template-agnostic, and reproduces on main. Candidate, low severity on reach (ordinary reasoning and tool-call turns are not affected; checked). **Posted 2026-11-26:** [axolotl-ai-cloud/axolotl#4318](https://github.com/axolotl-ai-cloud/axolotl/issues/4318); an Axolotl collaborator replied the same day ("may have been an edge case") and invited a contributor's fix. Still a candidate: whether that counts as confirmation is the human's call. **Reproduces on all three target tokenizers** (DeepSeek V4.1, Inkling, Hy4) through Axolotl's `chat_template` strategy, tokenizer only (`findings/P-0002/target_reach.py`); for Inkling and Hy4 that is the text strategy, not their documented multimodal path.
- **P-0003** (TRL 1.18.2, secondary only): **harness error (H-9)**, ours; fixed. Never counted against TRL.
- **P-0004** (Unsloth 2026.11.10, CPU backend only): a second `<|im_end|>` on chat-formatted rows (TRL #3318 behavior). Candidate; on the card as CPU-only, low impact; not shrunk. **Confirmed CPU-only on GPU:** GPU rows equal the render. Card only; revisit after the kill test.
- **P-0005** (Unsloth 2026.11.10): `train_on_responses_only` silently drops `pad_to_multiple_of` with padded batches (the explicit padded path; Unsloth's padding-free default keeps it). Class: config value silently ignored. Candidate, **reproduced on GPU**, rule 4 cleared; labels not affected, loss the same up to float rounding. Shrunk. **Posted 2026-11-25:** [unslothai/unsloth#19567](https://github.com/unslothai/unsloth/issues/19567) (main repo, as unsloth-zoo#698 asks); awaiting a maintainer. Explains the 8 special-tokens width fails of the M4 GPU calibration (pinned).
- **P-0006** (Unsloth 2026.11.10, non-thinking mode): `train_on_responses_only` trains the last turn's empty think block, which our hard-switch reference treats as a pre-fill. **Closed: not_configurable.** Unsloth documents non-thinking training via the `/no_think` soft switch; hard-switch training is not configurable.
- **P-0007** (Axolotl 0.21.1, M5): message fields the template analyzer can't see are dropped before rendering (only a variable named `message` counts), though the schema says unmapped template properties load directly. **Reach today** (`docs/evidence/2026-11-26-p0007-reach/`): 3 of 36 bundled templates (`exaone4`, `gemma4`, `gemma4_unified`); through the full path, EXAONE-4.0 (bundled and own template) and gemma-4-12B-it's own template lose reasoning and tool calls silently, and the bundled Gemma 4 templates raise a `TypeError` on OpenAI-format tool results. Latent: DeepSeek V4.1. Candidate, medium for users today; no duplicate, not fixed on main. Workaround: explicit `message_property_mappings`. **Shrunk:** `findings/P-0007/repro.py` (EXAONE-4.0, bundled `exaone4`), fails 3/3 on 0.21.1 and on main. Pinned to its one cell; the 9 `</think>` cells are KD-0004/KD-0005. **Posted 2026-11-27:** [axolotl-ai-cloud/axolotl#4320](https://github.com/axolotl-ai-cloud/axolotl/issues/4320).
- **P-0008** (TRL 1.18.2 `vlm` × Inkling-2-Mini, M5): on the vision path, an assistant turn with `content: ""` and tool calls or reasoning gains a trained empty text message. Candidate, low provisionally (reach unmeasured). **Reference:** Thinking Machines' tml-renderers 0.5.0 (sandboxed, `docs/evidence/2026-11-26-p0008-renderer/`) equals the HF template for `""`, null and absent, and renders no empty text message for any form; the template's list branch departs from it, and TRL's wrapping of `""` (#5496) is the trigger. **Shrunk for the model repo:** `findings/P-0008/repro.py`, fails 3/3. Pinned. **Posted 2026-11-27:** [Inkling-2-Mini discussion 62](https://huggingface.co/thinkingmachines/Inkling-2-Mini/discussions/62).
