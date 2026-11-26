# Status

Keep this file short and true. `/wrap-session` updates it at the end of every session.

- **Current phase:** Phase 0, kill test. M0-M4 done (calibration green for all three). **M5 prep:** modes cited, coverage preview, support check, DeepSeek's Jinja checked against its own `encoding.py`; the tiny model factory, D2 (`not_supported`) and D3 built; the adapters use them, and calibration is unchanged cell for cell. **Waiting on the human:** the vision path's processors (below), before the M5 predictions. Milestone numbers follow `docs/phases/phase-0-kill-test.md`.
- **Last updated:** 2026-11-26 (M5 prep built; vision-path pairs pending)
- **Gates:** A not started · B not started · C not started

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
- **Harness errors caught: 11** (`findings/LEDGER.md`). They are never counted against a framework.

## M5 coverage, the honest math (primary table, 432 cells = 9 pairs x 48)

From the coverage preview (`docs/evidence/2026-11-25-m5-coverage-preview/`), the support check (`docs/evidence/2026-11-25-m5-env-support/`) and the paths read from each framework's code (`docs/DECISIONS.md`, 2026-11-26). Nothing has run on a target yet, so "judgeable" means "at most": a cell can still end inconclusive.

- **Judgeable at most: 284 of 432.**
  - **On the release's own architecture: 76.** TRL × Inkling 44 (vision path, pending: its processor needs `torchvision`), Unsloth × DeepSeek V4.1 32 (text path; Unsloth's recipe for it still open).
  - **On stand-ins: 208.** Axolotl × DeepSeek V4.1 32 (`deepseek_v4`, text, decided). The other 176 are four vision-path pairs (Axolotl × Inkling, Axolotl × Hy4, Unsloth × Inkling, Unsloth × Hy4) that can be judged only if a stand-in processor is accepted, and for Unsloth × Inkling only if its stand-in can be built (it needs `scipy`).
- **Not judgeable at all: 148.** 36 n/a (4 per pair: the multi-turn check on single-assistant fixtures); 24 where our reference refuses (DeepSeek V4.1 `tool_call` and `empty_assistant`, in Axolotl and Unsloth); 88 `not_supported`, predicted from TRL's source (TRL refuses assistant-only loss on Hy4's and DeepSeek's templates: no generation markers, no stored training template).
- **If none of the six pending vision-path pairs can be judged: 64 of 432** (Axolotl × DeepSeek 32 on a stand-in, Unsloth × DeepSeek 32 on the release's architecture), all on one model.
- Secondary (non-thinking) table: at most 64 of 144 per framework; Inkling's 48 are n/a (D3: its card documents no non-thinking mode).

## This session (2026-11-25 to 26)

- **The human's decisions:** D1 (release architecture where the framework's documented path builds it, otherwise a same-family stand-in recorded as a deviation; a support table per pair; stand-in fails leave the repo only with a standalone repro; the vision path kept when a framework's docs use it), D2 (`not_supported`, pinned to the exact exception), D3 (no documented mode, n/a), D4 (secondary table in M5, not counted in the 432). The M0 gap is logged as a process lesson in Phase 5.
- **Modes cited** at each target's pinned revision (`MODE_SPECS`, tested). Hy4 and Inkling from the card; DeepSeek V4.1 is "template only": its card says the release has no Jinja template and points to `encoding.py`, whose `thinking_mode` is required.
- **DeepSeek V4.1:** `encoding.py` is the model's own format (committed by the model team in the release; the Jinja was added 2026-11-10 as a port). Run sandboxed under the human's conditions: the Jinja agrees on 42 of 42 renders (strings and token ids). No model-level root cause. Observation only: `encoding.py`'s own loader drops top-level tools when a conversation starts with a user turn.
- **Support check:** the release's architecture builds in 2 of 9 pairs (TRL × Inkling, Unsloth × DeepSeek V4.1, via Unsloth's own implementation). Hy4's transformers support is an unmerged PR.
- **Built:** `parity.tiny` (Qwen3 unchanged, tested), D2/D3 in the engine, `model_build` in every declaration, the adapters on the factory with a path table per framework, TRL's pinned refusal, `processor_config.json` allowed.
- **Re-calibration** (`docs/evidence/2026-11-26-m4-recalibration/`): every verdict identical to M4 apart from the new `model_build` (TRL 96/96, Axolotl 96/96); Unsloth's CPU backend 96/96 before and after. The GPU route is re-checked on the next Colab trip.
- Tests: core 561, TRL environment 590, Axolotl environment 589, Unsloth environment 592 (1 expected failure).

## Next actions

1. **Human: the vision-path processors** (`docs/DECISIONS.md`, 2026-11-26), for TRL × Inkling (TRL's own `vlm` extra adds `torchvision`), Axolotl × Inkling and × Hy4, and Unsloth × Inkling and × Hy4 (no release processor in the environment; a stand-in processor, or not judged). Also: Unsloth's recipe for DeepSeek V4.1 (Unsloth documents running it, not training it).
2. **Then the M5 predictions:** one folder per pair (`docs/evidence/<date>-m5-<framework>-<target>/predictions.md`), from docs and source alone, each saying the support-check and factory results were known; the M5 runner refuses a pair without its committed predictions.
3. **Then the Unsloth kit** at that commit, `MODELS` = Qwen3 first (re-checks the GPU calibration) plus the targets, writing each model's results as it finishes.
4. **Human:** run the P-0002 repro as `findings/P-0002/HOW-TO-RUN.md` says (it should exit 1), then post `findings/P-0002/issue.md` through Axolotl's issue form, and put the URL in the record.
5. Watch unslothai/unsloth#19567 (P-0005) for a maintainer's reply.

## Open questions

- The structural method's added-header widening is exercised only on DeepSeek (8 agreeing turns); the method as a whole covers 3 families. Accept, or ask for more coverage?
- Unsloth's recipe follows the recommended notebook's `padding_free=False` (padded batches, so P-0005 applies). Thinking-2507 leaves it unset, and then Unsloth chooses `True` and keeps the width (measured on GPU). Should the report card say which recipe it judged? It currently says so only in the adapter's sources.
- Possible docs gaps, not findings: Axolotl never says think tags go untrained by default; TRL's SFT docs never say how to pass `chat_template_kwargs`, and `SFTConfig` has no field for it.
- The guard reads command text: a shell command that only writes text naming an environment with no committed predictions is refused too. Use the Edit/Write tools for such files.
- Deferred to Phase 2 (the kill test is frozen at 8 fixtures): gpt-oss-20b and the `<|end|>` question; a truncation fixture.
- TRL drops rows left with no trainable token after truncation: the rule is logged, and it will be implemented when a fixture hits it.
- Unsloth's tiny-model folder is still named `model` (as calibrated). Naming it after the release could switch on Unsloth's name-keyed code; check on the GPU run before changing it.
- Hy4's template lacks its tokenizer's EOS (`<｜hy_end:opensource｜>`), so Axolotl would warn to set `eot_tokens`; decide with the Hy4 recipe.
- Follow-up: Unsloth's git main (what the Inkling notebook installs) as a framework version of its own, with its own calibration.
- Human: review the Hy4 license. Before Phase 3: close the `docker` group and WSL interop routes; edit the vault-path lines in `vault/README.md`, SETUP.md and Phase 3 M3.

## Findings so far

- **P-0001** (Axolotl 0.21.1): only the think opener is untrained. Known difference KD-0002/KD-0003, thinking mode only; possible docs gap.
- **P-0002** (Axolotl 0.21.1): an empty assistant turn's end-of-turn token is never labeled, silently. Template-agnostic, and reproduces on main. Candidate, low severity on reach (ordinary reasoning and tool-call turns are not affected; checked). Issue drafted, **not posted**.
- **P-0003** (TRL 1.18.2, secondary only): **harness error (H-9)**, ours; fixed. Never counted against TRL.
- **P-0004** (Unsloth 2026.11.10, CPU backend only): a second `<|im_end|>` on chat-formatted rows (TRL #3318 behavior). Candidate; on the card as CPU-only, low impact; not shrunk. **Confirmed CPU-only on GPU:** GPU rows equal the render. Card only; revisit after the kill test.
- **P-0005** (Unsloth 2026.11.10): `train_on_responses_only` silently drops `pad_to_multiple_of` with padded batches (the explicit padded path; Unsloth's padding-free default keeps it). Class: config value silently ignored. Candidate, **reproduced on GPU**, rule 4 cleared; labels not affected, loss the same up to float rounding. Shrunk. **Posted 2026-11-25:** [unslothai/unsloth#19567](https://github.com/unslothai/unsloth/issues/19567) (main repo, as unsloth-zoo#698 asks); awaiting a maintainer. Explains the 8 special-tokens width fails of the M4 GPU calibration (pinned).
- **P-0006** (Unsloth 2026.11.10, non-thinking mode): `train_on_responses_only` trains the last turn's empty think block, which our hard-switch reference treats as a pre-fill. **Closed: not_configurable.** Unsloth documents non-thinking training via the `/no_think` soft switch; hard-switch training is not configurable.
