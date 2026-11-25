# Status

Keep this file short and true. `/wrap-session` updates it at the end of every session.

- **Current phase:** Phase 0, kill test. M0-M4 done: **calibration is green for all three frameworks** (`runs/calibration-2026-11-25.md`). Next is M5, the 432 target runs. Milestone numbers follow `docs/phases/phase-0-kill-test.md`.
- **Last updated:** 2026-11-25 (M4 calibration green: TRL, Axolotl, Unsloth GPU)
- **Gates:** A not started · B not started · C not started

## Done

- Plan, starter kit, setup. The guard hook is tested on Windows and WSL2 (`tests/test_guard.py`).
- M0: targets and the Qwen3 calibration model pinned. M1: 8 fixtures and the Tier 1 template reference (`docs/evidence/golden-review.md`). M2: six Tier 1 checks with 21 planted faults (2 real-world) on an engine that enforces the self-test.
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

## This session (2026-11-25)

- GPU predictions first (`290eff2`, docs alone, 96 cells). Then the Colab kit (`docs/gpu-run/`):
  - `Verdict.backend`, with backend-exact root causes;
  - GPU primary and CPU secondary tables in the report;
  - a CUDA 12.6 lock variant;
  - P-0005 as an optional cell, and now a P-0005-only mode (`RUN = "p0005"`).
- The GPU run came back and was ingested (`ingest.py`: manifests, `git archive` rebuild, lockfile, GPU route, every verdict).
- Triaged:
  - P-0006, exploratory: the non-thinking fails are the last turn's empty think block, trained by `train_on_responses_only`;
  - P-0004 confirmed CPU-only;
  - the special-tokens inconclusives are consistent with P-0005, but the adapter's `pad_to` mapping isn't yet ruled out on GPU;
  - the P-0005 repro's GPU crash is ours (H-10, no LoRA attached); fixed;
  - skeptic-reviewed: P-0006 reframed as a difference of non-thinking convention (Unsloth documents the `/no_think` soft switch), and a second harness error found and fixed (H-11, the adapter collated in Unsloth's post-`__init__` inference state);
  - the stale secondary-table header in every summary was corrected.
- **Severity bar,** recorded before any target results and amended the same day before M5 (the human's decisions): severity = effect × reach.
  - Medium means a change to labels, loss or stop behavior on inputs that ordinary training data contains.
  - P-0002 and P-0005 are low. P-0002 is low on reach, checked on ordinary reasoning and tool-call shapes (`findings/P-0002/reach_check.py`).
- **M4 calibration** (`runs/calibration-2026-11-25.md`):
  - P-0002's special-tokens cells were re-pinned after the split.
  - The `reasoning` × non-thinking reference refusal blocked green. It was resolved by the human's option (a): a fixture whose assistant turn carries reasoning is not applicable in non-thinking mode, decided from the fixture and the mode before any framework output (`parity.serving.not_applicable_in_mode`). Re-run: only those 6 cells changed per framework; TRL and Axolotl are green.
  - Unsloth's GPU calibration zip was verified and merged. Its 8 special-tokens width fails were triaged against P-0005 (rule 4 cleared from each verdict's declaration) and pinned (backend gpu). Green.
- The human's decisions: Unsloth's non-thinking mode is `not_configurable` (P-0006 closed; the saved cells were re-derived, as-run results kept), and P-0004 stays card-only until after the kill test. A soft-switch reference is a Phase 2 idea.
- P-0005-only GPU trip ingested (`docs/evidence/2026-11-25-p0005-gpu/`):
  - rule 4 cleared, the drop reaches a real training step, and Unsloth's padding-free default keeps the setting;
  - labels not affected, loss the same up to float rounding (CPU);
  - the repro is shrunk to 30 lines, and fails on 2026.11.12 and on unsloth-zoo main;
  - the issue was drafted, reviewed, and posted by the human on 2026-11-25 as unslothai/unsloth#19567.
- `t1.special_tokens`' padding part is now two rules, each with its own plant: requested width honored (new real-world plant `pad_to_ignored`), and pad handling at the produced width. 22 planted faults (3 real-world).
- Tests: core 513, TRL environment 535, Axolotl environment 534, Unsloth environment 537 (1 expected failure).

## Next actions

1. **M5, the 432 runs:** 3 frameworks x 3 target models x 6 checks x 8 fixtures (primary). TRL and Axolotl run locally. Unsloth runs on GPU: rebuild the kit at the current commit (`docs/gpu-run/build_kit.py`) and set `MODELS` to the three targets. The `b73da05` kit predates the applicability rule.
2. **Human:** run the P-0002 repro as `findings/P-0002/HOW-TO-RUN.md` says (it should exit 1), then post `findings/P-0002/issue.md` through Axolotl's issue form, and put the URL in the record.
3. Watch unslothai/unsloth#19567 (P-0005) for a maintainer's reply.

## Open questions

- The structural method's added-header widening is exercised only on DeepSeek (8 agreeing turns); the method as a whole covers 3 families. Accept, or ask for more coverage?
- Unsloth: the two notebooks the recipe draws on differ on `padding_free` (the recommended one sets `False`; Thinking-2507 leaves it unset). On GPU that decides whether batches are padded, and so whether P-0005 applies. The P-0005 trip's `padding_free_unset` run measures what Unsloth chooses.
- Possible docs gaps, not findings: Axolotl never says think tags go untrained by default; TRL's SFT docs never say how to pass `chat_template_kwargs`, and `SFTConfig` has no field for it.
- The guard reads command text: a shell command that only writes text naming an environment with no committed predictions is refused too. Use the Edit/Write tools for such files.
- Deferred to Phase 2 (the kill test is frozen at 8 fixtures): gpt-oss-20b and the `<|end|>` question; a truncation fixture.
- TRL drops rows left with no trainable token after truncation: the rule is logged, and it will be implemented when a fixture hits it.
- Human: review the Hy4 license. Before Phase 3: close the `docker` group and WSL interop routes; edit the vault-path lines in `vault/README.md`, SETUP.md and Phase 3 M3.

## Findings so far

- **P-0001** (Axolotl 0.21.1): only the think opener is untrained. Known difference KD-0002/KD-0003, thinking mode only; possible docs gap.
- **P-0002** (Axolotl 0.21.1): an empty assistant turn's end-of-turn token is never labeled, silently. Template-agnostic, and reproduces on main. Candidate, low severity on reach (ordinary reasoning and tool-call turns are not affected; checked). Issue drafted, **not posted**.
- **P-0003** (TRL 1.18.2, secondary only): **harness error (H-9)**, ours; fixed. Never counted against TRL.
- **P-0004** (Unsloth 2026.11.10, CPU backend only): a second `<|im_end|>` on chat-formatted rows (TRL #3318 behavior). Candidate; on the card as CPU-only, low impact; not shrunk. **Confirmed CPU-only on GPU:** GPU rows equal the render. Card only; revisit after the kill test.
- **P-0005** (Unsloth 2026.11.10): `train_on_responses_only` silently drops `pad_to_multiple_of` with padded batches (the explicit padded path; Unsloth's padding-free default keeps it). Class: config value silently ignored. Candidate, **reproduced on GPU**, rule 4 cleared; labels not affected, loss the same up to float rounding. Shrunk. **Posted 2026-11-25:** [unslothai/unsloth#19567](https://github.com/unslothai/unsloth/issues/19567) (main repo, as unsloth-zoo#698 asks); awaiting a maintainer. Explains the 8 special-tokens width fails of the M4 GPU calibration (pinned).
- **P-0006** (Unsloth 2026.11.10, non-thinking mode): `train_on_responses_only` trains the last turn's empty think block, which our hard-switch reference treats as a pre-fill. **Closed: not_configurable.** Unsloth documents non-thinking training via the `/no_think` soft switch; hard-switch training is not configurable.
