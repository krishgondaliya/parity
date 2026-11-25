# Status

Keep this file short and true. `/wrap-session` updates it at the end of every session.

- **Current phase:** Phase 0, kill test. M0-M3 done; **M4 (formal calibration) in progress.** TRL and Axolotl ran locally and are not green, for one harness reason: our reference refuses the `reasoning` fixture in non-thinking mode (a decision for the human). Unsloth calibrates on GPU next. Milestone numbers follow `docs/phases/phase-0-kill-test.md`.
- **Last updated:** 2026-11-25 (M4 calibration: TRL and Axolotl run; Unsloth's GPU kit built)
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
- Severity bar recorded before any target results: continuing with full energy needs a medium finding (changes labels, loss or stop behavior). P-0002 and P-0005 are low (P-0002's kind of effect flagged for confirmation).
- M4 calibration, TRL and Axolotl (`runs/calibration-2026-11-25.md`): no untriaged fails (P-0002's special-tokens cells re-pinned after the split). Not green: the `reasoning` × non-thinking reference refusal.
- The human's decisions: Unsloth's non-thinking mode is `not_configurable` (P-0006 closed; the saved cells were re-derived, as-run results kept), and P-0004 stays card-only until after the kill test. A soft-switch reference is a Phase 2 idea.
- P-0005-only GPU trip ingested (`docs/evidence/2026-11-25-p0005-gpu/`):
  - rule 4 cleared, the drop reaches a real training step, and Unsloth's padding-free default keeps the setting;
  - labels not affected, loss the same up to float rounding (CPU);
  - the repro is shrunk to 30 lines, and fails on 2026.11.12 and on unsloth-zoo main;
  - the issue was drafted, reviewed, and posted by the human on 2026-11-25 as unslothai/unsloth#19567.
- `t1.special_tokens`' padding part is now two rules, each with its own plant: requested width honored (new real-world plant `pad_to_ignored`), and pad handling at the produced width. 22 planted faults (3 real-world).
- Tests: core 486, Unsloth environment 507 (1 expected failure). TRL (495) and Axolotl (494) passed after the backend change; they're untouched since.

## Next actions

1. **Human decision:** resolve `reasoning` × non-thinking (`docs/DECISIONS.md`, M4 calibration entry): (a) not_applicable from the fixture and mode (recommended), (b) a non-thinking reading in the reference, or (c) allow documented reference limits. Then re-run TRL and Axolotl calibration (`docs/evidence/2026-11-25-m4-calibration/run.py`, `report.py`).
2. **Unsloth GPU calibration** (the human): the kit with `RUN = "full"` (folder in this session's notes; rebuild with `docs/gpu-run/build_kit.py` after any commit). When the zip is back: `report.py --unsloth-zip <zip>`; /triage the special-tokens width fails (expected `pad_to_honored`, P-0005) and pin them to P-0005 (backend gpu).
3. **Human:** run the P-0002 repro as `findings/P-0002/HOW-TO-RUN.md` says (it should exit 1), then post `findings/P-0002/issue.md` through Axolotl's issue form, and put the URL in the record.
4. Then the kill test proper: 3 frameworks x 3 target models x 6 checks x 8 fixtures = 432 primary runs.

## Open questions

- The structural method's added-header widening is exercised only on DeepSeek (8 agreeing turns); the method as a whole covers 3 families. Accept, or ask for more coverage?
- `reasoning` has no non-thinking reading in our reference (it refuses), though TRL's template gives one. It was fine as inconclusive, but it now blocks M4 green: see Next actions 1.
- Unsloth: the two notebooks the recipe draws on differ on `padding_free` (the recommended one sets `False`; Thinking-2507 leaves it unset). On GPU that decides whether batches are padded, and so whether P-0005 applies. The P-0005 trip's `padding_free_unset` run measures what Unsloth chooses.
- Possible docs gaps, not findings: Axolotl never says think tags go untrained by default; TRL's SFT docs never say how to pass `chat_template_kwargs`, and `SFTConfig` has no field for it.
- The guard reads command text: a shell command that only writes text naming an environment with no committed predictions is refused too. Use the Edit/Write tools for such files.
- Deferred to Phase 2 (the kill test is frozen at 8 fixtures): gpt-oss-20b and the `<|end|>` question; a truncation fixture.
- TRL drops rows left with no trainable token after truncation: the rule is logged, and it will be implemented when a fixture hits it.
- Human: review the Hy4 license. Before Phase 3: close the `docker` group and WSL interop routes; edit the vault-path lines in `vault/README.md`, SETUP.md and Phase 3 M3.

## Findings so far

- **P-0001** (Axolotl 0.21.1): only the think opener is untrained. Known difference KD-0002/KD-0003, thinking mode only; possible docs gap.
- **P-0002** (Axolotl 0.21.1): an empty assistant turn's end-of-turn token is never labeled, silently. Template-agnostic, and reproduces on main. Candidate, low severity. Issue drafted, **not posted**.
- **P-0003** (TRL 1.18.2, secondary only): **harness error (H-9)**, ours; fixed. Never counted against TRL.
- **P-0004** (Unsloth 2026.11.10, CPU backend only): a second `<|im_end|>` on chat-formatted rows (TRL #3318 behavior). Candidate; on the card as CPU-only, low impact; not shrunk. **Confirmed CPU-only on GPU:** GPU rows equal the render. Card only; revisit after the kill test.
- **P-0005** (Unsloth 2026.11.10): `train_on_responses_only` silently drops `pad_to_multiple_of` with padded batches (the explicit padded path; Unsloth's padding-free default keeps it). Class: config value silently ignored. Candidate, **reproduced on GPU**, rule 4 cleared; labels not affected, loss the same up to float rounding. Shrunk. **Posted 2026-11-25:** [unslothai/unsloth#19567](https://github.com/unslothai/unsloth/issues/19567) (main repo, as unsloth-zoo#698 asks); awaiting a maintainer.
- **P-0006** (Unsloth 2026.11.10, non-thinking mode): `train_on_responses_only` trains the last turn's empty think block, which our hard-switch reference treats as a pre-fill. **Closed: not_configurable.** Unsloth documents non-thinking training via the `/no_think` soft switch; hard-switch training is not configurable.
