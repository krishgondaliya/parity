# Status

Keep this file short and true. `/wrap-session` updates it at the end of every session.

- **Current phase:** Phase 0, kill test. M0, M1 and M2 are done. M3 is in progress: TRL, Axolotl and Unsloth have run all six checks on the calibration model, in both serving modes. Unsloth ran only on its CPU backend, so its labels aren't judged yet. Milestone numbers follow `docs/phases/phase-0-kill-test.md`.
- **Last updated:** 2026-11-25
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
  - **Unsloth 2026.11.10, CPU backend** (`docs/evidence/2026-11-24-m3-unsloth/`): primary 8 pass, 8 fail (P-0004), 28 inconclusive, 4 n/a. 48/48 as predicted, but the predictions came after a disclosed probe.
- **Harness errors caught: 9** (`findings/LEDGER.md`). They are never counted against a framework.

## This session (2026-11-24 to 25)

- Unsloth adapter and environment:
  - `src/parity/adapters/unsloth.py`, `envs/unsloth-2026.11.10`;
  - the recommended Qwen3 notebook's recipe, plus `train_on_responses_only` from the Qwen3-Thinking-2507 notebook (no original-Qwen3 notebook has an assistant-only step);
  - loaded through `FastLanguageModel.from_pretrained` on a local tiny model with the real tokenizer;
  - the CPU fallback and other deviations are recorded in `environment.json`.
- Triaged P-0004 (Unsloth CPU backend appends a second `<|im_end|>`) and P-0005 (`train_on_responses_only` silently drops `pad_to_multiple_of`); skeptic-reviewed both. P-0004 is on the report card as CPU-only, low impact, via the new root-cause `card_label`.
- Predictions-first is enforced. The guard hook refuses to create or run anything in a framework environment until a committed `predictions.md` names that framework and version (`trl-1.18.2` grandfathered). `/new-adapter` starts with step 0.
- Earlier in the session:
  - P-0003 became harness error H-9; non-default modes now get each framework's mode setting;
  - TRL's per-row setting is documented at the "docstring only" citation level;
  - P-0002 was made template-agnostic, with permalinks and `HOW-TO-RUN.md`.
- 451 tests pass in the core environment. The suite last passed inside TRL's (449), Axolotl's (448) and Unsloth's (446) environments before today's guard-hook and card-label changes; those changes don't touch adapter code.

## Next actions

1. **Unsloth GPU run** (needs a CUDA machine; human to provide one).
   - On it, `cd envs/unsloth-2026.11.10 && uv sync` (the lock already resolves the CUDA torch wheel).
   - Before running, add a `backend` (cpu/gpu) record to Unsloth's verdicts and a backend split to the report: the GPU route becomes Unsloth's primary result, and the current results become the labeled secondary "CPU backend" table.
   - Then run `docs/evidence/2026-11-24-m3-unsloth/run.py`:
     - judge Unsloth's labels;
     - revisit P-0004;
     - run `findings/P-0005/repro.py` (padded batches), and once with Unsloth's GPU defaults (`padding_free` unset).
2. **Human:** run the P-0002 repro as `findings/P-0002/HOW-TO-RUN.md` says (it should exit 1), then post `findings/P-0002/issue.md` through Axolotl's issue form, and put the URL in the record.
3. Then the kill test proper: 3 frameworks x 3 target models x 6 checks x 8 fixtures = 432 primary runs.

## Open questions

- The structural method's added-header widening is exercised only on DeepSeek (8 agreeing turns); the method as a whole covers 3 families. Accept, or ask for more coverage?
- `reasoning` has no non-thinking reading in our reference (it refuses), though TRL's template gives one. Fine as inconclusive for now.
- Unsloth: the two notebooks the recipe draws on differ on `padding_free` (the recommended one sets `False`; Thinking-2507 leaves it unset). On GPU that decides whether batches are padded, and so whether P-0005 applies.
- Possible docs gaps, not findings: Axolotl never says think tags go untrained by default; TRL's SFT docs never say how to pass `chat_template_kwargs`, and `SFTConfig` has no field for it.
- The guard reads command text: a shell command that only writes text naming an environment with no committed predictions is refused too. Use the Edit/Write tools for such files.
- Deferred to Phase 2 (the kill test is frozen at 8 fixtures): gpt-oss-20b and the `<|end|>` question; a truncation fixture.
- TRL drops rows left with no trainable token after truncation: the rule is logged, and it will be implemented when a fixture hits it.
- Human: review the Hy4 license. Before Phase 3: close the `docker` group and WSL interop routes; edit the vault-path lines in `vault/README.md`, SETUP.md and Phase 3 M3.

## Findings so far

- **P-0001** (Axolotl 0.21.1): only the think opener is untrained. Known difference KD-0002/KD-0003, thinking mode only; possible docs gap.
- **P-0002** (Axolotl 0.21.1): an empty assistant turn's end-of-turn token is never labeled, silently. Template-agnostic, and reproduces on main. Candidate, low severity. Issue drafted, **not posted**.
- **P-0003** (TRL 1.18.2, secondary only): **harness error (H-9)**, ours; fixed. Never counted against TRL.
- **P-0004** (Unsloth 2026.11.10, CPU backend only): a second `<|im_end|>` on chat-formatted rows (TRL #3318 behavior; Unsloth's GPU route doesn't add it). Candidate; on the card as CPU-only, low impact; not shrunk; revisit on GPU.
- **P-0005** (Unsloth 2026.11.10): `train_on_responses_only` silently drops `pad_to_multiple_of` with padded batches. Class: config value silently ignored. Candidate; confirm on GPU before any draft.
