# Status

Keep this file short and true. `/wrap-session` updates it at the end of every session.

- **Current phase:** Phase 0, kill test. M0, M1 and M2 are done. M3 is in progress: TRL, Axolotl and Unsloth have run all six checks on the calibration model, in both serving modes. Milestone numbers follow `docs/phases/phase-0-kill-test.md`.
- **Last updated:** 2026-11-24
- **Gates:** A not started · B not started · C not started

## Done

- Plan written, starter kit installed, setup checked. Guard hook tested on Windows and WSL2 (`tests/test_guard.py`).
- M0: targets and the Qwen3 calibration model pinned by commit SHA. M1: 8 fixtures and the Tier 1 template reference (`docs/evidence/golden-review.md`).
- M2: six Tier 1 checks with 21 planted faults (2 real-world) on an engine that enforces the self-test.
- **Serving mode** (from P-0001, after results): `TrainCfg.serving_mode`, recorded in every verdict.
  - Primary results are in thinking mode, with framework defaults; the kill test stays at 432 runs.
  - The secondary non-thinking table gives each framework its documented mode setting (TRL: per-row `chat_template_kwargs`; Axolotl: the `chat_template_kwargs` config).
  - A framework with no documented way to set the mode gets `not_configurable` cells, never fails.
- **Root causes:** every fail links to a finding (`findings/root_causes.json`, exact evidence pins; differences through their KD entry). The report and Gate A count root causes, not cells.
- **Tool arguments:** token-level `history_false_as_python_tokens` plant guards every framework; `tool_arguments` is declared and verified by behavior (TRL passes through, Axolotl parses JSON), shown on the report card.
- **DeepSeek:** added tokens accepted as structural headers under the new guard (0 disagreements, 3 families). `multi_turn` and `reasoning` now compute; `empty_assistant` and `tool_call` still refuse.
- M3 re-run, both modes (`docs/evidence/2026-11-24-m3-rerun/`):
  - TRL 1.18.2: primary 40 pass, 4 KD-0001, 4 n/a (no bug fails, unchanged). Secondary 35 pass, 3 KD-0001, 6 inconclusive (`reasoning`), 4 n/a. One root cause, KD-0001.
  - Axolotl 0.21.1: primary 27 pass, 13 KD-0002/0003 (P-0001), 4 P-0002, 4 n/a, 0 inconclusive. Secondary 34 pass, 4 P-0002, 6 inconclusive (`reasoning`), 4 n/a. 21 fail verdicts from 2 root causes, 0 untriaged.
  - Unsloth 2026.11.10 (CPU backend; `src/parity/adapters/unsloth.py`, `envs/unsloth-2026.11.10`):
    - 48 of 48 primary cells as predicted (predictions after a disclosed probe).
    - Primary: 8 pass (end of turn), 8 fail (template parity, P-0004), 28 inconclusive, 4 n/a.
    - 15 fail verdicts from 1 root cause. Its labels aren't judged yet: every row carries one extra `<|im_end|>`, so nothing else aligns.
- **Harness errors caught: 9** (`findings/LEDGER.md`), including P-0003. A harness error is never counted against a framework.
- 427 tests pass in the core environment, and the suite passes inside TRL's and Axolotl's.

## Next actions

1. Human: run the P-0002 repro as `findings/P-0002/HOW-TO-RUN.md` says (it should exit 1), then post `findings/P-0002/issue.md` through Axolotl's issue form. Record the URL in the record.
2. Human: decide how the report card shows P-0004 (CPU-backend behavior, labeled CPU-only), and whether to `/shrink-repro P-0004`.
3. Triage the `pad_to_multiple_of` drop in `train_on_responses_only` as its own finding.
4. Unsloth's labels are not judged until rows align. Options: run on a GPU, or wait for an Unsloth CPU fix. Don't work around it in the adapter (rule 7).

## Open questions

- The structural method's added-header widening is exercised only on DeepSeek (8 agreeing turns); the method as a whole covers 3 families. Accept, or ask for more coverage?
- `reasoning` has no non-thinking reading in our reference (it refuses), though TRL's template gives one. Fine as inconclusive for now.
- Unsloth: no assistant-only recipe exists for the original Qwen3 models; the recommended Qwen3 notebook trains on every token. The recipe choice is logged.
- Possible docs gaps, not findings: Axolotl never says think tags go untrained by default; TRL's SFT docs never say how to pass `chat_template_kwargs`.
- Deferred to Phase 2 (kill test frozen at 8 fixtures): gpt-oss-20b and the `<|end|>` question; a truncation fixture.
- TRL drops rows left with no trainable token after truncation: rule logged, implemented when a fixture hits it.
- The Hy4 license hasn't been reviewed (human). Before Phase 3 (human): close the `docker` group and WSL interop routes; edit the vault-path lines in `vault/README.md`, SETUP.md and Phase 3 M3.

## Findings so far

- **P-0001** (Axolotl 0.21.1, Qwen3): only the think opener (`<think>` + blank lines) is untrained; reasoning text is trained. Known difference KD-0002/KD-0003, thinking mode only; possible docs gap.
- **P-0002** (Axolotl 0.21.1): an empty assistant turn's end-of-turn token is never labeled, silently. Not template-specific: reproduced on Qwen3, ChatML (SmolLM2), Llama 3.2, Mistral v0.3 and Gemma 3. Candidate, low severity (tool-call turns not affected). Reproduces on main; issue drafted, template-agnostic, with permalinks at `de8549a`. **Not posted.**
- **P-0004** (Unsloth 2026.11.10, CPU backend only): stock TRL preparation appends a second `<|im_end|>` to chat-formatted rows; Unsloth's GPU-route preparation doesn't. Candidate, low severity; the skeptic agrees. Not Unsloth's GPU recipe behavior.
- **P-0003** (TRL 1.18.2, secondary only): **harness error (H-9)**, ours. Fixed: frameworks now get their documented mode setting. Never counted against TRL.
