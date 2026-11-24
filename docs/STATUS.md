# Status

Keep this file short and true. `/wrap-session` updates it at the end of every session.

- **Current phase:** Phase 0, kill test. M0, M1 and M2 are done. M3 is in progress: TRL and Axolotl adapters are done and have run all six checks on the calibration model; Unsloth is next. Milestone numbers follow `docs/phases/phase-0-kill-test.md`.
- **Last updated:** 2026-11-24
- **Gates:** A not started · B not started · C not started

## Done

- Plan written, starter kit installed, setup checked.
- Guard hook works on Windows (Bash and PowerShell) and has tests (`tests/test_guard.py`). It also blocks hidden-vault access from `~/parity` in WSL2 (checked 2026-11-23).
- M0: targets and the Qwen3 calibration model pinned by commit SHA (`docs/DECISIONS.md`).
- M1: the 8 fixtures and the Tier 1 template reference. Goldens checkable by eye in `docs/evidence/golden-review.md`.
- M2: the six Tier 1 checks (`src/parity/checks/tier1/`), with 20 planted faults (2 real-world, with URLs), on a check engine that enforces the self-test (`src/parity/engine/`). Reviewed by the skeptic subagent; the human's follow-ups are done.
- M3, TRL 1.18.2 (`src/parity/adapters/trl.py`, `envs/trl-1.18.2`): on Qwen3-0.6B, 40 pass, 4 not applicable, 4 KD-0001 differences on exactly the predicted fixtures. Nothing went to triage (`docs/evidence/2026-11-24-m3-trl/`).
- M3, Axolotl 0.21.1 (`src/parity/adapters/axolotl.py`, `envs/axolotl-0.21.1`, telemetry opted out):
  - Predictions were committed from Axolotl's docs before the run; 45 of 48 cells matched.
  - On Qwen3-0.6B: 26 pass, 17 fail, 4 not applicable, 1 inconclusive (`docs/evidence/2026-11-24-m3-axolotl/`).
  - The fails were triaged into two root causes (P-0001, P-0002) and reviewed by the skeptic.
- Reference: structural delimiting of history turns, with guards. It agrees with the cut-out wherever both compute; DeepSeek still refuses on its header guard. `</think>` in DeepSeek history is prompt side by evidence.
- Environments are installed the way each framework's docs say, and deviations are recorded in `envs/<fw>-<ver>/environment.json`. 373 tests pass in the core environment; the suite also passes inside TRL's and Axolotl's.
- Phase 3 prerequisite logged: the hidden vault moves to a separate Linux user before any pair enters it, and the human runs hidden-vault evaluation. `Bash(sudo *)` is denied in `.claude/settings.json`.

## Next actions

1. Human: decide P-0001 (known-differences entry? reference thinking-mode decision?) and whether to take P-0002 forward.
2. If P-0002 goes ahead: `/shrink-repro P-0002` (use an empty *last* turn), then `/draft-issue P-0002`. The human posts.
3. M3: Unsloth (`/new-adapter unsloth`), after the decisions above.

## Open questions

- **Qwen3 thinking-mode ambiguity in the reference (from P-0001).** No-reasoning last turns render the same in both modes, but the empty think block is must-train in thinking mode and prompt side in non-thinking mode. Options: declare the mode per run and check both, or make mode-dependent tokens neutral.
- **Root causes shared between checks.** P-0001 fails both `t1.assistant_mask` and `t1.truncation`; "blocked by t1.template_parity" only covers template parity.
- **The `history_false_as_python` planted fault can't run through Axolotl.** Axolotl JSON-parses string arguments and refuses the Python repr, so `tool_call` is inconclusive on `t1.multiturn_prefix`. Options: a token-level variant, or a declared "arguments are parsed as JSON" property.
- DeepSeek V4.1 still refuses on its 4 multi-assistant fixtures. Open: accept tokenizer-declared added tokens (not only special ones) as headers for the structural method? See `docs/DECISIONS.md`, 2026-11-24.
- Deferred to Phase 2 (the kill test is frozen at 8 fixtures and 432 runs): gpt-oss-20b and the `<|end|>` question, and a truncation fixture. See `docs/phases/phase-2-tier1-breadth.md`.
- TRL drops rows left with no trainable token after truncation: the rule is logged, and it will be implemented when a fixture first hits it.
- The Hy4 license hasn't been reviewed (human). Do it before any Hy4 results are published.
- Before Phase 3 (human): close the `docker` group and WSL interop routes to root; edit the lines that name the vault path in `vault/README.md`, SETUP.md and Phase 3 M3.

## Findings so far

- **P-0001** (Axolotl 0.21.1, Qwen3): its default labeling never trains the think block. Proposed as a documented difference, partly our reference's mode assumption; not an upstream issue. Human decides.
- **P-0002** (Axolotl 0.21.1, Qwen3): an empty assistant turn's closing token is never trained, silently, despite `train_on_eos: turn`. Candidate, low severity; the skeptic agrees it's real.
