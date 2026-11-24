# Status

Keep this file short and true. `/wrap-session` updates it at the end of every session.

- **Current phase:** Phase 0, kill test. M0, M1 and M2 are done. M3 is in progress: the TRL adapter is done and has run all six checks on the calibration model; Axolotl and Unsloth are next. Milestone numbers follow `docs/phases/phase-0-kill-test.md`.
- **Last updated:** 2026-11-24
- **Gates:** A not started · B not started · C not started

## Done

- Plan written, starter kit installed, setup checked.
- Guard hook works on Windows (Bash and PowerShell) and has tests (`tests/test_guard.py`). It also blocks hidden-vault access from `~/parity` in WSL2 (checked 2026-11-23).
- M0: targets and the Qwen3 calibration model pinned by commit SHA (`docs/DECISIONS.md`).
- M1: the 8 fixtures and the Tier 1 template reference, cross-checked against TRL's Qwen3 training template. Goldens checkable by eye in `docs/evidence/golden-review.md`.
- M2: the six Tier 1 checks (`src/parity/checks/tier1/`), with 20 planted faults (2 real-world, with URLs), on a check engine that enforces the self-test (`src/parity/engine/`).
  - Reviewed by the skeptic subagent; its five bugs are fixed.
  - Follow-ups from the human's review are done: verified declarations (one per encode config), verdict flags, "blocked by t1.template_parity", model-level template properties on the report, narrow pre-fill anchoring.
- M3, TRL 1.18.2:
  - `src/parity/adapters/trl.py` runs TRL's public path; its locked environment is `envs/trl-1.18.2`.
  - On Qwen3-0.6B: 40 pass, 4 not applicable, 4 KD-0001 differences on exactly the predicted fixtures. All 131 planted-fault runs were caught; nothing went to triage (`docs/evidence/2026-11-24-m3-trl/`).
  - 370 tests pass in the core environment, and 392 in TRL's.
- M3's install check, done early (the platform spike): all five frameworks install and import their data path in WSL2 Ubuntu. `~/parity` is the only working copy.
- Adapter environments install the way each framework's docs say, and any deviation is recorded (`envs/<fw>-<ver>/environment.json`) and treated as a triage suspect. SkyRL installs from its repo (Phase 2).
- Phase 3 prerequisite logged: the hidden vault moves to a separate Linux user before any pair enters it, and the human runs hidden-vault evaluation. `Bash(sudo *)` is denied in `.claude/settings.json`.
- Design rules logged in `docs/DECISIONS.md`: mask zones, the known-differences registry (KD-0001), template properties, the Tier 1 pass rules, the engine rules, the mask reference, and two Tier 3 train/serve notes.

## Next actions

1. Human: review the M3 TRL report and the open questions below.
2. M3: Axolotl, then Unsloth (`/new-adapter axolotl`, `/new-adapter unsloth`).

## Open questions

- DeepSeek V4.1 still refuses on its 4 multi-assistant fixtures. The template merges the user messages around a removed assistant turn, so no cut-out delimits a history turn. Also open: which zone `</think>` belongs to in its history turns. See `docs/DECISIONS.md`, 2026-11-24.
- No target reaches "several stop tokens per turn" or "different closing tokens per turn". Proposal (not added): gpt-oss-20b as a second calibration tokenizer, after deciding how the reference treats `<|end|>`. Truncation-mode coverage needs a fixture, not a tokenizer. See `docs/evidence/2026-11-24-rule-coverage.md`.
- TRL drops rows left with no trainable token after truncation. No fixture reaches this today. If one does, decide whether it's a TRL-specific truncation mode (`docs/DECISIONS.md`, M3 TRL entry).
- M4 heads-up: if a framework skips `empty_assistant` on purpose and documents it, that's a known-differences entry; if it's undocumented, it's a candidate finding.
- The Hy4 license hasn't been reviewed (human). Do it before any Hy4 results are published.
- M3 open items still to settle with Axolotl and Unsloth: registry entries only from evidence, and Unsloth's CPU fallback.
- Before Phase 3 (human): close the `docker` group and WSL interop routes to root; edit the lines that name the vault path in `vault/README.md`, SETUP.md and Phase 3 M3.

## Findings so far

None yet.
