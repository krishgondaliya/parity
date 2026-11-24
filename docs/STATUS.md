# Status

Keep this file short and true. `/wrap-session` updates it at the end of every session.

- **Current phase:** Phase 0, kill test. M0 and M2 are done. The M1 platform spike ran on native Windows and led to the decision to move to WSL2 Ubuntu.
- **Last updated:** 2026-11-23
- **Gates:** A not started · B not started · C not started

## Done

- Plan written, starter kit installed, setup checked.
- Guard hook works on Windows (Bash and PowerShell) and has tests (`tests/test_guard.py`).
- M0: targets and the Qwen3 calibration model pinned by commit SHA (`docs/DECISIONS.md`).
- M2, done early: the 8 fixtures and the Tier 1 template reference, cross-checked against TRL's Qwen3 training template. 136 tests pass.
- Design rules logged: mask zones, the known-differences registry (KD-0001), template properties, and a Tier 3 train/serve note.
- M1 platform spike on native Windows: Axolotl can't install (it needs Linux-only `triton`), so development moves to WSL2 Ubuntu (`docs/DECISIONS.md`).

## Next actions

1. Re-run the platform spike inside WSL with real installs, then M3.
   - Prerequisite (human): move into WSL2 Ubuntu. Clone into `~/parity`, copy `.env`, redo SETUP.md step 3 in Linux, then run `uv sync` and `uv run pytest -q`.

## Open questions

- Mask zones for templates that ship their own generation markers (Inkling): decide in M3, before running Inkling.
- The Hy4 license hasn't been reviewed (human). Do it before any Hy4 results are published.
- M4 open items: reference per framework environment, registry entries only from evidence, Unsloth's CPU fallback. See `docs/DECISIONS.md`.

## Findings so far

None yet.
