# Status

Keep this file short and true. `/wrap-session` updates it at the end of every session.

- **Current phase:** Phase 0, kill test. M0, M1 and M2 are done. The M1 platform spike moved development to WSL2 Ubuntu and passed there with real installs. M3 is next.
- **Last updated:** 2026-11-23
- **Gates:** A not started · B not started · C not started

## Done

- Plan written, starter kit installed, setup checked.
- Guard hook works on Windows (Bash and PowerShell) and has tests (`tests/test_guard.py`).
- M0: targets and the Qwen3 calibration model pinned by commit SHA (`docs/DECISIONS.md`).
- M2, done early: the 8 fixtures and the Tier 1 template reference, cross-checked against TRL's Qwen3 training template. 136 tests pass.
- Design rules logged: mask zones, the known-differences registry (KD-0001), template properties, and a Tier 3 train/serve note.
- M1 platform spike on native Windows: Axolotl can't install (it needs Linux-only `triton`), so development moves to WSL2 Ubuntu (`docs/DECISIONS.md`).
- Repo cloned into `~/parity` in WSL2 Ubuntu; `uv sync` done and 136 tests pass there.
- M1 platform spike re-run in WSL2 with real installs. TRL, Axolotl, Unsloth, verl and SkyRL (`[skyrl-train]`) each install in their own environment and import their data path. Versions match the Windows resolutions. Together the five environments take 13.3 GiB.
  - One catch: the plain PyPI SkyRL install fails on Linux because the `flash-attn` sdist won't build. It works with SkyRL's own build settings.
  - Logged in `docs/DECISIONS.md`; evidence in `docs/evidence/2026-11-23-m1-wsl-spike/`.

## Next actions

1. Start Claude Code from `~/parity`, not from the Windows checkout under `/mnt/c` (the spike session still ran from there, and the Windows copy is now a commit behind).
2. M3.

## Open questions

- Mask zones for templates that ship their own generation markers (Inkling): decide in M3, before running Inkling.
- The Hy4 license hasn't been reviewed (human). Do it before any Hy4 results are published.
- M4 open items: reference per framework environment, registry entries only from evidence, Unsloth's CPU fallback, SkyRL's install route (plain PyPI fails on Linux). See `docs/DECISIONS.md`.

## Findings so far

None yet.
