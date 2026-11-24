# Status

Keep this file short and true. `/wrap-session` updates it at the end of every session.

- **Current phase:** Phase 0, kill test. M0 and M1 are done, and M3's install check was done early (the platform spike, which moved development to WSL2 Ubuntu and passed there with real installs). M2 is next. Milestone numbers follow `docs/phases/phase-0-kill-test.md`.
- **Last updated:** 2026-11-23
- **Gates:** A not started · B not started · C not started

## Done

- Plan written, starter kit installed, setup checked.
- Guard hook works on Windows (Bash and PowerShell) and has tests (`tests/test_guard.py`). It also blocks hidden-vault access from `~/parity` in WSL2 (checked 2026-11-23).
- M0: targets and the Qwen3 calibration model pinned by commit SHA (`docs/DECISIONS.md`).
- M1: the 8 fixtures and the Tier 1 template reference, cross-checked against TRL's Qwen3 training template. 136 tests pass.
- Design rules logged: mask zones, the known-differences registry (KD-0001), template properties, and a Tier 3 train/serve note.
- M3's install check, done early (the platform spike) on native Windows: Axolotl can't install (it needs Linux-only `triton`), so development moves to WSL2 Ubuntu (`docs/DECISIONS.md`).
- Repo cloned into `~/parity` in WSL2 Ubuntu; `uv sync` done and 136 tests pass there. `~/parity` is now the only working copy; the Windows checkout is retired.
- Platform spike re-run in WSL2 with real installs. TRL, Axolotl, Unsloth, verl and SkyRL (`[skyrl-train]`) each install in their own environment and import their data path. Versions match the Windows resolutions. Together the five environments take 13.3 GiB.
  - One catch: the plain PyPI SkyRL install fails on Linux because the `flash-attn` sdist won't build. Settled by a general rule: adapter environments install the way each framework's docs say, so SkyRL installs from its repo (`docs/DECISIONS.md`).
  - Logged in `docs/DECISIONS.md`; evidence in `docs/evidence/2026-11-23-m1-wsl-spike/`.
- Phase 3 prerequisite logged: the hidden vault moves to a separate Linux user before any pair enters it. `Bash(sudo *)` is denied in `.claude/settings.json`.

## Next actions

1. M2: the six Tier 1 checks, each with its planted fault. Plan proposed, not started.

## Open questions

- Mask zones for templates that ship their own generation markers (Inkling): decide in M2, before running Inkling.
- The Hy4 license hasn't been reviewed (human). Do it before any Hy4 results are published.
- M3 (adapter) open items: reference per framework environment, registry entries only from evidence, Unsloth's CPU fallback. See `docs/DECISIONS.md` (logged there under the old number, "M4").
- Before Phase 3: close the `docker` group and WSL interop routes to root (human), and update `/run-eval`, SETUP.md and `vault/README.md` to the new vault setup (the lines naming the vault path need the human, since the guard blocks them). See the Phase 3 prerequisite in `docs/DECISIONS.md`.

## Findings so far

None yet.
