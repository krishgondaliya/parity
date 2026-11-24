# Status

Keep this file short and true. `/wrap-session` updates it at the end of every session.

- **Current phase:** Phase 0, kill test (milestone M0: pick and pin targets)
- **Last updated:** 2026-11-23
- **Gates:** A not started · B not started · C not started

## Done

- Plan written (`docs/plan/`)
- Starter kit installed
- Setup checked: git, uv, gh (logged in), uv Python 3.12, private GitHub repo, `.env` exists and is git-ignored
- Guard hook tested on PowerShell: blocks `gh issue create` and the hidden vault path
- Phase 0 plan approved: the playbook's milestones plus an added M1 platform spike (M0 targets, M1 platform spike, M2 fixtures + reference, M3 engine + 6 checks, M4 adapters, M5 calibration, M6 432 runs, M7 triage, M8 Gate A)

## Next actions

1. M0: propose 5 recent open-weight releases, pick 3, pin them and the Qwen3 calibration model by commit SHA in `docs/DECISIONS.md`.
2. M1: platform spike. One pinned uv env per framework under `envs/`; find out whether Unsloth (no NVIDIA GPU here) and Axolotl (Linux-first) run natively, in WSL2, or need a fallback.
3. Human: commit the Windows guard-hook fixes (`.claude/hooks/guard.py`, `.claude/settings.json`), still uncommitted.

## Open questions

- Which 3 recent open-weight releases are the targets? (M0, in progress)
- Unsloth without an NVIDIA GPU: Colab/Kaggle, or only its chat-template functions? (M1)
- Axolotl: native Windows or WSL2 Ubuntu-22.04? (M1)

## Findings so far

None yet.
