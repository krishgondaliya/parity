# Status

Keep this file short and true. `/wrap-session` updates it at the end of every session.

- **Current phase:** Phase 0, kill test. M0, M1 and M2 are done, and M3's install check was done early (the platform spike, which moved development to WSL2 Ubuntu). M3, the adapters, is next, after the human reviews M2. Milestone numbers follow `docs/phases/phase-0-kill-test.md`.
- **Last updated:** 2026-11-24
- **Gates:** A not started · B not started · C not started

## Done

- Plan written, starter kit installed, setup checked.
- Guard hook works on Windows (Bash and PowerShell) and has tests (`tests/test_guard.py`). It also blocks hidden-vault access from `~/parity` in WSL2 (checked 2026-11-23).
- M0: targets and the Qwen3 calibration model pinned by commit SHA (`docs/DECISIONS.md`).
- M1: the 8 fixtures and the Tier 1 template reference, cross-checked against TRL's Qwen3 training template.
- M2: the six Tier 1 checks (`src/parity/checks/tier1/`), with 15 planted faults (3 real-world), on a check engine that enforces the self-test (`src/parity/engine/`). Reviewed by the skeptic subagent; its five bugs are fixed.
  - Tested against two stand-in encoders on Qwen3 (`tests/stubs.py`; not adapters). With Qwen3's own template, every applicable verdict passes and every planted fault is caught.
  - With TRL's training template, template parity shows KD-0001 on exactly the 4 fixtures DECISIONS.md predicted, and everything else passes.
  - 351 tests pass. No framework has been run yet.
- Design rules logged: mask zones, the known-differences registry (KD-0001, now a data file), template properties, a Tier 3 train/serve note, the check engine rules, the Tier 1 pass rules, and the mask reference (prefix diff, cross-checked against generation markers; this settles the Inkling question).
- M3's install check, done early (the platform spike): Axolotl can't install on native Windows, so development moved to WSL2 Ubuntu. Re-run there with real installs: all five frameworks install and import their data path (13.3 GiB together). `~/parity` is the only working copy; the Windows checkout is retired.
- Adapter environments install the way each framework's docs say, and any deviation is recorded and treated as a triage suspect. SkyRL installs from its repo (Phase 2).
- Phase 3 prerequisite logged: the hidden vault moves to a separate Linux user before any pair enters it, and the human runs hidden-vault evaluation (`/run-eval`, Phase 3 and 4 docs and SETUP.md updated). `Bash(sudo *)` is denied in `.claude/settings.json`.

## Next actions

1. Human: decide the open questions from the M2 skeptic review (below), then M3.
2. M3: TRL, Axolotl and Unsloth adapters (encode only), each in an environment installed per the framework's docs. Each declares its template, label convention, truncation mode, batching and pad token from its own config, and maps `pad_to` onto its own collator.

## Open questions

- From the M2 skeptic review (human's call; details in `docs/DECISIONS.md`, 2026-11-24): DeepSeek V4.1 is fully inconclusive (its generation prompt pre-fills `<think>`, which history turns don't have); several rules have no planted fault guarding them on Qwen3; most planted faults are blatant; plus smaller design questions.
- M4 heads-up: if a framework skips `empty_assistant` on purpose and documents it, that's a known-differences entry; if it's undocumented, it's a candidate finding.
- The Hy4 license hasn't been reviewed (human). Do it before any Hy4 results are published.
- M3 (adapter) open items: reference per framework environment, registry entries only from evidence, Unsloth's CPU fallback, and the metadata format for install deviations. See `docs/DECISIONS.md` (the first three were logged under the old number, "M4").
- Before Phase 3 (human): close the `docker` group and WSL interop routes to root; edit the lines that name the vault path in `vault/README.md`, SETUP.md and Phase 3 M3. See the Phase 3 prerequisite in `docs/DECISIONS.md`.

## Findings so far

None yet.
