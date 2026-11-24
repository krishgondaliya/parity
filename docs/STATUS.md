# Status

Keep this file short and true. `/wrap-session` updates it at the end of every session.

- **Current phase:** Phase 0, kill test. M0, M1 and M2 are done. M3 is in progress: TRL and Axolotl are done and re-run in both serving modes; Unsloth is next (not started, by instruction). Milestone numbers follow `docs/phases/phase-0-kill-test.md`.
- **Last updated:** 2026-11-24
- **Gates:** A not started · B not started · C not started

## Done

- Plan written, starter kit installed, setup checked. Guard hook tested on Windows and WSL2 (`tests/test_guard.py`).
- M0: targets and the Qwen3 calibration model pinned by commit SHA. M1: 8 fixtures and the Tier 1 template reference (`docs/evidence/golden-review.md`).
- M2: six Tier 1 checks with 21 planted faults (2 real-world) on an engine that enforces the self-test.
- **Serving mode** (from P-0001, after results): `TrainCfg.serving_mode`, recorded in every verdict. Primary results in thinking mode; every run also judged in non-thinking mode as a labeled secondary table (same encoded rows, second reference).
- **Root causes:** every fail links to a finding (`findings/root_causes.json`, exact evidence pins; differences through their KD entry). The report and Gate A count root causes, not cells.
- **Tool arguments:** token-level `history_false_as_python_tokens` plant guards every framework; `tool_arguments` is declared and verified by behavior (TRL passes through, Axolotl parses JSON), shown on the report card.
- **DeepSeek:** added tokens accepted as structural headers under the new guard (0 disagreements, 3 families). `multi_turn` and `reasoning` now compute; `empty_assistant` and `tool_call` still refuse.
- M3 re-run, both modes (`docs/evidence/2026-11-24-m3-rerun/`):
  - TRL 1.18.2: primary 40 pass, 4 KD-0001, 4 n/a. 21 fail verdicts from 2 root causes (KD-0001, P-0003), 0 untriaged.
  - Axolotl 0.21.1: primary 27 pass, 13 KD-0002/0003 (P-0001), 4 P-0002, 4 n/a, 0 inconclusive. 21 fail verdicts from 2 root causes, 0 untriaged.
- 423 tests pass in the core environment; 444 in TRL's and 443 in Axolotl's.

## Next actions

1. Human: run `findings/P-0002/repro.py` (it should exit 1), then post `findings/P-0002/issue.md` through Axolotl's issue form. Record the URL in the record.
2. Human: decide P-0003's fix: give frameworks their mode setting in secondary runs (preferred), or keep same rows and label such cells "not configured for this mode".
3. M3: Unsloth (`/new-adapter unsloth`), when you say go.

## Open questions

- P-0003 decision (above). Until then, TRL's 14 secondary fails are pinned to P-0003 as ours.
- The structural method's added-header widening is exercised only on DeepSeek (8 agreeing turns); the method as a whole covers 3 families. Accept, or ask for more coverage?
- `reasoning` has no non-thinking reading in our reference (it refuses), though TRL's template gives one. Fine as inconclusive for now.
- Possible docs gaps, not findings: Axolotl never says think tags go untrained by default; TRL's SFT docs never say how to pass `chat_template_kwargs`.
- Deferred to Phase 2 (kill test frozen at 8 fixtures): gpt-oss-20b and the `<|end|>` question; a truncation fixture.
- TRL drops rows left with no trainable token after truncation: rule logged, implemented when a fixture hits it.
- The Hy4 license hasn't been reviewed (human). Before Phase 3 (human): close the `docker` group and WSL interop routes; edit the vault-path lines in `vault/README.md`, SETUP.md and Phase 3 M3.

## Findings so far

- **P-0001** (Axolotl 0.21.1, Qwen3): only the think opener (`<think>` + blank lines) is untrained; reasoning text is trained. Known difference KD-0002/KD-0003, thinking mode only; possible docs gap.
- **P-0002** (Axolotl 0.21.1, Qwen3): an empty assistant turn's `<|im_end|>` is never labeled, silently. Candidate, low severity (tool-call turns not affected). Shrunk repro, reproduces on main; issue drafted and reviewed, **not posted**.
- **P-0003** (TRL 1.18.2, secondary only): ours, an evaluation-design artifact. TRL labels correctly when told the mode.
