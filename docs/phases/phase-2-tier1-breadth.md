# Phase 2: Tier 1 breadth

**Goal:** all six Tier 1 checks across five frameworks, with self-tests enforced, producing a full Tier 1 matrix for the latest releases.

**Ends with:** the full Tier 1 matrix on the public card. This becomes the hand-written suite that Phase 3 freezes as v0.

## Milestones

### M1. Add verl and SkyRL adapters
Done when both adapters return what their SFT and RL data paths would train on, through their public entry points, and calibration is green.

```
/start-phase 2
```
After approval:
```
/new-adapter verl
/new-adapter skyrl
```
These are RL-first frameworks. Find their documented SFT path and their multi-turn rollout-to-training path, and adapt both if they differ. Log the choice in DECISIONS.md.

### M2. One locked environment per framework version
Done when every framework@version has its own lockfile under `envs/`, and runs record which lockfile they used.

### M3. The full matrix
Done when all 6 checks × 5 frameworks × the latest releases × 8 fixtures have run, the card shows the matrix, and calibration is green for every framework.

```
/calibrate all
Run the full Tier 1 matrix for the 3 most recent releases across all 5 frameworks. Rebuild the card.
```

### M4. Keep the findings flow moving
Every new fail goes through `/triage`, then `/shrink-repro`, the skeptic, `/draft-issue`, and you post it.

### M5. Pick the transfer framework now
Choose the one framework the self-improving loop will **never** see (for example verl), for the Phase 6 transfer test. Record it in `docs/DECISIONS.md`. It stays in the hand-written suite, but loop agents never get its source, adapter or verdicts.

### Deferred from Phase 0 (decide at the start of this phase)
The kill test is frozen at 8 fixtures and 432 runs. Real results exist, so changing its design now would be post-hoc. These open items wait for this phase (`docs/DECISIONS.md`, 2026-11-24):
- **A second calibration tokenizer for per-turn closing tokens.** No Phase 0 target has more than one stop token per turn, or different closing tokens on different turns. `openai/gpt-oss-20b` (ungated, Apache-2.0, tokenizer and template only) does: `<|return|>` for a final answer, `<|call|>` for a tool call, `<|end|>` for history. But `<|end|>` isn't one of its stop tokens, so the reference refuses its history turns today.
  - Open question: when is a closing token valid? Likely rule: if it's a declared stop token, or a documented history-form replacement of one, with the doc link recorded.
  - Evidence: `docs/evidence/2026-11-24-rule-coverage.md`.
- **A truncation fixture where modes disagree beyond `long_answer`.** For example, a multi-turn conversation long enough that a right cut falls inside an earlier assistant turn and a left cut drops the system prompt. Adding it changes the run count, so it starts here, not in Phase 0.

## Definition of done

- [ ] 5 adapters, calibration green on all of them
- [ ] A lockfile per framework version, recorded in every verdict
- [ ] The full Tier 1 matrix on the public card
- [ ] Every fail triaged; the ledger is up to date
- [ ] The transfer framework is recorded in DECISIONS.md

## Pitfalls

- **RL frameworks prepare data differently for rollouts than for SFT.** Be explicit about which path each verdict covers.
- **Chasing every "difference".** Label documented behavior as `difference` and move on.
