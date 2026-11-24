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

## Definition of done

- [ ] 5 adapters, calibration green on all of them
- [ ] A lockfile per framework version, recorded in every verdict
- [ ] The full Tier 1 matrix on the public card
- [ ] Every fail triaged; the ledger is up to date
- [ ] The transfer framework is recorded in DECISIONS.md

## Pitfalls

- **RL frameworks prepare data differently for rollouts than for SFT.** Be explicit about which path each verdict covers.
- **Chasing every "difference".** Label documented behavior as `difference` and move on.
