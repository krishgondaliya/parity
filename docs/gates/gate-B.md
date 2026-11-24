# Gate B: does the loop actually improve the suite?

**Question:** at equal suite size and budget, does the self-improving loop catch more hidden real bugs than the frozen suite, a loop without the breaker, and human-directed checks, with zero false alarms?

| # | Criterion | Evidence |
| --- | --- | --- |
| B1 | `suite-v0` frozen and scored before the loop ran | Git tag + `runs/eval/<date>-v0.json` |
| B2 | Hidden vault outside the repo; guard hook tested; no agent prompt or log contains hidden content | Hook test + `eval-auditor` report |
| B3 | Loop iterations logged (writer version, model, cost, gate results) | `runs/loop/` |
| B4 | False alarms = 0 for every suite version being claimed | `runs/eval/history.csv` |
| B5 | Loop recall at equal suite size k beats B1, B2 and B3, with non-overlapping Wilson intervals | Comparison report in `runs/eval/` |
| B6 | Gain present in all 3 seeds | Per-seed results |
| B7 | Budgets comparable: API dollars for the loop, calendar window and logged hours for B3 | Budget table in the gate record |
| B8 | Transfer framework never touched by the loop | `eval-auditor` report |

## What you may claim if it passes

"The loop improves the suite": the evidence for this is B4-B6.

## Decision

- **Pass:** go to Phase 5.
- **Rerun:** if intervals overlap only because the hidden vault is small, grow the vault (Phase 3 M1-M3) and rescore. Never tune the loop while looking at hidden results.
- **Fail:** keep the hand-written suite, publish the negative result honestly ("the loop did not beat human-directed checks at equal budget"), and continue with Phase 5 without the self-improvement claim.

Run `/gate-check B`, then record your decision in `docs/DECISIONS.md`.
