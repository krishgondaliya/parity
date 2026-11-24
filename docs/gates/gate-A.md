# Gate A: is there anything real to find?

**Question:** across 3 recent releases × TRL, Unsloth and Axolotl, did Parity find at least one silent bug that a maintainer confirms?

Every criterion needs an evidence file. No file means not met.

| # | Criterion | Evidence |
| --- | --- | --- |
| A1 | Targets and calibration model pinned with exact revisions | `docs/DECISIONS.md` entry |
| A2 | Calibration fully green: every clean run passes, every planted fault caught | `runs/calibration-<date>.md` |
| A3 | All 432 runs completed | `runs/phase0/summary.md` |
| A4 | Every fail triaged, with the skeptic's verdict attached | `findings/*/record.md` |
| A5 | At least 1 issue posted by you, with a standalone repro | `findings/*/issue.md` + URL in the record |
| A6 | **At least 1 maintainer confirmation** (a reply acknowledging it, a label, or a fix) | Link + quote in `record.md`, status `confirmed` |
| A7 | No retractions, or each retraction explained and its check fixed | Ledger + `docs/DECISIONS.md` |

## Decision

- **Pass (A1-A7 met):** go to Phase 1.
- **Pending (everything met except A6, no maintainer reply yet):** extend once, up to 2 weeks, with one polite follow-up. Then decide.
- **Fail (zero real candidates after triage, or every candidate rejected by maintainers):** stop Parity and switch to Crashproof Rollouts (`docs/plan/04-decision-history.md`). Write up what Parity checked and found clean. That's still a public, honest result.

Run `/gate-check A`, then record your decision in `docs/DECISIONS.md`.
