# Gate C: does it generalize, and does the improver improve?

**Questions:**
1. Does the frozen suite catch confirmed bugs in releases that came out after the freeze, and hold up on a framework the loop never saw?
2. Do later writer versions produce more accepted checks per dollar than earlier ones on the same fresh probe set?

| # | Criterion | Evidence |
| --- | --- | --- |
| C1 | Freeze tag and date recorded before any counted release | Git tag + `docs/DECISIONS.md` |
| C2 | Frozen suite catches more maintainer-confirmed post-freeze bugs than v0 | `runs/prospective.csv` with links |
| C3 | Transfer: hidden-vault recall on the held-out framework improves from v0 to frozen | Eval results for that framework |
| C4 | Recursion: accepted checks per dollar rise across writer versions on the same probe set (used once, then retired) | Recursion table and plot |
| C5 | False alarms still 0 | `runs/eval/history.csv` |
| C6 | `eval-auditor` finds no leakage or confounds | Auditor report |

## What you may claim

| Criteria met | Claim |
| --- | --- |
| C1, C2, C3, C5, C6 | "The suite **generalizes**" |
| All of the above + C4 | "The improver **improves**": the self-improving claim |
| C1-C3, C5, C6 without C4 | "**Self-extending**, and it generalizes" |

## Decision

Record the exact claim you've earned in `docs/DECISIONS.md` and in the README. Then go to Phase 7.

Run `/gate-check C` first.
