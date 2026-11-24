---
name: run-eval
description: Record and compare a Parity suite version's hidden-vault score from the aggregate numbers the human's evaluation run printed (Phase 3+). Claude Code never runs hidden-vault evaluation itself.
argument-hint: "[suite-version, e.g. v0, v3, B2]"
---

Record the evaluation of suite $0, following `docs/plan/02-detailed-blueprint.md` (section "Evaluation protocol") and the Phase 3 prerequisite in `docs/DECISIONS.md`.

1. **The human runs hidden-vault evaluation, not you.** It runs as the separate vault user, through the one wrapper that prints only aggregate numbers, and every run is logged. Don't run it, don't try to read the vault, and don't switch users. If the numbers for $0 aren't in front of you, ask the human to run the wrapper for $0 and paste what it printed.
2. Check that the output holds **aggregates only**: hidden-vault recall with a 95% Wilson interval, recall at equal suite size k, the false-alarm count on the calibration set, the fixed-mutant score and the suite size. If it shows anything per-pair (names, bug classes, diffs), stop, don't save it, and tell the human.
3. Save the result to `runs/eval/<date>-$0.json` and append one line to `runs/eval/history.csv` (version, date, n, recall, low, high, false alarms, suite size). Include the wrapper's run-log entry so the number can be traced to one logged run.
4. Compare against v0 and the previous version. Say plainly whether the change is outside the intervals or within noise.
5. If false alarms are above 0, that version doesn't count. Say so first.

You may run the evaluation harness yourself only on `vault/open/`, for example to test it.
