---
name: run-eval
description: Score a Parity suite version or baseline on the hidden vault, reporting aggregates only (Phase 3+).
argument-hint: "[suite-version, e.g. v0, v3, B2]"
---

Evaluate suite $0, following `docs/plan/02-detailed-blueprint.md` (section "Evaluation protocol").

1. Run the evaluation harness command. It reads the hidden vault through `PARITY_HIDDEN_VAULT`. **Never** open, list or print hidden vault contents yourself.
2. The harness must print **aggregates only**: hidden-vault recall with a 95% Wilson interval, recall at equal suite size k, the false-alarm count on the calibration set, and the fixed-mutant score. No per-pair names, bug classes or diffs.
3. Save the result to `runs/eval/<date>-$0.json` and append one line to `runs/eval/history.csv` (version, date, n, recall, low, high, false alarms, suite size).
4. Compare against v0 and the previous version. Say plainly whether the change is outside the intervals or within noise.
5. If false alarms are above 0, that version doesn't count. Say so first.
