---
name: gate-check
description: Collect the evidence for a Parity gate (A, B or C) and write a gate record. The human makes the decision.
argument-hint: "[A|B|C]"
disable-model-invocation: true
---

Run the gate check for Gate $0.

1. Read `docs/gates/gate-$0.md`. For every criterion, find the evidence file it names. If the evidence doesn't exist, the criterion is **not met**: never infer it.
2. For Gates B and C, ask the `eval-auditor` subagent to audit the evaluation for leakage, suite-size confounds and unequal budgets, and include its findings.
3. Write `docs/gates/records/gate-$0-<date>.md` with:
   - A table: criterion, met or not met, evidence path.
   - The auditor's findings (B and C only).
   - The claim the evidence supports, worded exactly as allowed under "What you may claim" in `docs/plan/02-detailed-blueprint.md`.
   - A recommendation (pass, fail or rerun) with reasons.
4. **Stop and ask me for the decision.** After I decide, add an entry to `docs/DECISIONS.md` and update `docs/STATUS.md`.
