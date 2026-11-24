---
name: new-check
description: Write a new Parity check together with its planted fault and tests. Use whenever a check is added or changed.
argument-hint: "[check-id, e.g. t1.end_of_turn]"
---

Write check `$0`, following `docs/plan/02-detailed-blueprint.md` (sections "Check contract and verdict schema", "Pass rules and tolerances", "Planted faults").

A check is done only when all of these exist:

1. `run(adapter, release, fixture) -> Verdict`, which compares the adapter's output against a reference the framework did not produce. Use an exact comparison wherever possible.
2. `plant(adapter) -> Adapter`, which returns the same adapter with this check's specific bug injected. Prefer a real bug someone already hit (see the planted-faults table).
3. Tests:
   - The check passes on the known-good calibration setup (no false alarm).
   - The check fails on its planted fault (the self-test).
   - The engine marks the verdict `inconclusive` when the planted fault is missed.
4. A verdict JSON matching `parity.verdict/v0`, with evidence: the first divergence index and 8 decoded tokens on each side.
5. A one-paragraph docstring: the bug class it catches, the reference used, and the pass rule.

Never loosen a tolerance to make a test pass. If a numeric threshold is involved, measure the noise floor and log it in `docs/DECISIONS.md`.
