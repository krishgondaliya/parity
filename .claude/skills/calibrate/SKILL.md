---
name: calibrate
description: Run every Parity check and planted fault on the known-good calibration model for each framework. Must pass before running on any new release.
argument-hint: "[framework-name or all]"
---

Calibrate checks for: $ARGUMENTS (default: all frameworks with an adapter).

1. Use the known-good calibration model (Qwen3 family, which has a TRL training template) and all 8 fixtures.
2. For each framework × check × fixture:
   - The clean run must pass.
   - The planted-fault run must fail.
3. Write `runs/calibration-<date>.md` as a table of framework × check: clean result, planted result, notes.
4. If anything is off, **stop and fix it first**: no release results count until calibration is fully green. Name the likely culprit: the check, the adapter, the fixture, or a real framework issue on the calibration model itself. A real issue goes to `/triage`.
5. Update `docs/STATUS.md` with the calibration status.
