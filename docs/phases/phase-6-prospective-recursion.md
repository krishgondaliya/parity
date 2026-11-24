# Phase 6: Prospective and recursion tests

**Goal:** earn (or honestly decline) the words "generalizes" and "self-improving", using evidence the suite couldn't have been tuned to.

**Ends with:** Gate C (`docs/gates/gate-C.md`).

## Milestones

### M1. Freeze
```
git tag suite-freeze-<date>
```
Record the freeze date in `docs/DECISIONS.md`. From now on, the loop keeps running, but the frozen tag is what gets judged.

```
/start-phase 6
```

### M2. Prospective tracking
Done when, for every release after the freeze date, you have recorded how many maintainer-confirmed bugs the frozen suite caught vs v0. Use `runs/prospective.csv`: release, date, frozen catches, v0 catches, confirmation links.

### M3. Transfer test
Done when hidden-vault recall on the transfer framework (never seen by the loop) has been scored for v0 and for the frozen suite.

### M4. Recursion test
Done when each archived writer version got the same **probe set** once, and accepted checks per dollar were recorded for each version.

- **Probe set:** 10 bug-class descriptions held back from the open vault. Descriptions only, never the pairs.
- Retire the probe set after use. A new test needs a new probe set.

```
Run the recursion protocol: for each archived writer version W1..WN, give the same probe set once with the same budget,
pass proposals through the gatekeeper, and record accepted checks per dollar. Plot it against version number.
```

### M5. Gate C
```
/gate-check C
```

## Definition of done

- [ ] Freeze tag and date recorded
- [ ] Prospective table covering every post-freeze release, with confirmation links
- [ ] Transfer recall scored (v0 vs frozen)
- [ ] Recursion table and plot; probe set retired
- [ ] `eval-auditor` report; Gate C record written and decided

## What you may claim

- Passing prospective catches and transfer: "the suite **generalizes**".
- Also passing recursion: "the improver **improves**". That's the self-improving claim.
- Everything except recursion: say "**self-extending**". It's still a strong, honest result.
