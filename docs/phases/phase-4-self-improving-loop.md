# Phase 4: Self-improving loop

**Goal:** build Parity's own writer and breaker agents plus the executed gates, run the loop, and compare it fairly against three baselines on the hidden vault.

**Ends with:** Gate B (`docs/gates/gate-B.md`): the loop beats all three baselines at equal suite size and budget, with zero false alarms.

## Two kinds of agents. Don't mix them up.

| | Claude Code (your dev tool) | Parity's loop agents (product code) |
| --- | --- | --- |
| What | Helps you build Parity | Part of Parity: `src/parity/loop/` |
| Runs | Interactively, in your terminal | As a program, via an LLM API, with a budget cap |
| Writes checks for | Baseline B3 (human-directed) | The loop's suite versions vN |
| Sees the hidden vault | Never | Never |

## Milestones

### M1. Gatekeeper (plain code, no LLM)
Done when a proposed check (a code file plus its planted fault) is accepted or rejected by the gates below, with the failing gate named. Tests cover every gate.

1. **No false alarm:** passes on the calibration set, 3 runs each.
2. **Self-test:** catches its own planted fault.
3. **Catches its target:** a real pair from `vault/open/`, or the breaker's breakage.
4. **Redundancy filter:** catches something the current suite misses.

```
/start-phase 4
```

### M2. Breaker agent
Done when it produces "breakages" (small patches to a framework's data path, plus the expected symptom), runs them against the suite, and lists the misses. It runs in its own sandboxed copy of the framework, never your working env. It gets no access to the transfer framework or the hidden vault.

### M3. Writer agent
Done when, given the misses plus public context (issues, changelogs, template diffs), it proposes checks and planted faults that go through the gatekeeper. It may edit its own prompt and tools between iterations, and each edit is archived as a new writer version.

### M4. Budget, logging, versions
- A hard dollar cap per iteration (`LOOP_BUDGET_USD_PER_ITERATION`), enforced in code.
- Logged per proposal: writer version, model ID, cost, gate results with reasons, wall-clock time.
- Accepted checks merge as suite version vN (git tag `suite-vN`).

### M5. Baselines and seeds
- **B1:** the frozen v0.
- **B2:** the loop with the breaker turned off.
- **B3:** human-directed checks, written by you with Claude Code over the same calendar window. Log your hours.
- **Seeds:** 3 independent loop runs from v0.

```
/run-eval v1
/run-eval B2-final
/run-eval B3-final
```

### M6. Gate B
```
/gate-check B
```

## Definition of done

- [ ] Gatekeeper with tests for all 4 steps
- [ ] Breaker and writer agents with budget caps, logs, and archived writer versions
- [ ] Loop iterations scored with `/run-eval`; `runs/eval/history.csv` filled in
- [ ] B1, B2 and B3 scored, 3 seeds run
- [ ] `eval-auditor` report attached; Gate B record written and decided

## Pitfalls

- **Suite-size confound.** Always compare at equal k. The redundancy filter helps keep this honest.
- **Leakage.** If any agent prompt or log ever contains hidden-vault content, that run is void. Re-split the vault and rerun.
- **Overfitting to the breaker.** The hidden vault is the referee, not the breaker's mutants.
- **Spend.** Watch the API bill weekly, and stop at the cap even mid-iteration.
