---
name: start-phase
description: Start a Parity phase. Reads the phase playbook, status and gate, then proposes a plan without writing code.
argument-hint: "[phase-number 0-7]"
disable-model-invocation: true
---

We are starting Parity phase $0.

1. Read `docs/STATUS.md`, `docs/DECISIONS.md`, and the playbook in `docs/phases/` whose filename starts with `phase-$0-`.
2. If the phase ends in a gate, read the matching file in `docs/gates/`.
3. Check the previous phase's exit criteria against the repo. If anything is missing, list it and stop.
4. Propose a plan for this phase:
   - Goal in one sentence, and what "done" means (copy the playbook's Definition of done).
   - Milestones in order, each one ending with something runnable and tested.
   - For each milestone: files to create or change, tests to add, and the command that proves it works.
   - Risks and how we'll notice them early.
5. **Do not write or edit code yet.** Wait for my approval of the plan.
6. After I approve: update `docs/STATUS.md` (current phase, next actions) and commit only that file.
