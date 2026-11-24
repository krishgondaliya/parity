---
name: triage
description: Triage a failing Parity verdict. Decides whether it is our bug, intended framework behavior, a model-template issue, or a candidate framework bug. Use on every fail before anything else.
argument-hint: "[path to verdict JSON or run id]"
---

Triage: $ARGUMENTS

Assume our code is wrong until proven otherwise. Go through these in order, and write down what you checked:

1. **Calibration:** is the check green on the known-good model for this framework? If not, it's our bug. Fix it, re-run, and stop here.
2. **Adapter:** does the adapter call the framework's public path exactly as a user would, with the documented option for this objective? Re-read the framework docs for that option.
3. **Fixture:** is the fixture valid for this model? (For example, tool calls on a model whose template has no tool support aren't a fair test.)
4. **Reference:** does the model's own `apply_chat_template` agree with its model card and any official renderer? If the references disagree with each other, it's a **model-release** finding, not a framework one.
5. **Intended behavior:** do the framework docs say this behavior is deliberate? If yes, label it `difference`, not `bug`.
6. **Duplicate:** search the framework's issues read-only (`gh search issues` or `gh issue list --search`) and link any matches.

Then:

- Write `findings/<next-id>/record.md` from `findings/_TEMPLATE.md`, with status `candidate` and every check above recorded.
- Ask the `skeptic` subagent to review the record and repro, and include its verdict in the record.
- Suggest `/shrink-repro` as the next step only if both you and the skeptic call it a real candidate.
