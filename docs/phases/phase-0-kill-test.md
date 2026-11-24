# Phase 0: Kill test

**Goal:** find out, cheaply, whether real silent bugs exist in how popular frameworks prepare training data for recent open-weight models.

**Ends with:** Gate A (`docs/gates/gate-A.md`). At least 1 maintainer-confirmed bug means continue; zero means switch to Crashproof Rollouts.

**Suggested time box:** about 3 weeks, CPU only, ~$0.

## Scope (keep it this small)

- **3 models:** the 3 most recent major open-weight releases. At least one uses reasoning (think) blocks and at least one supports tool calling.
- **3 frameworks:** TRL, Unsloth, Axolotl. Data path only (`encode`), no training.
- **6 Tier 1 checks:** template parity, assistant mask, end of turn, special tokens, truncation, multi-turn prefix.
- **8 fixtures** (see the blueprint).
- **Total:** 3 × 3 × 6 × 8 = 432 check-fixture runs.
- **Scrappy is fine.** Phase 1 turns this into a proper package. Prioritize correct and tested over pretty.

## Milestones

### M0. Pick the targets
Done when the 3 models plus the calibration model (Qwen3 family) are listed in `docs/DECISIONS.md` with exact revisions.

```
Search for the most recent major open-weight model releases (last ~2 months) that support chat fine-tuning.
Propose 5 candidates with: org/name, release date, whether they use reasoning blocks, whether they support tool calls,
and whether the tokenizer/chat template is downloadable without accepting a gated license.
Recommend 3. Don't download anything yet.
```

### M1. Fixtures and references
Done when 8 fixtures exist, and the reference produces token ids plus an assistant-token mask for each fixture on the calibration model, with tests.

```
/start-phase 0
```
Then, after approving the plan:
```
Build the 8 fixtures and the Tier 1 reference in src/parity/references/template.py.
Reference tokens come from the model's own apply_chat_template.
Reference assistant spans: use return_assistant_tokens_mask when the template has {% generation %} markers;
otherwise compute spans by rendering the conversation prefix with and without each assistant turn and diffing.
Cross-check against TRL's training template where one exists for that family. Write tests on the calibration model.
```

### M2. The 6 checks, each with its planted fault
Done when each check has run, plant and tests, following the `/new-check` contract.

```
/new-check t1.template_parity
```
(Repeat for `t1.assistant_mask`, `t1.end_of_turn`, `t1.special_tokens`, `t1.truncation` and `t1.multiturn_prefix`.)

### M3. Three adapters (encode only)
Done when each adapter returns exactly what its framework would train on, through the framework's public path.

```
/new-adapter trl
/new-adapter axolotl
/new-adapter unsloth
```
**Check this first for Unsloth:** it has historically needed an NVIDIA GPU just to import. If it won't import on your laptop, run that adapter in a free Colab or Kaggle GPU session, or test the specific chat-template functions it uses, and log the choice in `docs/DECISIONS.md`.

### M4. Calibration
Done when `runs/calibration-<date>.md` is fully green: every clean run passes and every planted fault is caught.

```
/calibrate all
```

### M5. The 432 runs
Done when the verdict files exist for all runs, plus a summary table (model × framework × check: pass / fail / inconclusive).

```
Run all Tier 1 checks for the 3 target models × 3 frameworks × 8 fixtures. Save verdicts under runs/phase0/.
Write runs/phase0/summary.md with a model × framework × check table. Don't interpret fails yet.
```

### M6. Triage, shrink, draft
Done when every fail has a record, and every real candidate has a repro and a reviewed draft.

```
/triage runs/phase0/<verdict-file>.json
/shrink-repro P-0001
/draft-issue P-0001
```
Then **you** read `findings/P-0001/issue.md` and the repro, post the issue by hand, and paste the URL into the record.

### M7. Wait, respond, decide
Done when maintainers have responded or the waiting period has passed. Then: `/gate-check A`

## Definition of done

- [ ] Targets and calibration model pinned in `docs/DECISIONS.md`
- [ ] 6 checks with planted faults and tests; calibration fully green
- [ ] 3 adapters through the public path, with mapping notes
- [ ] All 432 runs done, with a summary table
- [ ] Every fail triaged; real candidates shrunk, skeptic-reviewed, drafted, posted by you
- [ ] Gate A record written and decided

## Pitfalls

- **Most early fails are ours.** Expect adapter mistakes first. That's normal: fix, re-run, move on.
- **Template disagreements are findings too.** If a model's tokenizer template disagrees with its model card, report it to the model team, not the framework.
- **Don't scale before calibration is green.** A wrong check run 432 times is just 432 wrong answers.
- **Be patient with maintainers.** Give them time. One polite follow-up after about a week is fine.
