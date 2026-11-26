# Phase 5: Tier 2 numerics and Day-0 automation

**Goal:** add the six numeric checks on tiny random-weight models, and make new releases trigger runs automatically, so report cards publish within 48 hours.

**Ends with:** two real releases carded within 48 hours, numerics included.

## Milestones

### M1. Tiny model factory
Done when `src/parity/tiny/` builds a 2-layer, small-hidden-size, random-weight copy of any release's architecture from its config (via `from_config`), on CPU, in seconds, with tests on 3 architectures.

```
/start-phase 5
```

### M2. Six numeric checks (each with its planted fault)
```
/new-check t2.loss_parity
/new-check t2.grad_accumulation
/new-check t2.packing_isolation
/new-check t2.optimizer_step
/new-check t2.save_resume
/new-check t2.export_roundtrip
```
For each check that uses the noise-floor rule: measure σ_run (run-to-run variation) and δ_bf16 (the bf16-vs-fp32 gap) per framework × architecture, log them in `docs/DECISIONS.md`, and publish the ratio next to each verdict.

Planted faults to reuse:
- Grad accumulation: average the loss per micro-batch (the 2024 Transformers bug).
- Save/resume: don't restore the step counter.
- Export: transpose one expert's weight.

### M3. Day-0 automation
Done when a scheduled GitHub Actions job polls the watchlist, starts a per-framework matrix job for Tiers 1 and 2, and opens a **draft PR in your own repo** with the new verdicts. Your merge publishes the card.

```
Write .github/workflows/day0.yml: scheduled poll of a watchlist (model orgs on the Hugging Face Hub + framework releases),
then a matrix job per framework running Tier 1 + Tier 2 on CPU, then open a draft PR in this repo with the verdict files.
No posting to any other repo, ever. Document the watchlist in docs/DECISIONS.md.
```

**Process lesson from Phase 0 (M5 prep, 2026-11-25): check architecture support, not only the tokenizer.** Phase 0's M0 picked targets because their tokenizers load without repo code. Two of the three turned out to have an architecture no transformers release registers, and the third only the newest one, so most framework × target pairs could not build the model. The watcher's first step for a new release must therefore be a support check, per framework at its pinned version:
- does the tokenizer load (no remote code)?
- does the config resolve, and does a tiny `from_config` model build, with the framework's own registrations applied (import the framework first: Unsloth registers some architectures itself)?
- for a text-plus-vision release, which path does the framework document for a text-only chat fine-tune?

Record the answers as the release's "loads the release's architecture" table on the card, before any check runs. See `docs/DECISIONS.md`, 2026-11-25, "M5 prep".

### M4. The 48-hour test
Done when two real releases go from public release to a merged card update in under 48 hours each. Record the timestamps.

## Definition of done

- [ ] The tiny model factory is tested on 3 architectures
- [ ] 6 numeric checks with planted faults; noise floors measured and logged
- [ ] The Day-0 workflow opens draft PRs in your repo only
- [ ] Two releases carded within 48 hours (timestamps recorded)

## Pitfalls

- **Tiny models miss size-only bugs.** Say so on the card. Tier 3 covers the GPU seam later.
- **Thin margins.** If a noise-floor ratio lands between 3.4x and 4.4x, call it `inconclusive`, not pass or fail.
