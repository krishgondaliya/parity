# Phase 7: The RL seam, then go further

**Goal:** add Tier 3 checks for the place where rollouts and training meet (sampled tokens, logprobs, serving format), run them on a small rented GPU, then decide what comes after.

**Ends with:** Tier 3 verdicts on the card for verl and SkyRL.

## Budget

- Rent one GPU only for active sessions, and shut it down after each one. A small model (e.g. Qwen3 0.6B) is enough.
- Rough budget: under $100 for the phase. Log spend in `docs/STATUS.md`.

## Milestones

### M1. GPU session script
Done when one command spins up the rented machine's environment (Docker or a uv lockfile), runs a Tier 3 job and copies the verdicts back. The repo stays the source of truth.

```
/start-phase 7
```

### M2. Three Tier 3 checks (each with its planted fault)
```
/new-check t3.token_in_token_out
/new-check t3.logprob_parity
/new-check t3.serving_parity
```
- **Token in, token out:** the trainer trains on exactly the sampled token ids. Planted fault: retokenize the decoded text.
- **Logprob parity:** trainer logprobs vs rollout-engine logprobs on the same tokens, judged by the noise-floor rule. Planted fault: the wrong temperature. Note that verl already logs mismatch metrics; Parity's value is the calibrated verdict.
- **Serving parity:** a deployed chat endpoint builds the same tokens as the training format. Planted fault: a missing generation prompt.
  - Known train/serve mismatches it must cover (`docs/DECISIONS.md`, Tier 3 notes):
    - TRL trains Qwen3 with think blocks that serving strips.
    - Templates whose generation prompt pre-fills tokens that history turns render without. DeepSeek V4.1 pre-fills `<think>`; Qwen3 in non-thinking mode pre-fills an empty think block. See the model-level property "pre-fill missing from history".

### M3. Runs and card
Run on verl and SkyRL with 1-2 recent releases, and add a Tier 3 column to the card with its scope stated.

## Then: go further (pick one, with evidence)

- **Keep it open:** keep the Day-0 lab running, write up the findings and the self-improvement results, and apply with the ledger as your portfolio.
- **Startup path:** offer private scans of a team's own configs and data. The first outside team using it is the signal.
- **Back to the fallback:** if Tier 3 turns up rollout-reliability pain, Crashproof Rollouts becomes a natural next project.

Record the choice in `docs/DECISIONS.md`.
