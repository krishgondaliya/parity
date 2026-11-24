# Worked example: what Parity does on release day

As of Nov 23, 2026. From the chat walkthrough.

## First, a scope note

Parity only applies to **open-weight** models. Gemini 4 Neon (announced Nov 18, 2026) can't be a Parity target: it ships through Google's Gemini API, AI Studio, Antigravity and the Gemini app ([Google](https://blog.google/innovation-and-ai/models-and-research/gemini-models/introducing-gemini-4-neon/)), so there are no weights for TRL or Unsloth to fine-tune.

So the example below pretends Google drops an open-weight cousin on Hugging Face. Call it **Neon-8B** (made up). Framework names A, B and C are placeholders. The failure shown is hypothetical, but it's a real bug class (Plumbline caught a missing end-of-turn label in a real release).

## Day 0, 10am: Neon-8B goes public

**Step 1: Parity notices.** The watcher sees a new model repo and downloads only the small files: tokenizer, config and chat template (a few MB, not the full weights).

**Step 2: Make a test conversation.**

> User: What's 2+2?
> Assistant: 4

**Step 3: Ask the official template what the training data should look like.** Say the template renders that chat as (made-up format):

`<|user|> What's 2+2? <|end|> <|assistant|> 4 <|end|>`

The rule for chat fine-tuning: **train only on the assistant's part, including the `<|end|>` after it.** That last token is how the model learns to stop talking.

**Step 4: Ask each framework to prepare the same chat for training, then compare.**

| Token | Official answer: train on it? | Framework A | Framework B |
|---|---|---|---|
| `<|user|>` | no | no | no |
| What's 2+2? | no | no | no |
| `<|end|>` (after the question) | no | no | no |
| `<|assistant|>` | no | no | no |
| 4 | **yes** | yes | yes |
| `<|end|>` (after the answer) | **yes** | yes | **no** |

Framework B trains on "4" but skips the final `<|end|>`. Anyone who fine-tunes Neon-8B with it gets a model that **never learns to stop**: it rambles forever, and the loss curve still looks perfectly normal.

**Step 5: Make sure the check itself isn't lying.** Before trusting the result, Parity breaks a known-good setup on purpose the same way (it hides the end token) and confirms the check catches it.

**Step 6: Confirm and shrink.** Re-run in a clean environment, then cut it down to a ~15-line script anyone can paste and see fail in seconds.

**Step 7: You review and file.** Read the repro, then open one polite issue: *"[assistant-only loss] end-of-turn token not supervised for Neon-8B (Framework B v1.2)"*.

## Day 1: the report card goes up

| Neon-8B | Template | Mask | End of turn | Special tokens | Truncation | Multi-turn |
|---|---|---|---|---|---|---|
| Framework A | pass | pass | pass | pass | pass | pass |
| Framework B | pass | pass | **fail** | pass | pass | pass |
| Framework C | pass | pass | pass | pass | pass | inconclusive |

## A week later

Framework B ships a fix. Parity re-runs the check, the cell turns green, and the ledger shows the finding went **found → confirmed → fixed**. That ledger entry is the resume line.

## Where Tier 2 fits

After this, Parity builds a tiny 2-layer "Neon-shaped" model with random weights to check the math (gradient accumulation, save/resume, etc.) on a laptop, still without downloading the real 8B weights.
