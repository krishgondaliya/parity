# Predictions: TRL 1.18.2 on deepseek-ai/DeepSeek-V4.1-Flash-1017@08c8657f02291e11b6f6b63e0e16014f8e748518 (both serving modes)

Written from TRL's source and DeepSeek V4.1's template alone. Committed before TRL runs on DeepSeek V4.1. All 96 cells are graded: 48 primary (thinking) and 48 secondary (non-thinking).

**Not blind, stated plainly.** These were known when this file was written, and they are in the repo:
- **The support check** (`docs/evidence/2026-11-25-m5-env-support/`): no TRL environment builds `deepseek_v41`, so the model is the `deepseek_v4` stand-in (D1), recorded as a deviation.
- **The coverage preview** (`docs/evidence/2026-11-25-m5-coverage-preview/`): our reference refuses `tool_call` and `empty_assistant` on this model.
- **The DeepSeek comparison** (`docs/evidence/2026-11-25-deepseek-encoding/`): the Jinja template agrees with the model's own `encoding.py` on all 8 fixtures.
- **The adapter's pinned refusal** (`trl.no_training_template`) was written from TRL's source. It has never run against this template.

## Sources
- **TRL 1.18.2** (installed files byte-identical to the tag):
  - with `assistant_only_loss=True`, `SFTTrainer.__init__` calls `get_training_chat_template(processing_class)` (`sft_trainer.py` L998-L1001);
  - that returns a stored training template only for a template it knows, or `None` for one that already has `{% generation %}` markers. Otherwise it raises `ValueError("The chat template is not training-compatible (missing prefix-preservation or `{% generation %}` markers) and patching is not supported for this template. Please manually modify the chat template for training.")` (`chat_template_utils.py` L1513-L1516).
- **DeepSeek V4.1's `chat_template.jinja`** at the pinned revision has no `{% generation %}` markers (M0's reading, and our reference records `marker_check: no_markers` on every fixture). It isn't byte-identical to any template TRL ships (checked against `trl/chat_templates/*.jinja`, 2026-11-25).
- **The processing class:** the stand-in is a causal LM, so `model_accepts_images` is False and TRL loads an `AutoTokenizer` (`sft_trainer.py` L714-L718). That is also what a model id would give: the release declares `DeepseekV41ForCausalLM`.

## The recipe
- `src/parity/adapters/trl.py`, in `envs/trl-1.18.2`: a tiny `deepseek_v4` stand-in, the release's own tokenizer, `SFTConfig(assistant_only_loss=True)`.
- Non-thinking mode: per-row `chat_template_kwargs={"thinking": False}`.

## What the sources say
- The refusal depends only on the template, so it doesn't depend on the stand-in or on the serving mode (the template text is the same in both).
- `SFTTrainer.__init__` raises before any row is encoded. The adapter matches the exception exactly against `trl.no_training_template` and raises `FrameworkRefused`, and the runner records `not_supported` (D2).

## Predicted cells
- **Thinking (primary):** 44 `not_supported`, 4 n/a (the multi-turn prefix check on the four single-assistant fixtures, decided before TRL is asked). Our reference's refusals on `tool_call` and `empty_assistant` never come into play: the refusal comes first.
- **Non-thinking (secondary):** 38 `not_supported`, 10 n/a (`reasoning` in all six checks, by the mode rule; the multi-turn prefix check on the four single-assistant fixtures).

| Mode | Check | single_turn | multi_turn | system_prompt | tool_call | reasoning | empty_assistant | long_answer | non_ascii |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| thinking | five checks | not_supported | not_supported | not_supported | not_supported | not_supported | not_supported | not_supported | not_supported |
| thinking | t1.multiturn_prefix | n/a | not_supported | n/a | not_supported | not_supported | not_supported | n/a | n/a |
| non_thinking | five checks | not_supported | not_supported | not_supported | not_supported | n/a | not_supported | not_supported | not_supported |
| non_thinking | t1.multiturn_prefix | n/a | not_supported | n/a | not_supported | n/a | not_supported | n/a | n/a |

## Also predicted (not graded cells)
- Every `not_supported` verdict names `trl.no_training_template` with TRL's exact message, and records the declaration error. No planted fault runs.
- The support table for this pair says the release's architecture doesn't load in this environment (`deepseek_v41` isn't recognized), with the exact error.

## Where this could be wrong (named before the run)
- **If constructing the trainer fails earlier** (for example, the stand-in model or the tokenizer), the exception isn't the pinned one. Then the cells are harness errors (inconclusive), never `not_supported`.
- **If TRL's message differs from the pinned text by a single character,** the same applies, and the adapter's refusal pin is what to fix (rule 4).
