# Findings ledger

Every finding, its status, and the running count of harness errors caught. Each finding has its own folder (`findings/<id>/`). Fails link to findings through `findings/root_causes.json`, and the report counts fails by root cause.

## Findings

| ID | Framework | What | Status | Counted against the framework? |
| --- | --- | --- | --- | --- |
| [P-0001](P-0001/record.md) | axolotl 0.21.1 | With Qwen3's own template in thinking mode, the think-block opener (`<think>` and blank lines) is untrained; reasoning text is trained | difference: KD-0002, KD-0003 (thinking mode only); possible docs gap | No. Documented and tested behavior |
| [P-0002](P-0002/record.md) | axolotl 0.21.1 | An empty assistant turn's end token is never labeled, silently | candidate; repro shrunk, issue drafted, not posted | Yes, as a candidate, once confirmed upstream |
| [P-0003](P-0003/record.md) | trl 1.18.2 | Non-thinking cells judged rows that were never encoded in non-thinking mode | **harness error (ours)** | Never |
| [P-0004](P-0004/record.md) | unsloth 2026.11.10 (CPU backend) | On CPU, Unsloth's Qwen3 recipe trains on rows with a second `<|im_end|>`: stock TRL preparation appends it (TRL #3318 behavior); Unsloth's GPU-route preparation doesn't | candidate (CPU backend only); on the card as CPU-only, low impact; not shrunk. GPU run: rows equal the render, so CPU-only is confirmed. Card only, no upstream note; revisit after the kill test (the human's decision, 2026-11-25) | Yes, as CPU-backend behavior; not as the GPU recipe's |
| [P-0005](P-0005/record.md) | unsloth 2026.11.10 | `train_on_responses_only` silently drops `SFTConfig.pad_to_multiple_of` (class: config value silently ignored) | candidate, **reproduced on GPU** (T4; rule 4 cleared). Labels not affected; loss the same up to float rounding (CPU). Repro shrunk (30 lines; fails on 2026.11.12 and on unsloth-zoo main). **Posted 2026-11-25:** [unslothai/unsloth#19567](https://github.com/unslothai/unsloth/issues/19567) (filed in the main repo, as unsloth-zoo#698 asks); awaiting a maintainer. Explains the 15 inconclusive special-tokens cells of the GPU run, and pins the 8 special-tokens width fails of the M4 calibration GPU run | Yes, as a candidate, once a maintainer confirms |
| [P-0006](P-0006/record.md) | unsloth 2026.11.10 (GPU) | Non-thinking mode: `train_on_responses_only` trains the last turn's empty think block, which the reference puts on the prompt side as a pre-fill | **closed: not_configurable** (the human's decision, 2026-11-25). Unsloth documents non-thinking training via the `/no_think` soft switch; hard-switch training is not configurable | No. Its cells are not judged |

## Harness errors caught: 11

These are our own errors that produced, or could have produced, a wrong or unsupported verdict: in the reference, a check, the engine, an adapter, a test, or the evaluation design. The first entries are taken from the decision log. Add each new one here when it is caught. A harness error never counts against a framework.

| # | Date | What was wrong | Caught by | Fixed |
| --- | --- | --- | --- | --- |
| H-1 | 2026-11-24 | A pass without its self-test could be counted | M2 skeptic review | `docs/DECISIONS.md`, "M2 skeptic review: five bugs fixed" (1) |
| H-2 | 2026-11-24 | The reference cache ignored dict key order, so conversations differing only in key order shared stale token ids | M2 skeptic review | same entry (2) |
| H-3 | 2026-11-24 | The reference trusted templates that ignore `add_generation_prompt`, making the assistant header must-train | M2 skeptic review | same entry (3) |
| H-4 | 2026-11-24 | The `false`→`False` planted fault crashed on templates that refuse string arguments, making cells inconclusive | M2 skeptic review | same entry (4) |
| H-5 | 2026-11-24 | No planted fault guarded the padding part of `t1.special_tokens` | M2 skeptic review | same entry (5) |
| H-6 | 2026-11-24 | `test_template_breaks_match_the_references_prefix_breaks` was vacuous: it could never fail | M2 skeptic review | same entry ("also found") |
| H-7 | 2026-11-24 | The reference silently assumed thinking mode for Qwen3, which caused part of P-0001's fails | P-0001 skeptic | Declared serving mode, `3198068` |
| H-8 | 2026-11-24 | `t1.truncation` recorded "the row matches no truncation mode" when only the loss was wrong | P-0003 skeptic | `001bfbe` |
| H-9 | 2026-11-24 | P-0003: non-thinking cells judged rows the framework encoded in its default (thinking) mode | P-0003 triage and skeptic | Frameworks get their documented mode setting in non-default modes (see `docs/DECISIONS.md`) |
| H-10 | 2026-11-25 | `findings/P-0005/repro.py` loaded the model 4-bit but attached no LoRA, unlike the recommended notebook. On the GPU route the model is really 4-bit, so `SFTTrainer` refused ("You cannot perform fine-tuning on purely quantized models"), and both GPU repro runs crashed. On CPU, Unsloth loads 16-bit weights, so it never showed. No verdict was affected; the kit recorded both runs as not completed, not as "dropped" | The human, reading the GPU results zip | `get_peft_model` before the trainer, as the notebook does; re-checked on CPU (unchanged result) |
| H-11 | 2026-11-25 | On Unsloth's GPU route the adapter read the collator in the state `SFTTrainer.__init__` leaves: it ends with `model.for_inference()`, so the tokenizer pads on the left, while `train()` calls `model.for_training()` first and pads on the right. GPU batches and the declaration would have recorded left padding that training never uses. No verdict was affected: the check accepts either side, and the padding part never got that far | The skeptic, on the GPU run | The adapter makes the same `for_training` call first (`_enter_training_mode`); CPU re-run identical, 96 of 96 |
