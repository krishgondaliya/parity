# Findings ledger

Every finding, its status, and the running count of harness errors caught. Each finding has its own folder (`findings/<id>/`). Fails link to findings through `findings/root_causes.json`, and the report counts fails by root cause.

## Findings

| ID | Framework | What | Status | Counted against the framework? |
| --- | --- | --- | --- | --- |
| [P-0001](P-0001/record.md) | axolotl 0.21.1 | With Qwen3's own template in thinking mode, the think-block opener (`<think>` and blank lines) is untrained; reasoning text is trained | difference: KD-0002, KD-0003 (thinking mode only); possible docs gap | No. Documented and tested behavior |
| [P-0002](P-0002/record.md) | axolotl 0.21.1 | An empty assistant turn's end token is never labeled, silently | candidate; repro shrunk, issue drafted, not posted | Yes, as a candidate, once confirmed upstream |
| [P-0003](P-0003/record.md) | trl 1.18.2 | Non-thinking cells judged rows that were never encoded in non-thinking mode | **harness error (ours)** | Never |
| [P-0004](P-0004/record.md) | unsloth 2026.11.10 (CPU backend) | On CPU, Unsloth's Qwen3 recipe trains on rows with a second `<|im_end|>`: stock TRL preparation appends it (TRL #3318 behavior); Unsloth's GPU-route preparation doesn't | candidate (CPU backend only); on the card as CPU-only, low impact; not shrunk; revisit after the GPU run | Yes, as CPU-backend behavior; not as the GPU recipe's |
| [P-0005](P-0005/record.md) | unsloth 2026.11.10 | `train_on_responses_only` silently drops `SFTConfig.pad_to_multiple_of` (class: config value silently ignored) | candidate; confirm on the GPU run before any draft | Yes, as a candidate, once confirmed |

## Harness errors caught: 9

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
