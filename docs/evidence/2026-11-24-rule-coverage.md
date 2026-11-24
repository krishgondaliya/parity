# Which rules the targets reach that Qwen3 can't (2026-11-24)

The M2 skeptic review found rules that can never fire on Qwen3, the calibration model. This note checks whether the three pinned Phase 0 targets reach them, using their own tokenizers and templates through the reference (tokenizer, template and config files only, no weights). Releases are pinned in `docs/DECISIONS.md` (M0).

| Rule (check) | Qwen3 | Inkling-2-Mini | Hy4-flash | DeepSeek-V4.1 | Reached? |
| --- | --- | --- | --- | --- | --- |
| Real BOS in the render (`t1.special_tokens` leading run; double-BOS fault becomes a true second BOS) | no BOS | no BOS | yes, all 8 fixtures | yes, the 4 fixtures that compute | yes: Hy4, DeepSeek |
| Pad token is a stop token | yes (`<\|endoftext\|>`) | no | no | yes | yes |
| Pad token is the turn-closing token (the pad = EOS masking bug is live) | no | no | no | yes (`<｜end▁of▁sentence｜>`) | yes: DeepSeek |
| Stop tokens on the prompt side (pad = EOS masking would also hit prompt tokens) | yes (`<\|im_end\|>`) | no | yes | no | yes |
| More than one stop token inside one assistant span | no | no | no | no | **no** |
| Different closing tokens on different turns (e.g. tool calls vs answers) | no | no | no | no | **no** |
| Neutral separator after the stop token | yes (13 tokens) | none | none | none | Qwen3 only |
| Generation markers to cross-check | no | yes | no | no | yes: Inkling |
| Template preserves the prefix at some boundaries (the "framework breaks where the template doesn't" branch, on a model's own template) | breaks at every boundary | preserves every boundary | breaks only at `reasoning` boundary 2 | n/a (its multi-turn fixtures don't compute) | yes: Inkling, Hy4 |
| Truncation modes disagree | only `long_answer` | only `long_answer` | only `long_answer` | only `long_answer` | fixture-bound |

Fixtures that compute: Qwen3 8, Inkling 8, Hy4 8, DeepSeek 4 (its history turns refuse; see `docs/DECISIONS.md`, 2026-11-24).

## What no target reaches

1. **Several stop tokens per assistant turn, or different closing tokens on different turns.**
   - On all four models every assistant turn closes with one stop token, the same on every turn. So two things go unexercised:
     - `closing_token_id` (special tokens, truncation) looks only at the first span;
     - `end_of_turn` assumes a single closing token per turn.
2. **Truncation modes disagreeing anywhere but `long_answer`.** This depends on the fixtures, not the tokenizer.
   - Every other fixture runs at its exact length, where right, left and drop give the same row.
   - `long_answer`'s cut always falls inside its single answer.
   - A tokenizer can't change this; a fixture can. For example, a multi-turn conversation long enough that a right cut falls inside an earlier assistant turn and a left cut drops the system prompt.

## Proposal (not added)

**Calibration tokenizer for rule 1: `openai/gpt-oss-20b`.**
- Ungated, Apache-2.0, checked with the Hub API on 2026-11-24. Revision at the time: `6cee5e81ee83917806bbde320786a8fb61efebee`. Tokenizer and template only.
- Its harmony format closes turns with different tokens: `<|return|>` for a final answer, `<|call|>` for a tool call, `<|end|>` for history. Its stop tokens (generation_config) are `<|return|>`, `<|call|>` and `<|endoftext|>`.
- What happened when I ran the reference on it:
  - `single_turn` computes, closing with `<|return|>`.
  - `multi_turn` refuses with "no stop token inside the assistant message": history turns end with `<|end|>`, which isn't a stop token.
  - `tool_call` refuses: its template rejects the conversation with the tool-calling turn removed. That now surfaces as a clean reference refusal; before this note it was a raw Jinja error.
- So adopting it needs one decision first: how the reference treats a turn terminator that isn't a generation stop (`<|end|>`). For example:
  - treat it as the turn's closing token for history turns only;
  - or accept "history turns can't be judged" on harmony models.
- Until that's decided, a cheaper way to guard `end_of_turn`'s multi-closing-token logic is a test-only variant of Qwen3's template that closes tool-call turns with `<|endoftext|>` (also a Qwen3 stop token). That needs no new download, but it is a test template, not a calibration model.

Also checked: `microsoft/Phi-3.5-mini-instruct` (MIT) and `HuggingFaceTB/SmolLM3-3B` (Apache-2.0) are ungated, but both close every turn the same way, so they don't reach rule 1.

Rule 2 needs a fixture, not a tokenizer. Adding a fixture changes the 432-run count, so it's your call.
