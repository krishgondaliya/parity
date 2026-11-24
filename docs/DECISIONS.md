# Decision log

One entry per decision that changes scope, a pass rule, a tolerance, a target list, or a gate outcome. Newest first.

Format:

```
## YYYY-MM-DD: <short title>
- Decision:
- Why:
- Evidence: <file paths, run ids, links>
- Reversible? yes/no
```

## 2026-11-23: Open note for Tier 3: TRL trains Qwen3 with think blocks that serving strips
- Note (not a decision, not a finding):
  - TRL 1.18.2 trains Qwen3 through `qwen3_training.jinja`, which keeps the think block in every assistant turn.
  - At inference, Qwen3's official template strips reasoning from assistant turns before the last user message.
  - So in multi-turn conversations, the history a served model sees differs from the history it was trained on. This is a train/serve mismatch.
- Follow-up: the Tier 3 serving-parity check must cover this case. Compare the training-format tokens with the served tokens on the `multi_turn`, `reasoning`, `tool_call` and `empty_assistant` fixtures, then decide whether it is a documented difference or a candidate finding.
- Evidence: `tests/test_reference_template.py::test_trl_template_renders_the_same_tokens_only_with_a_single_assistant_turn`; the TRL template header (link in the registry entry below).

## 2026-11-23: Template properties are judged before frameworks (multi-turn prefix)
- Decision:
  1. The multi-turn prefix check first tests the model's official template alone (`prefix_breaks` in the reference).
  2. If the official template isn't prefix-preserving, every framework using it inherits `difference (template)` at exactly those boundaries. The difference is attributed to the model, not the framework.
  3. A framework fails only if it breaks the prefix property beyond what its declared template does. Its breaks must be a subset of the breaks of the template it says it uses, whether that's the official template or its own training template.
  4. Known, documented template properties are not new findings: no finding record and no issue.
- Why: otherwise every framework would be blamed for the model's template. On Qwen3 that would mean 4 false fails out of 8 fixtures on the calibration model itself (rule 4: assume our code and our reference are wrong first).
- Evidence: Qwen3's official template breaks after assistant turns (`multi_turn` (2, 4); `tool_call`, `reasoning`, `empty_assistant` (2)), per `test_qwen3_template_is_not_prefix_preserving_after_assistant_turns`. The TRL training template's header says it was changed to be prefix-preserving.
- Reversible? yes

## 2026-11-23: Known-differences registry
- Decision:
  - A divergence counts as `difference` only if it matches a registry entry **exactly**. Everything else stays `fail`.
  - Each entry covers one documented, deliberate behavior and has: id, framework, version range, model family (plus the exact template it applies to), check id, exact divergence signature, and doc link.
  - No wildcard entries: no "any version", no "any model", no "any check". Widening an entry, for example to a new framework version, means verifying it again and logging it here.
  - Calibration is green when there are zero unexpected fails. Matched differences are listed by entry id in the calibration report.
  - The registry becomes a data file with the first check in M3. Adding or changing an entry needs an entry in this log.
- First entry, **KD-0001**:
  - Framework / versions: `trl ==1.18.2`, the only version verified so far.
  - Model family: Qwen3. It applies only when the model's chat template is byte-identical to TRL's `qwen3.jinja`, because that's the comparison TRL uses to swap in its training template. Verified for `Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca` (4,168 chars, identical). The training template used must have sha256 `1907918feb7afcf18fac8db1a419985b4773ac434e353cfae91b1949dcbf592b`.
  - Check: `t1.template_parity`.
  - Signature: start from the official render. For every assistant turn the official template renders without a think block, insert `<think>\n` + `reasoning_content.strip("\n")` + `\n</think>\n\n` immediately after that turn's `<|im_start|>assistant\n`. TRL's render must equal the result exactly; any other divergence is a `fail`.
  - Expected on the calibration fixtures: `multi_turn`, `tool_call`, `reasoning` and `empty_assistant` diverge in this way only; the other 4 fixtures don't diverge at all.
  - Docs:
    - The template header, lines 1–9 ("Removed the loop.index0 > ns.last_query_index conditional; always include thinking block"): https://github.com/huggingface/trl/blob/v1.18.2/trl/chat_templates/qwen3_training.jinja
    - TRL's chat templates README, lines 5–10 (training templates are swapped in by identity comparison, for prefix preservation and generation markers): https://github.com/huggingface/trl/blob/v1.18.2/trl/chat_templates/README.md
- Why: rule 4 and the zero-retractions goal. A deliberate, documented choice must not be filed as a bug. An exact, non-wildcard match makes sure the registry can't hide a real bug that happens to sit next to a documented behavior.
- Reversible? yes

## 2026-11-23: Mask zones for the assistant-mask check
- Decision: every reference token falls into exactly one zone.
  - **must-train:** assistant content plus its stop token. That's everything the template renders for the assistant turn after the generation prompt (think block, text, tool calls), through the stop token. In the reference this is `[token_start, token_end)`.
  - **must-not-train:** system, user and tool content, and every role header, including whatever the generation prompt pre-fills.
  - **neutral:** separator tokens after the stop token and before the next header, like the `\n` after `<|im_end|>`. In the reference this is `[token_end, token_message_end)`.
  - Pass rule: fail if any must-train token is unsupervised, or any must-not-train token is supervised. Neutral tokens never cause a fail either way. The verdict records the framework's neutral convention (`trained`, `not_trained` or `mixed`, with counts).
- Why: the newline after the stop token is never generated, so training on it is harmless, and frameworks differ on it. TRL's markers train it; our reference doesn't. Calling that a fail would be noise, but recording it keeps the difference visible.
- Open for M3: how the zones map onto templates that ship their own generation markers (Inkling). The likely rule: marked tokens through the stop token are must-train, and marked tokens after it are neutral. Decide and log this before running Inkling.
- Reversible? yes

## 2026-11-23: Tier 1 reference, and how assistant spans are defined
- Decision:
  - Reference tokens are `apply_chat_template(messages, tools=...)` with default options and the model's own template (transformers 5.24.0, pinned in `uv.lock`).
  - Templates with `{% generation %}` markers: the template's markers define the mask, read through `return_assistant_tokens_mask`.
  - Templates without markers: a prefix diff anchored in the full render. Message start = `len(render(messages[:k]))`. Content start = the same render with `add_generation_prompt=True`. Message end = found by cutting the message out of the full render and checking the result exactly equals `render(messages without k)`.
  - The supervised span runs from the content start through the turn's last stop token (tokenizer EOS plus the `eos_token_id`s in `generation_config.json`). The separator after the stop token (Qwen3: `\n`) is recorded as `token_message_end` but not supervised, because the model never generates it.
  - Any ambiguity raises `TemplateReferenceError`, which becomes `inconclusive`, never `pass`. Examples: a boundary inside a token, a prefix that doesn't match, or no stop token.
- Why:
  - The playbook's literal method (diff `render(messages[:k])` against `render(messages[:k+1])`) fails on the calibration model itself. Qwen3 renders a turn with an empty `<think></think>` when it is the last message, but without one once later messages follow.
  - Anchoring every boundary in the full render avoids that problem without re-implementing any template.
- Evidence:
  - `tests/test_reference_template.py`.
  - On TRL 1.18.2's `qwen3_training.jinja` (vendored in `tests/data/trl/`), the prefix-diff spans exactly equal the mask from TRL's generation markers on all 8 fixtures.
  - Hand-written golden spans match on 5 fixtures.
- Consequences to settle in M3, before calibration:
  1. **Trailing separator.** TRL's markers also supervise the `\n` after `<|im_end|>`, and ours don't. The assistant-mask check must decide whether a separator after the stop token is allowed. Log that tolerance here when it's decided.
  2. **TRL's training template differs from Qwen3's official one by design.** TRL always emits a think block, and its template header documents this. On `multi_turn`, `tool_call`, `reasoning` and `empty_assistant`, TRL's tokens differ from the official template's. If TRL uses that template, template parity on the calibration model will show a documented `difference`. Calibration must count it as expected, not as a fail.
  3. **Qwen3's own template isn't prefix-preserving after assistant turns.** `prefix_breaks` per fixture: `multi_turn` (2, 4); `tool_call`, `reasoning` and `empty_assistant` (2). The multi-turn prefix check must blame these on the template, not on a framework.
- Update (same day): all three are settled by the entries above. Item 1 by "Mask zones", item 2 by "Known-differences registry" (KD-0001), item 3 by "Template properties are judged before frameworks".
- Reversible? yes

## 2026-11-23: Phase 0 targets and calibration model pinned (M0)
- Decision: run the kill test on these 3 targets plus 1 calibration model, each pinned to an exact Hugging Face commit:

  | Role | Model | Revision (commit SHA) | Released | Reasoning | Tool calls | `{% generation %}` markers | License |
  | --- | --- | --- | --- | --- | --- | --- | --- |
  | Target | `thinkingmachines/Inkling-2-Mini` | `7a67ae2314b44ef05c87fd1d4d311c58aaeb18f6` | 2026-11-20 | yes | yes | yes (ships its own) | Apache-2.0 |
  | Target | `tencent/Hy4-flash` | `eccd3123eec73f2040f4c9ed0de1883593470510` | 2026-11-16 | yes | yes | no | Tencent custom (`tencent-hy-license`) |
  | Target | `deepseek-ai/DeepSeek-V4.1-Flash-1017` | `08c8657f02291e11b6f6b63e0e16014f8e748518` | 2026-10-16 | yes | yes | no | MIT |
  | Calibration | `Qwen/Qwen3-0.6B` | `c1899de289a04d12100db370d81485cdf75e47ca` | 2025-04-26 | yes | yes | no (TRL ships `qwen3_training.jinja`) | Apache-2.0 |

- Why:
  - All three targets are major-lab releases from the last ~5 weeks. All are ungated, and all load with a standard tokenizer class, so no repo code runs.
  - All have both reasoning blocks and tool calls, so they exercise all 8 fixtures. All appear to strip `<think>` from earlier turns, which exercises the multi-turn prefix check.
  - They cover two mask cases. Inkling ships `{% generation %}` markers, so we test whether those markers are right. Hy4 and DeepSeek ship none, so each framework has to find the assistant spans on its own.
  - DeepSeek V4.1 Flash is the most widely fine-tuned candidate.
  - Qwen3 is the known-good calibration family, because TRL ships a training template for it (TRL #5471).
  - Candidates I rejected:
    - `mistralai/Mistral-Large-4-1T-A52B`: TRL already ships a training template for it, so TRL bugs there are less likely.
    - `microsoft/Aion-1.0-Instruct-2.3B`: no reasoning blocks.
    - `moonshotai/Kimi-K3.1-Pro`: its tokenizer is custom repo code (`tokenization_kimi.py` + tiktoken).
    - Muse Spark Open and GLM-5.3: released before the ~2-month window.
- Evidence:
  - Hugging Face Hub API, queried 2026-11-23. Each SHA resolves, and each was the repo's latest commit when pinned.
  - At each pinned revision, `tokenizer.json` and `tokenizer_config.json` are present, and the targets also have `chat_template.jinja`.
  - Template features were read from the template text at the pinned SHA. None contain CR line endings.
  - The TRL template list was read from `trl/chat_templates/` on main.
- Notes to carry forward:
  - Inkling and Hy4 were both updated on the pin date (2026-11-23). If they fix their templates later, diff the new version against these pins.
  - Hy4 uses the `TokenizersBackend` tokenizer class, which needs transformers v5. If a framework env pins transformers v4, it may not load the tokenizer. M1 will show this.
  - The Hy4 license hasn't been reviewed yet. The human should read it before any Hy4 results are published.
- Reversible? yes (re-pin and re-run; earlier verdicts keep their recorded revisions)

## 2026-11-23: Parity chosen as the career-first project
- Decision: build Parity (Day-0 correctness lab) with a self-improving check suite and a two-question evaluation as core deliverables.
- Why: highest portfolio ceiling, objectively checkable, CPU-first. See `docs/plan/04-decision-history.md`.
- Evidence: plan doc https://claude.ai/artifact/LuCqsapFCxPhWiaD5RJzcu
- Reversible? yes (fallback: Crashproof Rollouts if Gate A fails)
