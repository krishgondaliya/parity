# P-0001: With Qwen3's own template, Axolotl's default chat_template labeling never trains the think block

- Status: difference (decided by the human, 2026-11-24): KD-0002 (`t1.assistant_mask`) and KD-0003 (`t1.truncation`), thinking mode only. Not an upstream issue. Possible docs gap logged.
- Date found: 2026-11-24
- Model: Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca
- Framework: axolotl@0.21.1   (lockfile: envs/axolotl-0.21.1/uv.lock)
- Check: t1.assistant_mask (all 8 fixtures), t1.truncation (7 boundary fixtures, same root cause)   Kind: bug (as run; no registry entry)
- Verdict files: docs/evidence/2026-11-24-m3-axolotl/verdicts/t1.assistant_mask__*.json, t1.truncation__*.json
- Predicted from Axolotl's docs before the run: yes (docs/evidence/2026-11-24-m3-axolotl/predictions.md)

## Evidence
- Config: `type: chat_template`, `roles_to_train: ["assistant"]`, `train_on_eos: turn`. `chat_template` is left at its documented default, `tokenizer_default` (Qwen3's own template).
- Expected: the last assistant turn renders `<|im_start|>assistant\n<think>\n\n</think>\n\n<answer><|im_end|>`.
  - In thinking mode (the default) the generation prompt ends at `<|im_start|>assistant\n`, so the model writes the think block itself.
  - So `<think>`, `\n\n`, `</think>`, `\n\n` are must-train.
- Got: those 4 tokens have label −100 on every fixture; the answer and `<|im_end|>` are trained.
  - `single_turn`: positions 13–16.
  - With `reasoning_content` (`reasoning`): only the opening `<think>` and `\n` are untrained, while `\n</think>\n\n` is trained. The model learns to close its think block but not to open it.
- Mechanism (axolotl/prompt_strategies/chat_template.py @ v0.21.1): `find_turn` returns each turn's field boundaries (content, reasoning, tool calls), and only that span is labeled. Template structure before the first field gets no loss.

## Triage (from /triage)
- **Calibration green on known-good model:** this run IS the calibration model.
  - The checks are green on Qwen3 through TRL 1.18.2 (KD-0001 only; docs/evidence/2026-11-24-m3-trl/) and on both test stand-ins, so the check logic is not the suspect.
  - Axolotl on Qwen3 is what's being triaged.
- **Adapter uses the documented public path and option:** yes.
  - The YAML `type: chat_template` with `roles_to_train: ["assistant"]` and `train_on_eos: turn` is the documented assistant-only recipe (docs/dataset-formats/conversation.qmd @ v0.21.1).
  - Cross-check: the real `axolotl preprocess config.yml --debug-num-examples 0` CLI produced rows identical to the adapter's, labels included, on `single_turn` and `empty_assistant`.
- **Fixture valid for this model:** yes. Plain user/assistant conversations, rendered by Qwen3's own template.
- **Reference consistent:** yes. It agrees with:
  - the hand-written golden spans (tests/test_reference_template.py);
  - TRL's maintained Qwen3 training template, which puts the think block inside `{% generation %}` in thinking mode, and emits outside it only what the generation prompt pre-fills ("so that assistant-only loss doesn't train it").
  - The model card was not consulted (rule 6 allowlist); the skeptic may.
- **Documented as intended behavior: partly.**
  - The `generation_tags` option is described as labeling "assistant turns from the chat template's {% generation %} blocks, including the template structure inside them, instead of locating each turn's fields" (axolotl/utils/schemas/datasets.py; docs/dataset-formats/conversation.qmd "Labeling from generation markers").
  - Axolotl's own test `tests/prompt_strategies/test_chat_templates_generation_tags.py` says markers are needed so "channel headers and the empty analysis block are generated, not just field contents".
  - So the default deliberately labels fields only. But nothing documents that Qwen3's think tags go untrained.
  - The documented remedy doesn't apply here: Qwen3's own template has no generation markers, so `generation_tags` has nothing to read.
- **Duplicates searched:** `gh search issues --repo axolotl-ai-cloud/axolotl` for "think qwen3", "think tokens masked qwen3 chat_template", "<think> not trained", "reasoning_content mask", "chat_template". No duplicate.
  - Related, same turn locator: #4119 (open; the locator trains a role-header token when content starts with whitespace) and #3754 (closed; Gemma turn terminator never trained).

## Skeptic verdict
- **Verdict: intended behavior (documented difference). Confidence: medium.**
  - On the 7 fixtures without reasoning, part of the fail comes from **our reference's thinking-mode assumption** (rule 4).
- **Tested design.** `tests/prompt_strategies/test_chat_templates_thinking.py:177` @ v0.21.1 (from #4190) asserts `trained == "greet</think>hello<|im_end|>part</think>see you<|im_end|>"`. The opening `<think>` is untrained and `</think>` trained: exactly our `reasoning` symptom.
  - Two documentation lines describe the same design: conversation.qmd says `generation_tags` "also trains template structure inside them", and the schema says "instead of locating each turn's fields".
- **The mode ambiguity is ours.**
  - Qwen3's own template renders the last turn identically in both modes. With `enable_thinking=False` the generation prompt already ends in `<think>

</think>

`, so only `Red, yellow and blue.<|im_end|>
` is supervised. That is exactly what Axolotl trains.
  - A fixture with no reasoning looks like non-thinking data.
  - Axolotl's own Qwen3 examples train on FineTome-100k, which has no reasoning.
  - The model card says thinking mode "will generate think content wrapped in a `<think>...</think>` block"; an empty block is model-generated only under the `/no_think` soft switch.
  - Axolotl's bundled `qwen3.jinja` also treats `<think>` as prompt side: its generation prompt in thinking mode ends in `<think>

`.
- **Settings tried** through Axolotl's real preprocess path (scratch probe):
  - `chat_template: qwen3`, which is Axolotl's README recommendation: identical labels.
  - Plus `split_thinking: true`: identical labels.
  - `generation_tags: true` on Qwen3's own template: raises a `ValueError` (loud, not silent).
  - TRL's `qwen3_training.jinja` with `generation_tags: true`: trains the block, but changes the tokens. Not the documented recipe.
- **What remains.** On *reasoning* data with Qwen3's own template, the model writes `<think>` and it's never trained, and no doc warns. That is a docs gap, not a labeling bug.
- **Not fixed on main;** no duplicate (#3088, open, Qwen3 think tokens with `train_on_inputs: false`, is related but a different symptom).
- **What would change its mind:** a maintainer or doc saying the default labeling should cover everything the model generates in the turn.
- **Proposed handling (human decides):**
  1. A known-differences entry citing the test and the docs, not an issue.
  2. A reference decision on thinking-mode ambiguity: declare the mode per run and check both, or make tokens whose zone changes between Qwen3's documented modes neutral. In non-thinking mode, Axolotl passes the 7 no-reasoning cells.

## Decision (human, 2026-11-24)
- **Which tokens are untrained** (re-checked token by token on `reasoning` from Axolotl's prepared row): only `<think>` (51) and its `\n` (52). The reasoning text, `</think>`, `\n\n`, the answer and `<|im_end|>` are trained. On the other 7 fixtures, the empty block's 4 tokens. Think tags and blank lines only.
- **Reference:** the serving mode is now declared (`TrainCfg.serving_mode`), defaulting to the model's documented default, and recorded in every verdict. No neutral zone. Primary results stay in thinking mode; every run is also judged in non-thinking mode as a labeled secondary result. This change came from this finding, after seeing results (docs/DECISIONS.md).
- **Known difference:** KD-0002 / KD-0003, signature `think_opener_untrained`, pinned to `tests/prompt_strategies/test_chat_templates_thinking.py#L143-L177` @ v0.21.1.
- **Docs:** Axolotl's user docs don't say it; `conversation.qmd` L445 only implies it. Logged as a possible docs gap, not a finding.

## Cells after the decision (re-run 2026-11-24, `docs/evidence/2026-11-24-m3-rerun/axolotl/`)
- Thinking mode (primary): 13 differences, KD-0002 on `t1.assistant_mask` ×7 and KD-0003 on `t1.truncation` ×6. They link to P-0001 through the registry's `finding` field.
- `empty_assistant` stays a bug on both checks: KD-0002/0003 explain its think opener, and the residual (`<|im_end|>` at 14) is P-0002, to which those cells are linked.
- Non-thinking mode (secondary): Axolotl passes all 7 no-reasoning fixtures on both checks, except `empty_assistant` (P-0002). `reasoning` is inconclusive there by design.

## Repro
- findings/P-0001/repro.py (tokenizer and config files only, telemetry off): exits 1 (reproduces) 3/3 on axolotl 0.21.1.
- On main: not tested.

## Upstream
- None. It is a known difference (KD-0002, KD-0003), not an issue.
