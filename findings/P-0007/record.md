# P-0007: Axolotl drops every message field its template analyzer can't see; on DeepSeek V4.1, the reasoning and the tool calls

- Status: candidate
- Severity: medium (effect: on an affected template, reasoning SFT silently becomes empty-think SFT, and a tool-call turn becomes an empty turn, whose end token P-0002's rule then leaves untrained too; reach: any conversation with reasoning or tool calls on a template whose fields Axolotl's analyzer misses: DeepSeek V4.1's, and, preliminarily, Axolotl's own bundled `exaone4` and `gemma4` templates), by effect x reach (docs/DECISIONS.md)
- Date found: 2026-11-26
- Model: deepseek-ai/DeepSeek-V4.1-Flash-1017@08c8657f02291e11b6f6b63e0e16014f8e748518 (its tokenizer and chat template; the mechanism needs no architecture)
- Framework: axolotl@0.21.1   (lockfile: envs/axolotl-0.21.1/uv.lock)
- Check: t1.template_parity   Fixture: reasoning   Kind: bug (the fixture's other checks are inconclusive behind it)
- Verdict file: docs/evidence/2026-11-26-m5-axolotl-deepseek/verdicts/thinking/t1.template_parity__reasoning.json (M5, the `deepseek_v4` stand-in; D1)
- Predicted before the run: no (`docs/evidence/2026-11-26-m5-axolotl-deepseek/predictions.md` predicted template parity pass on `reasoning`).
- **Not this finding:** the 9 assistant-mask and truncation fails on single_turn, multi_turn, system_prompt, long_answer and non_ascii (the last turn's `</think>` untrained). Those fixtures carry no reasoning, the field drop doesn't touch them, and with the fields mapped explicitly `</think>` is still untrained there. They are KD-0002's first-field rule on DeepSeek's template, left untriaged for the human's decision (below).

## Evidence
- **Reasoning dropped** (M5 `t1.template_parity` on `reasoning`, first divergence at token 63): the template renders the last turn as `<｜Assistant｜><think>97 is not divisible by 2, 3, 5 or 7, …</think>Yes, 97 is prime.<｜end▁of▁sentence｜>`; Axolotl's row has `<｜Assistant｜><think></think>Yes, 97 is prime.<｜end▁of▁sentence｜>` (72 tokens, against 105).
- **Tool calls dropped** (`docs/evidence/2026-11-26-m5-axolotl-deepseek/triage_fields.py`, the same full path; our reference refuses this fixture on DeepSeek, so M5's cells can't show it): the template renders the tool-call turn as `<｜Assistant｜><think></think>\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="get_weather">…</｜DSML｜ calls><｜end▁of▁sentence｜>`; Axolotl's row has `<｜Assistant｜><think></think><｜end▁of▁sentence｜>`. The tool schema in the system prompt survives (it comes from `tools`); the call itself is gone.
- **The model generates `</think>`** (`triage_encoding_think.py`, the model's own `encoding/encoding.py`, sandboxed): in thinking mode the prompt ends `<｜Assistant｜><think>`, and the model generates `</think>Red, yellow and blue.<｜end▁of▁sentence｜>`, or the reasoning then `</think>…`. So on a dropped reasoning turn, neither the reasoning nor its `</think>` is trained.
- **Qwen3, the same path:** reasoning and tool calls reach the template, and the rows equal the render (`triage_fields.json`, `triage_reasoning.json`).
- **The skeptic's own reproduction, without Parity:** a plain `type: chat_template` config on an OpenAI-format jsonl, through `load_cfg` + `load_datasets`, with `base_model` a Qwen3 config folder and `tokenizer_config` the DeepSeek V4.1 files. Same result twice; no stand-in involved.

## Mechanism (axolotl 0.21.1; files byte-identical to the v0.21.1 tag; `chat_template.py` and `jinja_template_analyzer.py` unchanged in substance on main `de8549a`)
1. Config validation fills `message_property_mappings: {role: role, content: content}` (the validated dataset config, `triage_reasoning.json`). The mappings are never empty on the real `load_cfg` path, so the branch of `ChatTemplatePrompter.__init__` that would add `reasoning_content` (`prompt_strategies/chat_template.py` L82-L88) never runs.
2. `transform_message` (L1325-L1400) keeps the mapped fields, then any other field only if it is among `chat_template_msg_variables`: "Keep only the properties defined in the chat template and not already mapped".
3. Those come from `JinjaTemplateAnalyzer.get_message_vars("messages")` (`prompt_strategies/jinja_template_analyzer.py` L319-L335). In practice only properties of a variable literally named `message` count, anywhere in the template:
   - its loop tracking maps loop variable to source (Qwen3 `{message: messages}`, DeepSeek `{msg: messages, m: msgs}`), but the lookup is by the source name, so it is never followed, and `analyze_template` resets it;
   - `message.get('x')` records `get`, not `x` (the skeptic's toy templates).
4. DeepSeek V4.1's template loops `for msg in messages` (L84) and reads reasoning as `msg.reasoning_content` / `msg.get("thinking")`. It reads tool calls later as `m.tool_calls` over its own list (L130-L133). The analyzer finds no message fields at all (`[]`). Qwen3, Inkling and Hy4 name the variable `message` and keep theirs. Renaming `msg` to `message` in a copy brings back `reasoning_content`, but not `tool_calls`.
5. So `reasoning_content`, `tool_calls`, `tool_call_id` and `name` are dropped before rendering. Nothing says so: the only trace is a DEBUG line (L395-L397), and the strategy's logger is set to INFO (L28), so it never prints.

## The `</think>` fails on fixtures without reasoning (separate; for the human)
- On those turns Axolotl labels from the first field it has, the content, to the end. Its own test covers turns with reasoning (`tests/prompt_strategies/test_chat_templates_thinking.py` L143-L177: `greet</think>hello<|im_end|>…` trained), where the span starts at the reasoning and `</think>` is trained.
- With no reasoning, the template-emitted `</think>` before the content is outside the span. That's the same as Qwen3's empty think block (`triage_think_close.json`: Qwen3's `</think>` is unlabeled on single_turn and multi_turn too), which KD-0002 and KD-0003 cover for Qwen3's template.
- In DeepSeek's thinking mode the model generates that `</think>` (encoding.py, above), so it's untrained model output either way. The guidance for this triage assumed Qwen3 trains `</think>`. That holds only on turns with reasoning.
- **Open for the human:** treat the first-field rule on DeepSeek's template as a known difference (a new KD entry; KD-0002/KD-0003 are pinned to Qwen3's template hash), or as a candidate on both templates. Until then those 9 cells are untriaged. This record doesn't extend the KD entries.

## Triage (from /triage)
- **Calibration green on the known-good model:** yes (M4; re-calibrated 96/96 after the adapter change, `docs/evidence/2026-11-26-m4-recalibration/`).
- **Adapter uses the documented public path and option:** yes. `type: chat_template`, the tokenizer's own template, `roles_to_train: [assistant]`, `train_on_eos: turn`, as on Qwen3. The mappings are Axolotl's validated defaults; the adapter writes `{"messages": …}` (plus `tools`) and sets none.
- **Fixture valid for this model:** yes. `reasoning_content` is the field DeepSeek's template reads (and encoding.py's). The tool call follows the OpenAI format the template converts.
- **Reference consistent:** yes. The template agrees with the model's own `encoding.py` on all 8 fixtures (`docs/evidence/2026-11-25-deepseek-encoding/`).
- **Stand-in (D1):** the M5 run used `deepseek_v4`, but the validated mappings, the analyzer and `transform_message` read only the config and the template. The skeptic reproduced it with Qwen3's config. A standalone repro needs only the release's tokenizer and template.
- **Documented as intended behavior:** no. The schema documents the opposite (`utils/schemas/datasets.py` L152, `message_property_mappings`): "If a property exists in the template but not in this mapping, the system will attempt to load it directly from the message using the property name as the key." The conversation docs' tool-use example also shows `tool_calls` with no mapping config.
- **Workaround, tested by the skeptic:** explicit `message_property_mappings: {role: role, content: content, reasoning_content: reasoning_content, tool_calls: tool_calls}`. Then both rows equal the render, and `</think>` is trained on the reasoning turn. Setting `field_thinking` and `template_thinking_key` doesn't help.
- **Duplicates searched** (read-only, issues and PRs): "JinjaTemplateAnalyzer", "chat_template_msg_variables", "reasoning_content dropped", "tool_calls missing chat_template", "message_property_mappings reasoning_content", "deepseek v4 tool_calls", "properties on the message". No duplicate; the skeptic's further searches were cut short by the search API's rate limit. Related:
  - #3824 (open): the DPO strategy drops `tool_calls`, a different strategy;
  - #3768 (open PR): the analyzer raises `RecursionError` on deeply nested templates, a loud failure;
  - #3228 (open PR, "include tool in default message_property_mappings"): edits the empty-mapping branch the validated path never reaches;
  - #2762 (closed): generation tags in the analyzer.

## Skeptic verdict (2026-11-26)
- **Real candidate, for the dropped fields only.** Confidence high on the mechanism and the reproduction, medium on severity and reach as first framed. Our adapter and fixtures are not the cause.
- **Strongest argument against:** DeepSeek V4.1 isn't loadable in 0.21.1 (transformers 5.20.1 doesn't know `deepseek_v41`), so a maintainer can call it unsupported and say "map your fields for unusual templates". Two answers:
  - the schema promises the opposite (above);
  - Axolotl's own bundled templates hit the same rule: `exaone4` reads fields through `msg = messages[i]`, and `gemma4` through `message.get('reasoning_content')`. Preliminary: shown at the strategy level, not yet through the full preprocess path at render level.
- **Corrections, applied here:**
  - the `</think>` fails without reasoning are not part of this finding;
  - the severity is about reasoning SFT becoming empty-think SFT, plus the P-0002 interaction on an emptied tool-call turn (noted, not counted twice);
  - reach is framed around the analyzer, which applies to any template: no one can train DeepSeek V4.1 with Axolotl 0.21.1 today, and without tools the template drops history reasoning by itself;
  - the docs quote is replaced by the schema description;
  - the analyzer mechanism is stated precisely;
  - the debug-log lines and the logger level are corrected;
  - the workaround is now tested.
- **To check before any upstream draft:**
  1. A standalone repro: `tokenizer_config` pointing at the DeepSeek release files, `base_model` a supported model, exit 1 when a row differs from the render.
  2. `exaone4` and `gemma4` at render level through the full preprocess path. If they hold, the issue leads with the analyzer's variable-name rule, with DeepSeek as one instance.
  3. The duplicate searches the rate limit cut short: "message_property_mappings reasoning_content", "exaone4 tool_calls", "gemma4 reasoning_content".
  4. The non-reasoning `</think>` question stays out of the issue.
  5. No "bug" in the title (rule 8).

## Repro
- Not written yet. The skeptic's scratch script (outside the repo) is a starting point; a committed repro is M6 work.

## Upstream
- Issue URL (posted by human): none
- Maintainer response: none yet
- Fixed in: not yet
