# P-0007: Axolotl drops every message field its template analyzer can't see: reasoning and tool calls on EXAONE-4.0, Gemma 4 and DeepSeek V4.1 templates

- Status: **candidate, posted 2026-11-27**: https://github.com/axolotl-ai-cloud/axolotl/issues/4320, by the human. Repro shrunk. "Candidate" until a maintainer confirms (rule 8).
- Severity: medium (effect: reasoning SFT silently becomes empty-think SFT, and a tool-call turn loses its call, becoming an empty turn whose end token P-0002's rule then leaves untrained; reach, users today: EXAONE-4.0 with Axolotl's bundled `exaone4` or its own template, and gemma-4-12B-it with its own template, lose reasoning and tool calls silently on ordinary OpenAI-format data; the bundled `gemma4` and `gemma4_unified` raise an unexplained `TypeError` on OpenAI-format tool results, and drop reasoning silently only in Gemma's native tool format. Latent: DeepSeek V4.1, once Axolotl's transformers builds it), by effect x reach (docs/DECISIONS.md; `docs/evidence/2026-11-26-p0007-reach/`)
- Date found: 2026-11-26
- Model: deepseek-ai/DeepSeek-V4.1-Flash-1017@08c8657f02291e11b6f6b63e0e16014f8e748518 (its tokenizer and chat template; the mechanism needs no architecture)
- Framework: axolotl@0.21.1   (lockfile: envs/axolotl-0.21.1/uv.lock)
- Check: t1.template_parity   Fixture: reasoning   Kind: bug (the fixture's other checks are inconclusive behind it)
- Verdict file: docs/evidence/2026-11-26-m5-axolotl-deepseek/verdicts/thinking/t1.template_parity__reasoning.json (M5, the `deepseek_v4` stand-in; D1)
- Predicted before the run: no (`docs/evidence/2026-11-26-m5-axolotl-deepseek/predictions.md` predicted template parity pass on `reasoning`).
- **Not this finding:** the 9 assistant-mask and truncation fails on single_turn, multi_turn, system_prompt, long_answer and non_ascii (the last turn's `</think>` untrained). Those fixtures carry no reasoning, the field drop doesn't touch them, and with the fields mapped explicitly `</think>` is still untrained there. They are KD-0002's first-field rule on DeepSeek's template: known differences KD-0004 and KD-0005 since 2026-11-26 (the human's decision; same code path and test as KD-0002, traced in `docs/evidence/2026-11-26-kd-deepseek-think-close/`).

## Evidence
- **Reasoning dropped** (M5 `t1.template_parity` on `reasoning`, first divergence at token 63): the template renders the last turn as `<｜Assistant｜><think>97 is not divisible by 2, 3, 5 or 7, …</think>Yes, 97 is prime.<｜end▁of▁sentence｜>`; Axolotl's row has `<｜Assistant｜><think></think>Yes, 97 is prime.<｜end▁of▁sentence｜>` (72 tokens, against 105).
- **Tool calls dropped** (`docs/evidence/2026-11-26-m5-axolotl-deepseek/triage_fields.py`, the same full path; our reference refuses this fixture on DeepSeek, so M5's cells can't show it): the template renders the tool-call turn as `<｜Assistant｜><think></think>\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="get_weather">…</｜DSML｜ calls><｜end▁of▁sentence｜>`; Axolotl's row has `<｜Assistant｜><think></think><｜end▁of▁sentence｜>`. The tool schema in the system prompt survives (it comes from `tools`); the call itself is gone.
- **The model generates `</think>`** (`triage_encoding_think.py`, the model's own `encoding/encoding.py`, sandboxed): in thinking mode the prompt ends `<｜Assistant｜><think>`, and the model generates `</think>Red, yellow and blue.<｜end▁of▁sentence｜>`, or the reasoning then `</think>…`. So on a dropped reasoning turn, neither the reasoning nor its `</think>` is trained.
- **Qwen3, the same path:** reasoning and tool calls reach the template, and the rows equal the render (`triage_fields.json`, `triage_reasoning.json`).
- **The skeptic's own reproduction, without Parity:** a plain `type: chat_template` config on an OpenAI-format jsonl, through `load_cfg` + `load_datasets`, with `base_model` a Qwen3 config folder and `tokenizer_config` the DeepSeek V4.1 files. Same result twice; no stand-in involved.

## Reach today (`docs/evidence/2026-11-26-p0007-reach/`, 2026-11-26)
- **Bundled templates, 3 of 36** (`scan_bundled.py`, behavioral: a field counts if the render changes when it's removed, or when it's put back into the conversation as Axolotl keeps it):
  - `exaone4` drops `reasoning_content` and `tool_calls`;
  - `gemma4` and `gemma4_unified` drop `reasoning`, `reasoning_content`, `name` and `tool_call_id`.
  The other 33 keep every field they read in our probes.
- **Through Axolotl's full path** (`full_path.py`, `load_cfg` + `load_datasets`, plain OpenAI-format jsonl, Axolotl's validated defaults; models Axolotl 0.21.1 loads, tokenizer and config only):
  - EXAONE-4.0-1.2B, bundled `exaone4` and its own template (`skip_think: false`): the reasoning and the tool call are missing from the rows, silently.
  - gemma-4-12B-it with its own template (`tokenizer_default`): the reasoning and the tool call are missing, silently.
  - gemma-4-31B with `gemma4` and gemma-4-12B-it with `gemma4_unified` (Axolotl's v0.21.1 examples): a tool conversation in OpenAI format raises `TypeError: can only concatenate str (not "NoneType") to str` in preprocessing (the template reads the result's `name`, else `tool_call_id`, both dropped). In Gemma's native form (`tool_responses` on the assistant message) the reasoning is missing, silently. These templates render no reasoning on turns without a tool call.
  - With the fields mapped explicitly, every OpenAI-format row equals its render and the reasoning and calls are trained.
- **Latent:** DeepSeek V4.1 (above), once a transformers release Axolotl uses builds `deepseek_v41`.

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
- **Decided 2026-11-26 (human):** a known difference if and only if the same code path and documented test as KD-0002. Both hold, so these are KD-0004 and KD-0005, scoped to DeepSeek V4.1's template, thinking mode and conversations without reasoning (`docs/DECISIONS.md`).

## Triage (from /triage)
- **Calibration green on the known-good model:** yes (M4; re-calibrated 96/96 after the adapter change, `docs/evidence/2026-11-26-m4-recalibration/`).
- **Adapter uses the documented public path and option:** yes. `type: chat_template`, the tokenizer's own template, `roles_to_train: [assistant]`, `train_on_eos: turn`, as on Qwen3. The mappings are Axolotl's validated defaults; the adapter writes `{"messages": …}` (plus `tools`) and sets none.
- **Fixture valid for this model:** yes. `reasoning_content` is the field DeepSeek's template reads (and encoding.py's). The tool call follows the OpenAI format the template converts.
- **Reference consistent:** yes. The template agrees with the model's own `encoding.py` on all 8 fixtures (`docs/evidence/2026-11-25-deepseek-encoding/`).
- **Stand-in (D1):** the M5 run used `deepseek_v4`, but the validated mappings, the analyzer and `transform_message` read only the config and the template. The skeptic reproduced it with Qwen3's config. A standalone repro needs only the release's tokenizer and template.
- **Documented as intended behavior:** no. The schema documents the opposite (`utils/schemas/datasets.py` L152, `message_property_mappings`): "If a property exists in the template but not in this mapping, the system will attempt to load it directly from the message using the property name as the key." The conversation docs' tool-use example also shows `tool_calls` with no mapping config.
- **Workaround, tested by the skeptic:** explicit `message_property_mappings: {role: role, content: content, reasoning_content: reasoning_content, tool_calls: tool_calls}`. Then both rows equal the render, and `</think>` is trained on the reasoning turn. Setting `field_thinking` and `template_thinking_key` doesn't help.
- **Duplicates searched** (read-only, issues and PRs, 2026-11-26, including the searches the skeptic's rate limit cut short; the full list is in the reach README). No duplicate, and no fix on main (`de8549a`: the analyzer and both templates have no commits since v0.21.1; `transform_message` is identical). Related:
  - #3824 (open): the DPO strategy drops `tool_calls`, a different strategy;
  - #3768 (open PR): the analyzer raises `RecursionError` on deeply nested templates, a loud failure;
  - #3228 (open PR, "include tool in default message_property_mappings"): adds a `tool` key to the empty-mapping default, a branch the validated path never reaches;
  - #2762 (closed): generation tags in the analyzer;
  - #2260 (closed PR, 2025): added config for passing optional message fields, and asked why not pass the whole message;
  - #3825 (open PR, the DPO strategy, from #3824's author): unions the OpenAI message keys (`tool_calls`, `tool_call_id`, `name`, `reasoning_content`, `reasoning`) into the analyzer's fields, with the comment that the analysis misses `message.get(...)` reads (e.g. gemma4). DPO only; found by the issue-reviewer.

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
  2. Done 2026-11-26: `exaone4` and `gemma4` hold through the full preprocess path, and so do `gemma4_unified` and two models' own templates (Reach today). The issue leads with the analyzer's rule, with these templates as instances.
  3. Done 2026-11-26: the duplicate searches the rate limit cut short. No duplicate. The analyzer's rule is probed on four one-line templates (`docs/evidence/2026-11-26-p0007-reach/analyzer_probe.py`): only attributes of `message` are found, `message.get("x")` is recorded as `get`, and `msg` (loop or `set`) finds nothing.
  4. The non-reasoning `</think>` question stays out of the issue.
  5. No "bug" in the title (rule 8).

## Repro
- **`findings/P-0007/repro.py`** (2026-11-26), 30 lines, no Parity code, tokenizer and template only. EXAONE-4.0-1.2B's pinned tokenizer, Axolotl's bundled `exaone4` template, the dataset config validated by Axolotl's own `SFTDataset` and dumped as its config validation does (`exclude_none=True`), then `load` + `tokenize_prompt`. One conversation: a tool call with empty content, its result, and a final answer with reasoning (`skip_think: false`). It prints the validated mappings (`{role, content}`) and the analyzer's fields (`content`, `role`), then the template's render and Axolotl's row from the first user turn on: the render has the tool call and the reasoning, and the row has neither (the tool-call turn is empty, and the last turn's think block is empty). Exit 1.
- **Runs** (the current version, after the issue-reviewer's edit; the first version printed True/False lines and also failed 3+3): 3 of 3 exit 1, with identical output, on axolotl 0.21.1 (fresh venv from `requirements.txt`, transformers 5.20.1, torch 2.14.1) and 3 of 3 on main at `de8549a` (axolotl 0.21.2.dev0, transformers 5.22.0). There is no fixed version to test. Commands and expected output: `findings/P-0007/HOW-TO-RUN.md`; lockfile `findings/P-0007/uv.lock`.
- **Checked** (`findings/P-0007/path_check.py`, `path_check.json`): the full path (`load_cfg` + `load_datasets`, EXAONE-4.0's config and tokenizer files as `base_model`) gives the same input ids and labels as the repro's direct strategy call, and with `reasoning_content` and `tool_calls` mapped explicitly the row equals the render.
- **Why EXAONE-4.0 leads, not DeepSeek V4.1:** Axolotl 0.21.1 loads EXAONE-4.0 today (D1: a fail found on a stand-in leaves the repo only with a repro that doesn't need the release's architecture). DeepSeek V4.1, Gemma 4 and the scan are evidence for the issue text, not for the repro.
- **Repro slip, caught before recording:** the first version dumped the config without `exclude_none=True`, so `field_messages` was None and the script crashed with `ValueError: Messages is null`. That also exits 1, so the exit code alone wasn't enough: the output was checked, and every run above printed the three lines.

## Upstream
- Issue URL (posted by human): https://github.com/axolotl-ai-cloud/axolotl/issues/4320 (posted 2026-11-27)
- Maintainer response: none yet
- Fixed in: not yet
- **Posted 2026-11-27** (by the human): https://github.com/axolotl-ai-cloud/axolotl/issues/4320, from `findings/P-0007/issue.md` (reviewed by the issue-reviewer, 2026-11-26). Open when recorded (created 2026-11-27T05:32:10Z, the form's default `bug` label, no comments; checked read-only 2026-11-27T05:35Z). GitHub shows the title as "[chat_template] reasoning_content and tool_calls dropped before rendering for EXAONE 4 and Gemma 4 templates". The human gave it as "[chat_template] `reasoning_content` and `tool_calls` Dropped Before Rendering for EXAONE 4 and Gemma 4 Templates" (title case, per the form's checkbox); the record keeps what GitHub shows until the two agree.
