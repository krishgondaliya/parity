# P-0007: Axolotl drops reasoning and tool calls from DeepSeek V4.1 conversations before rendering them

- Status: candidate
- Severity: medium (effect: reasoning traces and tool calls vanish from the training rows, so their tokens get no loss and the model never learns to close its reasoning with `</think>`; reach: every DeepSeek V4.1 conversation with reasoning or tool calls, which ordinary training data contains), by effect x reach (docs/DECISIONS.md)
- Date found: 2026-11-26
- Model: deepseek-ai/DeepSeek-V4.1-Flash-1017@08c8657f02291e11b6f6b63e0e16014f8e748518 (its tokenizer and chat template; the mechanism needs no architecture)
- Framework: axolotl@0.21.1   (lockfile: envs/axolotl-0.21.1/uv.lock)
- Check: t1.template_parity (reasoning), t1.assistant_mask and t1.truncation (the untrained `</think>`)   Fixture: reasoning, single_turn, multi_turn, system_prompt, long_answer, non_ascii   Kind: bug
- Verdict files: docs/evidence/2026-11-26-m5-axolotl-deepseek/verdicts/thinking/ (M5, the `deepseek_v4` stand-in; D1)
- Predicted before the run: the `</think>` fails yes; the dropped reasoning no (`docs/evidence/2026-11-26-m5-axolotl-deepseek/predictions.md`).

## Evidence
- **Reasoning dropped** (M5 `t1.template_parity` on `reasoning`, first divergence at token 63): the template renders the last turn as `<｜Assistant｜><think>97 is not divisible by 2, 3, 5 or 7, …</think>Yes, 97 is prime.<｜end▁of▁sentence｜>`; Axolotl's row has `<｜Assistant｜><think></think>Yes, 97 is prime.<｜end▁of▁sentence｜>` (72 tokens, against 105).
- **Tool calls dropped** (`docs/evidence/2026-11-26-m5-axolotl-deepseek/triage_fields.py`, the same full path; our reference refuses this fixture on DeepSeek, so M5's cells can't show it): the template renders the tool-call turn as `<｜Assistant｜><think></think>\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="get_weather">…</｜DSML｜ calls><｜end▁of▁sentence｜>`; Axolotl's row has `<｜Assistant｜><think></think><｜end▁of▁sentence｜>`. The tool's schema in the system prompt survives (it comes from `tools`); the call itself is gone.
- **`</think>` untrained** (M5 `t1.assistant_mask` on single_turn, multi_turn, system_prompt, long_answer, non_ascii; `triage_think_close.py`): in the last turn, `<｜Assistant｜>` and `<think>` are prompt side and untrained, as they should be; `</think>` is must-train and untrained; the content and `<｜end▁of▁sentence｜>` are trained.
- **The model generates `</think>`** (`triage_encoding_think.py`, the model's own `encoding/encoding.py`, sandboxed): in thinking mode the prompt ends `<｜Assistant｜><think>`, and the model generates `</think>Red, yellow and blue.<｜end▁of▁sentence｜>`, or `<reasoning></think>…` when there is reasoning. The same after a history turn.
- **Qwen3, the same path:** reasoning and tool calls reach the template and the rows equal the render (`triage_fields.json`, `triage_reasoning.json`).

## Mechanism (axolotl 0.21.1; the cited files are byte-identical to the v0.21.1 tag where checked)
1. Axolotl's config validation fills `message_property_mappings: {role: role, content: content}` (the validated dataset config, `triage_reasoning.json`). With a non-empty mapping, `ChatTemplatePrompter.__init__` doesn't add the `reasoning_content` mapping (`prompt_strategies/chat_template.py` L84-L88: it adds it only when the mappings are empty).
2. `transform_message` (L1325-L1400) keeps the mapped fields, then any other field only if it is among `chat_template_msg_variables`: "Keep only the properties defined in the chat template and not already mapped".
3. Those come from `JinjaTemplateAnalyzer.get_message_vars("messages")` (`prompt_strategies/jinja_template_analyzer.py` L319-L335). It returns the properties of a variable literally named `message`, plus whatever its loop tracking links to `messages`. That tracking records nothing even for Qwen3's `for message in messages` (`iteration_targets` is empty), so in practice only a loop variable named `message` counts.
4. DeepSeek V4.1's template loops `for msg in messages` (L84) and reads reasoning as `msg.reasoning_content` / `msg.get("thinking")`. It reads tool calls later as `m.tool_calls` over its own list (L130-L133). The analyzer finds no message variables at all (`[]`; Qwen3: content, reasoning_content, role, tool_calls, …; Inkling and Hy4 also find theirs).
   - Renaming `msg` to `message` in a copy of the template makes it find `reasoning_content` (`tool_calls` would still be missed: they're read through `m`).
5. So `reasoning_content`, `tool_calls`, `tool_call_id` and `name` are dropped from every message before rendering. The only trace is a DEBUG-level log line listing the (empty) property set (L394-L396).
6. **The `</think>`:** Axolotl labels each assistant turn from its first field, reasoning_content else content, to its end (KD-0002's summary; L728-L756). On DeepSeek every turn reaches the template without reasoning, so the span starts at the content, and the template-emitted `</think>` before it is never trained.

## Why DeepSeek differs from Qwen3
- **Turns with reasoning:** on Qwen3 the reasoning reaches the template, so the span starts at the reasoning and `</think>` is trained. Axolotl's own test asserts exactly that (`tests/prompt_strategies/test_chat_templates_thinking.py` L143-L177, trained text `greet</think>hello<|im_end|>…`). On DeepSeek the reasoning never arrives (above), so neither the reasoning nor `</think>` is trained.
- **Turns without reasoning:** the two behave the same. On Qwen3, Axolotl leaves the whole empty think block `<think>\n\n</think>\n\n` untrained (`triage_think_close.json`: `</think>` unlabeled on single_turn and multi_turn), which our KD-0002/KD-0003 entries already cover for Qwen3's template. On DeepSeek the untrained `</think>` on non-reasoning turns is the same first-field rule. **Open for the human:** the guidance for this triage assumed Qwen3 trains `</think>`. That holds for turns with reasoning only, so whether the first-field rule on non-reasoning thinking-mode turns is a known difference (as on Qwen3) or part of the candidate, on both templates, is the human's call. This record doesn't extend KD-0002/KD-0003.

## Triage (from /triage)
- **Calibration green on the known-good model:** yes (M4; re-calibrated 96/96 after the adapter change, `docs/evidence/2026-11-26-m4-recalibration/`).
- **Adapter uses the documented public path and option:** yes. `type: chat_template`, the tokenizer's own template, `roles_to_train: [assistant]`, `train_on_eos: turn`, as on Qwen3. The mappings are Axolotl's validated defaults; the adapter sets none.
- **Fixture valid for this model:** yes. `reasoning_content` is the field DeepSeek's template reads (and encoding.py's); the tool call follows the OpenAI format the template converts.
- **Reference consistent:** yes. The template agrees with the model's own `encoding.py` on all 8 fixtures (`docs/evidence/2026-11-25-deepseek-encoding/`), and encoding.py shows `</think>` is generated.
- **Stand-in (D1):** the M5 run used `deepseek_v4` for the model, but nothing above reads the model: the validated mappings, the analyzer and `transform_message` see only the config and the template. A repro needs only the release's tokenizer and template (D1's condition for leaving the repo).
- **Documented as intended behavior:** no. Axolotl's docs (`docs/dataset-formats/conversation.qmd` at v0.21.1, L385-L405) say: for templates that support a separate `reasoning_content` field, "No additional configuration is needed beyond what the template already requires". DeepSeek V4.1's template supports it. Nothing documents that fields are dropped when the analyzer can't see them.
- **Workaround, not documented for this case:** listing the fields in `message_property_mappings` (e.g. `reasoning_content: reasoning_content`, `tool_calls: tool_calls`) would map them explicitly. Not tested here.
- **Duplicates searched** (read-only, issues and PRs): "JinjaTemplateAnalyzer", "chat_template_msg_variables", "reasoning_content dropped", "tool_calls missing chat_template", "message_property_mappings reasoning_content", "deepseek v4 tool_calls", "properties on the message". No duplicate. Related:
  - #3824 (open): the DPO `chat_template` strategy drops `tool_calls`, a different strategy;
  - #3768 (open PR): the analyzer raises `RecursionError` on deeply nested templates, a loud failure, not this.

## Skeptic verdict
(pending: the skeptic subagent's review goes here)

## Repro
- Not written yet. It needs only the tokenizer and template (the stand-in plays no part), so a standalone repro is possible (D1).

## Upstream
- Issue URL (posted by human): none
- Maintainer response: none yet
- Fixed in: not yet
