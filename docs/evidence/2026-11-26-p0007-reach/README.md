# P-0007's reach: which templates users can train today lose fields (2026-11-26)

The human's question: Axolotl 0.21.1 can't load DeepSeek V4.1 yet, so how far does P-0007 reach with templates users
can train now? Two scripts, both in the Axolotl 0.21.1 environment (transformers 5.20.1):

- `scan_bundled.py` → `scan_bundled.json`: all 36 bundled templates, render level, Axolotl's keep rule applied as
  its code states it (role, content, and the fields `JinjaTemplateAnalyzer` lists). It tests behavior, not the
  analyzer: a field counts if the render changes when it's removed from the conversation, or when it's put back alone
  into the conversation as Axolotl keeps it.
- `full_path.py` → `full_path.json`: the affected templates through Axolotl's own `load_cfg` + `load_datasets` (what
  `axolotl preprocess` runs), on a plain OpenAI-format jsonl. Each model is one Axolotl 0.21.1 loads (its model type is
  in transformers 5.20.1); only tokenizer and config files are downloaded (rule 6). Each conversation runs with
  Axolotl's validated defaults, then with the fields mapped explicitly in `message_property_mappings`.

## Bundled templates affected: 3 of 36

| Bundled template | Fields the template reads that Axolotl drops | What the analyzer lists |
| --- | --- | --- |
| `exaone4` | `reasoning_content`, `tool_calls` | role, content (the template loops `msg = messages[i]`) |
| `gemma4` | `reasoning`, `reasoning_content`, `name`, `tool_call_id` | role, content, `get`, tool_calls, tool_responses (`message.get('x')` records `get`) |
| `gemma4_unified` | the same as `gemma4` | the same as `gemma4` |

The other 33 keep every field they read in our probes. Ten of them raise on the tool-call probe; nine don't accept a
tool role or the probe's turn order, and `deepseek_v3` concatenates the arguments as a string, while Axolotl always
passes a dict. Those raises are the same with or without the dropped fields, so they are not P-0007.

## Through Axolotl's full path

Models: `LGAI-EXAONE/EXAONE-4.0-1.2B@3abf2810673c7c0778df64a73c2d52eab32d91c4`,
`google/gemma-4-31B@5bbc2fb1c1b2c611d06e3d9f23c170ba21659d89` (Axolotl's v0.21.1 example for `chat_template: gemma4`)
and `google/gemma-4-12B-it@43c9d014b231474fa7212dbb48a762566c2b255d` (its example for `gemma4_unified`). EXAONE-4.0
and gemma-4-12B-it also run with their own template (`chat_template: tokenizer_default`); gemma-4-31B ships none.

| Template | Reasoning turn | Tool-call conversation (OpenAI format) | Mapped explicitly |
| --- | --- | --- | --- |
| `exaone4`, bundled and the model's own (`skip_think: false`) | reasoning dropped, silently | the tool call dropped from the row, silently | row = render, all trained |
| `gemma4` (gemma-4-31B) | the template renders no reasoning on such turns | preprocessing raises `TypeError: can only concatenate str (not "NoneType") to str` | row = render, all trained |
| `gemma4_unified` (gemma-4-12B-it) | the same as `gemma4` | the same `TypeError` | row = render, all trained |
| gemma-4-12B-it's own template | reasoning dropped, silently (analyzer: role, content, `get`) | reasoning and the tool call dropped, silently | row = render, all trained |

With Gemma's native form (the tool result in `tool_responses` on the assistant message, which the analyzer keeps),
the bundled `gemma4` and `gemma4_unified` rows lose the reasoning silently, and the call is kept.

## Severity by the bar (docs/DECISIONS.md), split as the human asked

- **Users today: medium.** EXAONE-4.0 (bundled and its own template) and gemma-4-12B-it with its own template lose
  reasoning and tool calls silently on ordinary OpenAI-format data. That changes labels on inputs ordinary data
  contains. The bundled Gemma 4 templates fail loudly instead on OpenAI-format tool results (an unexplained
  `TypeError`, no hint at the dropped fields), and drop reasoning silently only in Gemma's native tool format.
- **Latent: DeepSeek V4.1,** once a transformers release Axolotl uses builds `deepseek_v41`: reasoning and tool calls
  dropped silently (M5, `docs/evidence/2026-11-26-m5-axolotl-deepseek/`).
- **Workaround, in every case:** list the fields in `message_property_mappings`.

## Upstream state (read-only, 2026-11-26)

- Main (`de8549a`, 2026-11-20): `jinja_template_analyzer.py`, `gemma4.jinja` and `exaone4.jinja` have no commits since
  v0.21.1. `transform_message` and `get_chat_template_msg_variables` are identical. v0.21.1 is the latest release.
- No duplicate in issues or PRs. Searched: JinjaTemplateAnalyzer, chat_template_msg_variables, get_message_vars,
  message_property_mappings reasoning_content, reasoning_content dropped / not passed, tool_calls dropped,
  tool_call_id dropped, exaone4, exaone4 reasoning, skip_think, gemma4 tool / tool_call_id / reasoning_content /
  TypeError, concatenate str NoneType, properties on the message. Related, not duplicates:
  - #3824 (open): the DPO strategy drops `tool_calls`;
  - #3228 (open PR): adds tool fields to the empty-mapping default, a branch the validated config never reaches;
  - #3768 (open PR): bounds the analyzer's recursion depth;
  - #2260 (closed PR, 2025): added config for passing optional message fields, and asked why not pass the whole
    message.
