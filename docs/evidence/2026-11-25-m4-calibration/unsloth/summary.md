> **Unsloth's calibration run is its GPU route** (Colab kit, RUN = "full"; verified in `provenance.json`). Non-thinking mode is not configurable for Unsloth (P-0006, the human's decision). The kit was built at commit `b73da05`, before the serving-mode applicability rule (2026-11-25), and that rule is the only code change since. So the six `reasoning` × non-thinking cells record `not_configurable`, where today's runner would record `not_applicable`. Neither is judged, and neither counts toward green. Every judged cell is unaffected.

# Tier 1 summary

## Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca

### Template properties

<a id="qwen-qwen3-0-6b-c1899de289a0-prefill-missing-a55ee1b16601-non-thinking"></a>
**Qwen/Qwen3-0.6B@c1899de289a0#prefill-missing@a55ee1b16601/non_thinking** (prefill_missing_from_history; template `model`, sha256 `a55ee1b16601`, non_thinking mode)

The generation prompt ends in a pre-fill ("<think>\n\n</think>\n\n") that earlier assistant turns render without. Where the reference can delimit those turns, it anchors their content where the two renders stop agreeing. Also a train/serve mismatch for Tier 3: served turns start with the pre-fill, trained history turns don't.

- `multi_turn`: message 1, message 3 render without the pre-fill
- `tool_call`: message 1 render without the pre-fill
- `empty_assistant`: message 1 render without the pre-fill

<a id="qwen-qwen3-0-6b-c1899de289a0-prefix-breaks-a55ee1b16601-non-thinking"></a>
**Qwen/Qwen3-0.6B@c1899de289a0#prefix-breaks@a55ee1b16601/non_thinking** (prefix_breaks; template `model`, sha256 `a55ee1b16601`, non_thinking mode)

Not prefix-preserving: at 4 boundary(ies) across 3 fixture(s), rendering a conversation up to an assistant turn is not the start of rendering the whole conversation. Frameworks that use this template inherit these breaks.

- `multi_turn`, boundary 2: removes `"<think>\n\n</think>\n\n"`, inserts nothing
- `multi_turn`, boundary 4: removes `"<think>\n\n</think>\n\n"`, inserts nothing
- `tool_call`, boundary 2: removes `"<think>\n\n</think>\n\n"`, inserts nothing
- `empty_assistant`, boundary 2: removes `"<think>\n\n</think>\n\n"`, inserts nothing

<a id="qwen-qwen3-0-6b-c1899de289a0-prefix-breaks-a55ee1b16601-thinking"></a>
**Qwen/Qwen3-0.6B@c1899de289a0#prefix-breaks@a55ee1b16601/thinking** (prefix_breaks; template `model`, sha256 `a55ee1b16601`, thinking mode)

Not prefix-preserving: at 5 boundary(ies) across 4 fixture(s), rendering a conversation up to an assistant turn is not the start of rendering the whole conversation. Frameworks that use this template inherit these breaks.

- `multi_turn`, boundary 2: removes `"<think>\n\n</think>\n\n"`, inserts nothing
- `multi_turn`, boundary 4: removes `"<think>\n\n</think>\n\n"`, inserts nothing
- `tool_call`, boundary 2: removes `"<think>\n\n</think>\n\n"`, inserts nothing
- `reasoning`, boundary 2: removes `"<think>\n91 is odd and its digits sum to 10, so 2 and 3 do not divide it. 91 / 7 = 13, so 91 = 7 * 13.\n</think>\n\n"`, inserts nothing
- `empty_assistant`, boundary 2: removes `"<think>\n\n</think>\n\n"`, inserts nothing

### Results: thinking mode (primary)

| Framework | Check | Result | Ran on | Notes |
| --- | --- | --- | --- | --- |
| unsloth@2026.11.10 (gpu) | t1.assistant_mask | pass | 8 of 8 |  |
| unsloth@2026.11.10 (gpu) | t1.end_of_turn | pass | 8 of 8 |  |
| unsloth@2026.11.10 (gpu) | t1.multiturn_prefix | pass | 4 of 8 | prefix not preserved; the breaks are the template's (see [Qwen/Qwen3-0.6B@c1899de289a0#prefix-breaks@a55ee1b16601/thinking](#qwen-qwen3-0-6b-c1899de289a0-prefix-breaks-a55ee1b16601-thinking)); not applicable on 4 fixture(s) |
| unsloth@2026.11.10 (gpu) | t1.special_tokens | fail | 8 of 8 | root cause: P-0005 |
| unsloth@2026.11.10 (gpu) | t1.template_parity | pass | 8 of 8 |  |
| unsloth@2026.11.10 (gpu) | t1.truncation | pass | 8 of 8 |  |

### Secondary results: non_thinking mode

Secondary: each framework encoded again with its documented setting for non_thinking mode (or not_configurable when it documents none), judged against a reference rendered in non_thinking mode.

| Framework | Check | Result | Ran on | Notes |
| --- | --- | --- | --- | --- |
| unsloth@2026.11.10 (gpu) | t1.assistant_mask | not_configurable | 0 of 8 | not configurable on 8 fixture(s): serving mode: Unsloth documents non-thinking training via the /no_think soft switch; hard-switch training is not configurable. |
| unsloth@2026.11.10 (gpu) | t1.end_of_turn | not_configurable | 0 of 8 | not configurable on 8 fixture(s): serving mode: Unsloth documents non-thinking training via the /no_think soft switch; hard-switch training is not configurable. |
| unsloth@2026.11.10 (gpu) | t1.multiturn_prefix | not_configurable | 0 of 8 | not applicable on 4 fixture(s); not configurable on 4 fixture(s): serving mode: Unsloth documents non-thinking training via the /no_think soft switch; hard-switch training is not configurable. |
| unsloth@2026.11.10 (gpu) | t1.special_tokens | not_configurable | 0 of 8 | not configurable on 8 fixture(s): serving mode: Unsloth documents non-thinking training via the /no_think soft switch; hard-switch training is not configurable. |
| unsloth@2026.11.10 (gpu) | t1.template_parity | not_configurable | 0 of 8 | not configurable on 8 fixture(s): serving mode: Unsloth documents non-thinking training via the /no_think soft switch; hard-switch training is not configurable. |
| unsloth@2026.11.10 (gpu) | t1.truncation | not_configurable | 0 of 8 | not configurable on 8 fixture(s): serving mode: Unsloth documents non-thinking training via the /no_think soft switch; hard-switch training is not configurable. |

### Framework properties

| Framework | Property | Declared | Verified by behavior |
| --- | --- | --- | --- |
| unsloth@2026.11.10 (gpu) | tool-call arguments given as a string | rendered verbatim | consistent (t1.multiturn_prefix on tool_call) |

### Fails by root cause

8 fail verdict(s) (8 of kind bug, 0 known differences; 8 thinking (primary), 0 non_thinking) come from 1 framework root cause(s): 1 linked, 0 untriaged (each untriaged fail counts as its own).

| Root cause | Fail verdicts | Where |
| --- | --- | --- |
| P-0005 | 8 | unsloth@2026.11.10 (gpu) t1.special_tokens/empty_assistant; unsloth@2026.11.10 (gpu) t1.special_tokens/long_answer; unsloth@2026.11.10 (gpu) t1.special_tokens/multi_turn; unsloth@2026.11.10 (gpu) t1.special_tokens/non_ascii; unsloth@2026.11.10 (gpu) t1.special_tokens/reasoning; unsloth@2026.11.10 (gpu) t1.special_tokens/single_turn; unsloth@2026.11.10 (gpu) t1.special_tokens/system_prompt; unsloth@2026.11.10 (gpu) t1.special_tokens/tool_call |
