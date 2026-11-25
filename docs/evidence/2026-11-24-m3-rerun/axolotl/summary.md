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

Not prefix-preserving: at 5 boundary(ies) across 4 fixture(s), rendering a conversation up to an assistant turn is not the start of rendering the whole conversation. Frameworks that use this template inherit these breaks.

- `multi_turn`, boundary 2: removes `"<think>\n\n</think>\n\n"`, inserts nothing
- `multi_turn`, boundary 4: removes `"<think>\n\n</think>\n\n"`, inserts nothing
- `tool_call`, boundary 2: removes `"<think>\n\n</think>\n\n"`, inserts nothing
- `reasoning`, boundary 2: removes `"<think>\n91 is odd and its digits sum to 10, so 2 and 3 do not divide it. 91 / 7 = 13, so 91 = 7 * 13.\n</think>\n\n"`, inserts nothing
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
| axolotl@0.21.1 | t1.assistant_mask | fail | 8 of 8 | known differences: KD-0002; root cause: P-0001, P-0002 |
| axolotl@0.21.1 | t1.end_of_turn | fail | 8 of 8 | root cause: P-0002 |
| axolotl@0.21.1 | t1.multiturn_prefix | pass | 4 of 8 | prefix not preserved; the breaks are the template's (see [Qwen/Qwen3-0.6B@c1899de289a0#prefix-breaks@a55ee1b16601/thinking](#qwen-qwen3-0-6b-c1899de289a0-prefix-breaks-a55ee1b16601-thinking)); not applicable on 4 fixture(s) |
| axolotl@0.21.1 | t1.special_tokens | fail | 8 of 8 | root cause: P-0002 |
| axolotl@0.21.1 | t1.template_parity | pass | 8 of 8 |  |
| axolotl@0.21.1 | t1.truncation | fail | 8 of 8 | known differences: KD-0003; root cause: P-0001, P-0002 |

### Secondary results: non_thinking mode

Secondary: each framework encoded again with its documented setting for non_thinking mode (or not_configurable when it documents none), judged against a reference rendered in non_thinking mode.

| Framework | Check | Result | Ran on | Notes |
| --- | --- | --- | --- | --- |
| axolotl@0.21.1 | t1.assistant_mask | fail | 8 of 8 | root cause: P-0002; spans anchored past a pre-fill missing from history (see [Qwen/Qwen3-0.6B@c1899de289a0#prefill-missing@a55ee1b16601/non_thinking](#qwen-qwen3-0-6b-c1899de289a0-prefill-missing-a55ee1b16601-non-thinking)) |
| axolotl@0.21.1 | t1.end_of_turn | fail | 8 of 8 | root cause: P-0002; spans anchored past a pre-fill missing from history (see [Qwen/Qwen3-0.6B@c1899de289a0#prefill-missing@a55ee1b16601/non_thinking](#qwen-qwen3-0-6b-c1899de289a0-prefill-missing-a55ee1b16601-non-thinking)) |
| axolotl@0.21.1 | t1.multiturn_prefix | inconclusive | 4 of 8 | spans anchored past a pre-fill missing from history (see [Qwen/Qwen3-0.6B@c1899de289a0#prefill-missing@a55ee1b16601/non_thinking](#qwen-qwen3-0-6b-c1899de289a0-prefill-missing-a55ee1b16601-non-thinking)); prefix not preserved; the breaks are the template's (see [Qwen/Qwen3-0.6B@c1899de289a0#prefix-breaks@a55ee1b16601/non_thinking](#qwen-qwen3-0-6b-c1899de289a0-prefix-breaks-a55ee1b16601-non-thinking)); not applicable on 4 fixture(s) |
| axolotl@0.21.1 | t1.special_tokens | fail | 8 of 8 | root cause: P-0002; spans anchored past a pre-fill missing from history (see [Qwen/Qwen3-0.6B@c1899de289a0#prefill-missing@a55ee1b16601/non_thinking](#qwen-qwen3-0-6b-c1899de289a0-prefill-missing-a55ee1b16601-non-thinking)) |
| axolotl@0.21.1 | t1.template_parity | inconclusive | 8 of 8 |  |
| axolotl@0.21.1 | t1.truncation | fail | 8 of 8 | root cause: P-0002; spans anchored past a pre-fill missing from history (see [Qwen/Qwen3-0.6B@c1899de289a0#prefill-missing@a55ee1b16601/non_thinking](#qwen-qwen3-0-6b-c1899de289a0-prefill-missing-a55ee1b16601-non-thinking)) |

### Framework properties

| Framework | Property | Declared | Verified by behavior |
| --- | --- | --- | --- |
| axolotl@0.21.1 | tool-call arguments given as a string | parsed as JSON, then rendered like a dict | consistent (t1.multiturn_prefix on tool_call) |

### Fails by root cause

21 fail verdict(s) (8 of kind bug, 13 known differences; 17 thinking (primary), 4 non_thinking) come from 2 framework root cause(s): 2 linked, 0 untriaged (each untriaged fail counts as its own).

| Root cause | Fail verdicts | Where |
| --- | --- | --- |
| P-0001 | 13 | axolotl@0.21.1 t1.assistant_mask/long_answer; axolotl@0.21.1 t1.assistant_mask/multi_turn; axolotl@0.21.1 t1.assistant_mask/non_ascii; axolotl@0.21.1 t1.assistant_mask/reasoning; axolotl@0.21.1 t1.assistant_mask/single_turn; axolotl@0.21.1 t1.assistant_mask/system_prompt; axolotl@0.21.1 t1.assistant_mask/tool_call; axolotl@0.21.1 t1.truncation/multi_turn; axolotl@0.21.1 t1.truncation/non_ascii; axolotl@0.21.1 t1.truncation/reasoning; axolotl@0.21.1 t1.truncation/single_turn; axolotl@0.21.1 t1.truncation/system_prompt; axolotl@0.21.1 t1.truncation/tool_call |
| P-0002 | 8 | axolotl@0.21.1 t1.assistant_mask/empty_assistant; axolotl@0.21.1 t1.assistant_mask/empty_assistant (non_thinking); axolotl@0.21.1 t1.end_of_turn/empty_assistant; axolotl@0.21.1 t1.end_of_turn/empty_assistant (non_thinking); axolotl@0.21.1 t1.special_tokens/empty_assistant; axolotl@0.21.1 t1.special_tokens/empty_assistant (non_thinking); axolotl@0.21.1 t1.truncation/empty_assistant; axolotl@0.21.1 t1.truncation/empty_assistant (non_thinking) |
