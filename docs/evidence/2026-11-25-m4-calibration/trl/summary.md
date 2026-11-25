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
| trl@1.18.2 | t1.assistant_mask | pass | 8 of 8 |  |
| trl@1.18.2 | t1.end_of_turn | pass | 8 of 8 |  |
| trl@1.18.2 | t1.multiturn_prefix | pass | 4 of 8 | not applicable on 4 fixture(s) |
| trl@1.18.2 | t1.special_tokens | pass | 8 of 8 |  |
| trl@1.18.2 | t1.template_parity | difference | 8 of 8 | known differences: KD-0001; root cause: KD-0001 |
| trl@1.18.2 | t1.truncation | pass | 8 of 8 |  |

### Secondary results: non_thinking mode

Secondary: each framework encoded again with its documented setting for non_thinking mode (or not_configurable when it documents none), judged against a reference rendered in non_thinking mode.

| Framework | Check | Result | Ran on | Notes |
| --- | --- | --- | --- | --- |
| trl@1.18.2 | t1.assistant_mask | pass | 7 of 8 | not applicable on 1 fixture(s) |
| trl@1.18.2 | t1.end_of_turn | pass | 7 of 8 | not applicable on 1 fixture(s) |
| trl@1.18.2 | t1.multiturn_prefix | pass | 3 of 8 | not applicable on 5 fixture(s) |
| trl@1.18.2 | t1.special_tokens | pass | 7 of 8 | not applicable on 1 fixture(s) |
| trl@1.18.2 | t1.template_parity | difference | 7 of 8 | known differences: KD-0001; root cause: KD-0001; not applicable on 1 fixture(s) |
| trl@1.18.2 | t1.truncation | pass | 7 of 8 | not applicable on 1 fixture(s) |

### Framework properties

| Framework | Property | Declared | Verified by behavior |
| --- | --- | --- | --- |
| trl@1.18.2 | tool-call arguments given as a string | rendered verbatim | consistent (t1.multiturn_prefix on tool_call) |

### Fails by root cause

7 fail verdict(s) (0 of kind bug, 7 known differences; 4 thinking (primary), 3 non_thinking) come from 1 framework root cause(s): 1 linked, 0 untriaged (each untriaged fail counts as its own).

| Root cause | Fail verdicts | Where |
| --- | --- | --- |
| KD-0001 | 7 | trl@1.18.2 t1.template_parity/empty_assistant; trl@1.18.2 t1.template_parity/empty_assistant (non_thinking); trl@1.18.2 t1.template_parity/multi_turn; trl@1.18.2 t1.template_parity/multi_turn (non_thinking); trl@1.18.2 t1.template_parity/reasoning; trl@1.18.2 t1.template_parity/tool_call; trl@1.18.2 t1.template_parity/tool_call (non_thinking) |
