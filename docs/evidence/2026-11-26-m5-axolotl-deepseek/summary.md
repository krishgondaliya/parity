# Tier 1 summary

## deepseek-ai/DeepSeek-V4.1-Flash-1017@08c8657f02291e11b6f6b63e0e16014f8e748518

### Template properties

<a id="deepseek-ai-deepseek-v4-1-flash-1017-08c8657f0229-prefill-missing-95b7033a693b-thinking"></a>
**deepseek-ai/DeepSeek-V4.1-Flash-1017@08c8657f0229#prefill-missing@95b7033a693b/thinking** (prefill_missing_from_history; template `model`, sha256 `95b7033a693b`, thinking mode)

The generation prompt ends in a pre-fill ("<think>") that earlier assistant turns render without. Where the reference can delimit those turns, it anchors their content where the two renders stop agreeing. Also a train/serve mismatch for Tier 3: served turns start with the pre-fill, trained history turns don't.

- `multi_turn`: message 1, message 3 render without the pre-fill
- `reasoning`: message 1 render without the pre-fill
- `empty_assistant`: message 1 render without the pre-fill

<a id="deepseek-ai-deepseek-v4-1-flash-1017-08c8657f0229-prefix-breaks-95b7033a693b-thinking"></a>
**deepseek-ai/DeepSeek-V4.1-Flash-1017@08c8657f0229#prefix-breaks@95b7033a693b/thinking** (prefix_breaks; template `model`, sha256 `95b7033a693b`, thinking mode)

Not prefix-preserving: at 4 boundary(ies) across 3 fixture(s), rendering a conversation up to an assistant turn is not the start of rendering the whole conversation. Frameworks that use this template inherit these breaks.

- `multi_turn`, boundary 2: removes `"<think>"`, inserts nothing
- `multi_turn`, boundary 4: removes `"<think>"`, inserts nothing
- `reasoning`, boundary 2: removes `"<think>91 is odd and its digits sum to 10, so 2 and 3 do not divide it. 91 / 7 = 13, so 91 = 7 * 13."`, inserts nothing
- `empty_assistant`, boundary 2: removes `"<think>"`, inserts nothing

### Results: thinking mode (primary)

| Framework | Check | Result | Ran on | Notes |
| --- | --- | --- | --- | --- |
| axolotl@0.21.1 | t1.assistant_mask | fail | 8 of 8 | 5 untriaged fail(s) |
| axolotl@0.21.1 | t1.end_of_turn | inconclusive | 8 of 8 |  |
| axolotl@0.21.1 | t1.multiturn_prefix | inconclusive | 4 of 8 | prefix not preserved; the breaks are the template's (see [deepseek-ai/DeepSeek-V4.1-Flash-1017@08c8657f0229#prefix-breaks@95b7033a693b/thinking](#deepseek-ai-deepseek-v4-1-flash-1017-08c8657f0229-prefix-breaks-95b7033a693b-thinking)); blocked by t1.template_parity on 1 fixture(s); not applicable on 4 fixture(s) |
| axolotl@0.21.1 | t1.special_tokens | inconclusive | 8 of 8 |  |
| axolotl@0.21.1 | t1.template_parity | fail | 8 of 8 | 1 untriaged fail(s) |
| axolotl@0.21.1 | t1.truncation | fail | 8 of 8 | 4 untriaged fail(s) |

### Secondary results: non_thinking mode

Secondary: each framework encoded again with its documented setting for non_thinking mode (or not_configurable when it documents none), judged against a reference rendered in non_thinking mode.

| Framework | Check | Result | Ran on | Notes |
| --- | --- | --- | --- | --- |
| axolotl@0.21.1 | t1.assistant_mask | inconclusive | 7 of 8 | not applicable on 1 fixture(s) |
| axolotl@0.21.1 | t1.end_of_turn | inconclusive | 7 of 8 | not applicable on 1 fixture(s) |
| axolotl@0.21.1 | t1.multiturn_prefix | inconclusive | 3 of 8 | not applicable on 5 fixture(s) |
| axolotl@0.21.1 | t1.special_tokens | inconclusive | 7 of 8 | not applicable on 1 fixture(s) |
| axolotl@0.21.1 | t1.template_parity | inconclusive | 7 of 8 | not applicable on 1 fixture(s) |
| axolotl@0.21.1 | t1.truncation | inconclusive | 7 of 8 | not applicable on 1 fixture(s) |

### Framework properties

| Framework | Property | Declared | Verified by behavior |
| --- | --- | --- | --- |
| axolotl@0.21.1 | tool-call arguments given as a string | parsed as JSON, then rendered like a dict | unverified |

### Fails by root cause

10 fail verdict(s) (10 of kind bug, 0 known differences; 10 thinking (primary), 0 non_thinking) come from 10 framework root cause(s): 0 linked, 10 untriaged (each untriaged fail counts as its own).

| Root cause | Fail verdicts | Where |
| --- | --- | --- |
| untriaged | 1 | axolotl@0.21.1 t1.assistant_mask/long_answer (thinking) |
| untriaged | 1 | axolotl@0.21.1 t1.assistant_mask/multi_turn (thinking) |
| untriaged | 1 | axolotl@0.21.1 t1.assistant_mask/non_ascii (thinking) |
| untriaged | 1 | axolotl@0.21.1 t1.assistant_mask/single_turn (thinking) |
| untriaged | 1 | axolotl@0.21.1 t1.assistant_mask/system_prompt (thinking) |
| untriaged | 1 | axolotl@0.21.1 t1.template_parity/reasoning (thinking) |
| untriaged | 1 | axolotl@0.21.1 t1.truncation/multi_turn (thinking) |
| untriaged | 1 | axolotl@0.21.1 t1.truncation/non_ascii (thinking) |
| untriaged | 1 | axolotl@0.21.1 t1.truncation/single_turn (thinking) |
| untriaged | 1 | axolotl@0.21.1 t1.truncation/system_prompt (thinking) |
