# Tier 1 summary

## thinkingmachines/Inkling-2-Mini@7a67ae2314b44ef05c87fd1d4d311c58aaeb18f6

### Template properties

None found for the templates checked.

### Results: thinking mode (primary)

| Framework | Check | Result | Ran on | Notes |
| --- | --- | --- | --- | --- |
| trl@1.18.2 | t1.assistant_mask | inconclusive | 8 of 8 |  |
| trl@1.18.2 | t1.end_of_turn | inconclusive | 8 of 8 |  |
| trl@1.18.2 | t1.multiturn_prefix | inconclusive | 4 of 8 | not applicable on 4 fixture(s) |
| trl@1.18.2 | t1.special_tokens | inconclusive | 8 of 8 | blocked by t1.template_parity on 1 fixture(s) |
| trl@1.18.2 | t1.template_parity | fail | 8 of 8 | root cause: P-0008 |
| trl@1.18.2 | t1.truncation | inconclusive | 8 of 8 |  |

### Secondary results: non_thinking mode

Secondary: each framework encoded again with its documented setting for non_thinking mode (or not_configurable when it documents none), judged against a reference rendered in non_thinking mode.

| Framework | Check | Result | Ran on | Notes |
| --- | --- | --- | --- | --- |
| trl@1.18.2 | t1.assistant_mask | inconclusive | 0 of 8 | not applicable on 8 fixture(s) |
| trl@1.18.2 | t1.end_of_turn | inconclusive | 0 of 8 | not applicable on 8 fixture(s) |
| trl@1.18.2 | t1.multiturn_prefix | inconclusive | 0 of 8 | not applicable on 8 fixture(s) |
| trl@1.18.2 | t1.special_tokens | inconclusive | 0 of 8 | not applicable on 8 fixture(s) |
| trl@1.18.2 | t1.template_parity | inconclusive | 0 of 8 | not applicable on 8 fixture(s) |
| trl@1.18.2 | t1.truncation | inconclusive | 0 of 8 | not applicable on 8 fixture(s) |

### Framework properties

| Framework | Property | Declared | Verified by behavior |
| --- | --- | --- | --- |
| trl@1.18.2 | tool-call arguments given as a string | rendered verbatim | unverified: the declared template can't render string arguments (t1.multiturn_prefix on tool_call) |

### Fails by root cause

1 fail verdict(s) (1 of kind bug, 0 known differences; 1 thinking (primary), 0 non_thinking) come from 1 framework root cause(s): 1 linked, 0 untriaged (each untriaged fail counts as its own).

| Root cause | Fail verdicts | Where |
| --- | --- | --- |
| P-0008 | 1 | trl@1.18.2 t1.template_parity/tool_call |
