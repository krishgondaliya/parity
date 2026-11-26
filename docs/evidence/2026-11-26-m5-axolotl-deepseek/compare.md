# Axolotl 0.21.1 on deepseek-ai/DeepSeek-V4.1-Flash-1017@08c8657f02291e11b6f6b63e0e16014f8e748518: predicted vs actual

90 of 96 cells as predicted: thinking (primary) 42 of 48, non-thinking (secondary) 48 of 48.

Predictions: [`predictions.md`](predictions.md), committed before this run. Verdicts: `verdicts/`.

| Mode | Check | Fixture | Predicted | Actual | Match | Reason, where it differs |
| --- | --- | --- | --- | --- | --- | --- |
| thinking | t1.assistant_mask | empty_assistant | inconclusive | inconclusive | yes |  |
| thinking | t1.assistant_mask | long_answer | fail | fail | yes |  |
| thinking | t1.assistant_mask | multi_turn | fail | fail | yes |  |
| thinking | t1.assistant_mask | non_ascii | fail | fail | yes |  |
| thinking | t1.assistant_mask | reasoning | fail | inconclusive | **no** | the framework's tokens differ from its own declared template's render, so this check can't line them up with the reference; see t1.template_ |
| thinking | t1.assistant_mask | single_turn | fail | fail | yes |  |
| thinking | t1.assistant_mask | system_prompt | fail | fail | yes |  |
| thinking | t1.assistant_mask | tool_call | inconclusive | inconclusive | yes |  |
| thinking | t1.end_of_turn | empty_assistant | inconclusive | inconclusive | yes |  |
| thinking | t1.end_of_turn | long_answer | pass | pass | yes |  |
| thinking | t1.end_of_turn | multi_turn | pass | pass | yes |  |
| thinking | t1.end_of_turn | non_ascii | pass | pass | yes |  |
| thinking | t1.end_of_turn | reasoning | pass | inconclusive | **no** | the framework's tokens differ from its declared template's render before the closing token of message(s) [3], so they can't be lined up; see |
| thinking | t1.end_of_turn | single_turn | pass | pass | yes |  |
| thinking | t1.end_of_turn | system_prompt | pass | pass | yes |  |
| thinking | t1.end_of_turn | tool_call | inconclusive | inconclusive | yes |  |
| thinking | t1.multiturn_prefix | empty_assistant | inconclusive | inconclusive | yes |  |
| thinking | t1.multiturn_prefix | long_answer | not_applicable | not_applicable | yes |  |
| thinking | t1.multiturn_prefix | multi_turn | pass | pass | yes |  |
| thinking | t1.multiturn_prefix | non_ascii | not_applicable | not_applicable | yes |  |
| thinking | t1.multiturn_prefix | reasoning | pass | inconclusive | **no** | blocked by t1.template_parity: template parity already fails on this fixture (first divergence at token 63). One root cause, one finding; th |
| thinking | t1.multiturn_prefix | single_turn | not_applicable | not_applicable | yes |  |
| thinking | t1.multiturn_prefix | system_prompt | not_applicable | not_applicable | yes |  |
| thinking | t1.multiturn_prefix | tool_call | inconclusive | inconclusive | yes |  |
| thinking | t1.special_tokens | empty_assistant | inconclusive | inconclusive | yes |  |
| thinking | t1.special_tokens | long_answer | pass | pass | yes |  |
| thinking | t1.special_tokens | multi_turn | pass | pass | yes |  |
| thinking | t1.special_tokens | non_ascii | pass | pass | yes |  |
| thinking | t1.special_tokens | reasoning | pass | inconclusive | **no** | the row's tokens differ from the declared template's render, so stop-token loss is unchecked (see t1.template_parity); also planted fault(s) |
| thinking | t1.special_tokens | single_turn | pass | pass | yes |  |
| thinking | t1.special_tokens | system_prompt | pass | pass | yes |  |
| thinking | t1.special_tokens | tool_call | inconclusive | inconclusive | yes |  |
| thinking | t1.template_parity | empty_assistant | inconclusive | inconclusive | yes |  |
| thinking | t1.template_parity | long_answer | pass | pass | yes |  |
| thinking | t1.template_parity | multi_turn | pass | pass | yes |  |
| thinking | t1.template_parity | non_ascii | pass | pass | yes |  |
| thinking | t1.template_parity | reasoning | pass | fail | **no** |  |
| thinking | t1.template_parity | single_turn | pass | pass | yes |  |
| thinking | t1.template_parity | system_prompt | pass | pass | yes |  |
| thinking | t1.template_parity | tool_call | inconclusive | inconclusive | yes |  |
| thinking | t1.truncation | empty_assistant | inconclusive | inconclusive | yes |  |
| thinking | t1.truncation | long_answer | pass | pass | yes |  |
| thinking | t1.truncation | multi_turn | fail | fail | yes |  |
| thinking | t1.truncation | non_ascii | fail | fail | yes |  |
| thinking | t1.truncation | reasoning | fail | inconclusive | **no** | the framework's tokens differ from its own declared template's render, so this check can't line them up with the reference; see t1.template_ |
| thinking | t1.truncation | single_turn | fail | fail | yes |  |
| thinking | t1.truncation | system_prompt | fail | fail | yes |  |
| thinking | t1.truncation | tool_call | inconclusive | inconclusive | yes |  |
| non_thinking | t1.assistant_mask | empty_assistant | inconclusive | inconclusive | yes |  |
| non_thinking | t1.assistant_mask | long_answer | pass | pass | yes |  |
| non_thinking | t1.assistant_mask | multi_turn | pass | pass | yes |  |
| non_thinking | t1.assistant_mask | non_ascii | pass | pass | yes |  |
| non_thinking | t1.assistant_mask | reasoning | not_applicable | not_applicable | yes |  |
| non_thinking | t1.assistant_mask | single_turn | pass | pass | yes |  |
| non_thinking | t1.assistant_mask | system_prompt | pass | pass | yes |  |
| non_thinking | t1.assistant_mask | tool_call | inconclusive | inconclusive | yes |  |
| non_thinking | t1.end_of_turn | empty_assistant | inconclusive | inconclusive | yes |  |
| non_thinking | t1.end_of_turn | long_answer | pass | pass | yes |  |
| non_thinking | t1.end_of_turn | multi_turn | pass | pass | yes |  |
| non_thinking | t1.end_of_turn | non_ascii | pass | pass | yes |  |
| non_thinking | t1.end_of_turn | reasoning | not_applicable | not_applicable | yes |  |
| non_thinking | t1.end_of_turn | single_turn | pass | pass | yes |  |
| non_thinking | t1.end_of_turn | system_prompt | pass | pass | yes |  |
| non_thinking | t1.end_of_turn | tool_call | inconclusive | inconclusive | yes |  |
| non_thinking | t1.multiturn_prefix | empty_assistant | inconclusive | inconclusive | yes |  |
| non_thinking | t1.multiturn_prefix | long_answer | not_applicable | not_applicable | yes |  |
| non_thinking | t1.multiturn_prefix | multi_turn | pass | pass | yes |  |
| non_thinking | t1.multiturn_prefix | non_ascii | not_applicable | not_applicable | yes |  |
| non_thinking | t1.multiturn_prefix | reasoning | not_applicable | not_applicable | yes |  |
| non_thinking | t1.multiturn_prefix | single_turn | not_applicable | not_applicable | yes |  |
| non_thinking | t1.multiturn_prefix | system_prompt | not_applicable | not_applicable | yes |  |
| non_thinking | t1.multiturn_prefix | tool_call | inconclusive | inconclusive | yes |  |
| non_thinking | t1.special_tokens | empty_assistant | inconclusive | inconclusive | yes |  |
| non_thinking | t1.special_tokens | long_answer | pass | pass | yes |  |
| non_thinking | t1.special_tokens | multi_turn | pass | pass | yes |  |
| non_thinking | t1.special_tokens | non_ascii | pass | pass | yes |  |
| non_thinking | t1.special_tokens | reasoning | not_applicable | not_applicable | yes |  |
| non_thinking | t1.special_tokens | single_turn | pass | pass | yes |  |
| non_thinking | t1.special_tokens | system_prompt | pass | pass | yes |  |
| non_thinking | t1.special_tokens | tool_call | inconclusive | inconclusive | yes |  |
| non_thinking | t1.template_parity | empty_assistant | inconclusive | inconclusive | yes |  |
| non_thinking | t1.template_parity | long_answer | pass | pass | yes |  |
| non_thinking | t1.template_parity | multi_turn | pass | pass | yes |  |
| non_thinking | t1.template_parity | non_ascii | pass | pass | yes |  |
| non_thinking | t1.template_parity | reasoning | not_applicable | not_applicable | yes |  |
| non_thinking | t1.template_parity | single_turn | pass | pass | yes |  |
| non_thinking | t1.template_parity | system_prompt | pass | pass | yes |  |
| non_thinking | t1.template_parity | tool_call | inconclusive | inconclusive | yes |  |
| non_thinking | t1.truncation | empty_assistant | inconclusive | inconclusive | yes |  |
| non_thinking | t1.truncation | long_answer | pass | pass | yes |  |
| non_thinking | t1.truncation | multi_turn | pass | pass | yes |  |
| non_thinking | t1.truncation | non_ascii | pass | pass | yes |  |
| non_thinking | t1.truncation | reasoning | not_applicable | not_applicable | yes |  |
| non_thinking | t1.truncation | single_turn | pass | pass | yes |  |
| non_thinking | t1.truncation | system_prompt | pass | pass | yes |  |
| non_thinking | t1.truncation | tool_call | inconclusive | inconclusive | yes |  |
