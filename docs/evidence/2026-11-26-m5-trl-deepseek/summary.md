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
| trl@1.18.2 | t1.assistant_mask | not_supported | 0 of 8 | not supported on 8 fixture(s): framework refuses: trl.no_training_template: builtins.ValueError: The chat template is not training-compatible (missing prefix-preservation or `{% generation %}` markers) and patching is not supported for this template. Please manually modify the chat template for training. (https://github.com/huggingface/trl/blob/v1.18.2/trl/chat_template_utils.py#L1513-L1516) |
| trl@1.18.2 | t1.end_of_turn | not_supported | 0 of 8 | not supported on 8 fixture(s): framework refuses: trl.no_training_template: builtins.ValueError: The chat template is not training-compatible (missing prefix-preservation or `{% generation %}` markers) and patching is not supported for this template. Please manually modify the chat template for training. (https://github.com/huggingface/trl/blob/v1.18.2/trl/chat_template_utils.py#L1513-L1516) |
| trl@1.18.2 | t1.multiturn_prefix | not_supported | 0 of 8 | not applicable on 4 fixture(s); not supported on 4 fixture(s): framework refuses: trl.no_training_template: builtins.ValueError: The chat template is not training-compatible (missing prefix-preservation or `{% generation %}` markers) and patching is not supported for this template. Please manually modify the chat template for training. (https://github.com/huggingface/trl/blob/v1.18.2/trl/chat_template_utils.py#L1513-L1516) |
| trl@1.18.2 | t1.special_tokens | not_supported | 0 of 8 | not supported on 8 fixture(s): framework refuses: trl.no_training_template: builtins.ValueError: The chat template is not training-compatible (missing prefix-preservation or `{% generation %}` markers) and patching is not supported for this template. Please manually modify the chat template for training. (https://github.com/huggingface/trl/blob/v1.18.2/trl/chat_template_utils.py#L1513-L1516) |
| trl@1.18.2 | t1.template_parity | not_supported | 0 of 8 | not supported on 8 fixture(s): framework refuses: trl.no_training_template: builtins.ValueError: The chat template is not training-compatible (missing prefix-preservation or `{% generation %}` markers) and patching is not supported for this template. Please manually modify the chat template for training. (https://github.com/huggingface/trl/blob/v1.18.2/trl/chat_template_utils.py#L1513-L1516) |
| trl@1.18.2 | t1.truncation | not_supported | 0 of 8 | not supported on 8 fixture(s): framework refuses: trl.no_training_template: builtins.ValueError: The chat template is not training-compatible (missing prefix-preservation or `{% generation %}` markers) and patching is not supported for this template. Please manually modify the chat template for training. (https://github.com/huggingface/trl/blob/v1.18.2/trl/chat_template_utils.py#L1513-L1516) |

### Secondary results: non_thinking mode

Secondary: each framework encoded again with its documented setting for non_thinking mode (or not_configurable when it documents none), judged against a reference rendered in non_thinking mode.

| Framework | Check | Result | Ran on | Notes |
| --- | --- | --- | --- | --- |
| trl@1.18.2 | t1.assistant_mask | not_supported | 0 of 8 | not applicable on 1 fixture(s); not supported on 7 fixture(s): framework refuses: trl.no_training_template: builtins.ValueError: The chat template is not training-compatible (missing prefix-preservation or `{% generation %}` markers) and patching is not supported for this template. Please manually modify the chat template for training. (https://github.com/huggingface/trl/blob/v1.18.2/trl/chat_template_utils.py#L1513-L1516) |
| trl@1.18.2 | t1.end_of_turn | not_supported | 0 of 8 | not applicable on 1 fixture(s); not supported on 7 fixture(s): framework refuses: trl.no_training_template: builtins.ValueError: The chat template is not training-compatible (missing prefix-preservation or `{% generation %}` markers) and patching is not supported for this template. Please manually modify the chat template for training. (https://github.com/huggingface/trl/blob/v1.18.2/trl/chat_template_utils.py#L1513-L1516) |
| trl@1.18.2 | t1.multiturn_prefix | not_supported | 0 of 8 | not applicable on 5 fixture(s); not supported on 3 fixture(s): framework refuses: trl.no_training_template: builtins.ValueError: The chat template is not training-compatible (missing prefix-preservation or `{% generation %}` markers) and patching is not supported for this template. Please manually modify the chat template for training. (https://github.com/huggingface/trl/blob/v1.18.2/trl/chat_template_utils.py#L1513-L1516) |
| trl@1.18.2 | t1.special_tokens | not_supported | 0 of 8 | not applicable on 1 fixture(s); not supported on 7 fixture(s): framework refuses: trl.no_training_template: builtins.ValueError: The chat template is not training-compatible (missing prefix-preservation or `{% generation %}` markers) and patching is not supported for this template. Please manually modify the chat template for training. (https://github.com/huggingface/trl/blob/v1.18.2/trl/chat_template_utils.py#L1513-L1516) |
| trl@1.18.2 | t1.template_parity | not_supported | 0 of 8 | not applicable on 1 fixture(s); not supported on 7 fixture(s): framework refuses: trl.no_training_template: builtins.ValueError: The chat template is not training-compatible (missing prefix-preservation or `{% generation %}` markers) and patching is not supported for this template. Please manually modify the chat template for training. (https://github.com/huggingface/trl/blob/v1.18.2/trl/chat_template_utils.py#L1513-L1516) |
| trl@1.18.2 | t1.truncation | not_supported | 0 of 8 | not applicable on 1 fixture(s); not supported on 7 fixture(s): framework refuses: trl.no_training_template: builtins.ValueError: The chat template is not training-compatible (missing prefix-preservation or `{% generation %}` markers) and patching is not supported for this template. Please manually modify the chat template for training. (https://github.com/huggingface/trl/blob/v1.18.2/trl/chat_template_utils.py#L1513-L1516) |

### Fails by root cause

No fails.
