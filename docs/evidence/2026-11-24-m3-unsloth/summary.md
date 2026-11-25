> **Backend: CPU** (Unsloth's CPU fallback; `envs/unsloth-2026.11.10/environment.json`). These are Unsloth's CPU-backend results. Once a GPU run exists, Unsloth's GPU route is its primary result and this becomes the labeled secondary "CPU backend" table (the human's decision, 2026-11-24).

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
| unsloth@2026.11.10 (cpu) | t1.assistant_mask | inconclusive | 8 of 8 |  |
| unsloth@2026.11.10 (cpu) | t1.end_of_turn | pass | 8 of 8 |  |
| unsloth@2026.11.10 (cpu) | t1.multiturn_prefix | inconclusive | 4 of 8 | blocked by t1.template_parity on 4 fixture(s); not applicable on 4 fixture(s) |
| unsloth@2026.11.10 (cpu) | t1.special_tokens | inconclusive | 8 of 8 | blocked by t1.template_parity on 8 fixture(s) |
| unsloth@2026.11.10 (cpu) | t1.template_parity | fail | 8 of 8 | root cause: P-0004 |
| unsloth@2026.11.10 (cpu) | t1.truncation | inconclusive | 8 of 8 |  |

### Secondary results: non_thinking mode

Secondary: the same encoded rows as the primary results, judged against a second reference rendered in non_thinking mode. No extra framework runs.

| Framework | Check | Result | Ran on | Notes |
| --- | --- | --- | --- | --- |
| unsloth@2026.11.10 (cpu) | t1.assistant_mask | inconclusive | 8 of 8 | spans anchored past a pre-fill missing from history (see [Qwen/Qwen3-0.6B@c1899de289a0#prefill-missing@a55ee1b16601/non_thinking](#qwen-qwen3-0-6b-c1899de289a0-prefill-missing-a55ee1b16601-non-thinking)) |
| unsloth@2026.11.10 (cpu) | t1.end_of_turn | inconclusive | 8 of 8 | spans anchored past a pre-fill missing from history (see [Qwen/Qwen3-0.6B@c1899de289a0#prefill-missing@a55ee1b16601/non_thinking](#qwen-qwen3-0-6b-c1899de289a0-prefill-missing-a55ee1b16601-non-thinking)) |
| unsloth@2026.11.10 (cpu) | t1.multiturn_prefix | inconclusive | 4 of 8 | spans anchored past a pre-fill missing from history (see [Qwen/Qwen3-0.6B@c1899de289a0#prefill-missing@a55ee1b16601/non_thinking](#qwen-qwen3-0-6b-c1899de289a0-prefill-missing-a55ee1b16601-non-thinking)); blocked by t1.template_parity on 3 fixture(s); not applicable on 4 fixture(s) |
| unsloth@2026.11.10 (cpu) | t1.special_tokens | inconclusive | 8 of 8 | spans anchored past a pre-fill missing from history (see [Qwen/Qwen3-0.6B@c1899de289a0#prefill-missing@a55ee1b16601/non_thinking](#qwen-qwen3-0-6b-c1899de289a0-prefill-missing-a55ee1b16601-non-thinking)); blocked by t1.template_parity on 7 fixture(s) |
| unsloth@2026.11.10 (cpu) | t1.template_parity | fail | 8 of 8 | root cause: P-0004 |
| unsloth@2026.11.10 (cpu) | t1.truncation | inconclusive | 8 of 8 | spans anchored past a pre-fill missing from history (see [Qwen/Qwen3-0.6B@c1899de289a0#prefill-missing@a55ee1b16601/non_thinking](#qwen-qwen3-0-6b-c1899de289a0-prefill-missing-a55ee1b16601-non-thinking)) |

### Framework properties

| Framework | Property | Declared | Verified by behavior |
| --- | --- | --- | --- |
| unsloth@2026.11.10 (cpu) | tool-call arguments given as a string | rendered verbatim | mismatch: the tokens match neither the verbatim nor the re-serialized render (t1.multiturn_prefix on tool_call) |

### Fails by root cause

15 fail verdict(s) (15 of kind bug, 0 known differences; 8 thinking (primary), 7 non_thinking) come from 1 framework root cause(s): 1 linked, 0 untriaged (each untriaged fail counts as its own).

| Root cause | Fail verdicts | Where |
| --- | --- | --- |
| P-0004 (Unsloth CPU backend only; low impact) | 15 | unsloth@2026.11.10 (cpu) t1.template_parity/empty_assistant; unsloth@2026.11.10 (cpu) t1.template_parity/empty_assistant (non_thinking); unsloth@2026.11.10 (cpu) t1.template_parity/long_answer; unsloth@2026.11.10 (cpu) t1.template_parity/long_answer (non_thinking); unsloth@2026.11.10 (cpu) t1.template_parity/multi_turn; unsloth@2026.11.10 (cpu) t1.template_parity/multi_turn (non_thinking); unsloth@2026.11.10 (cpu) t1.template_parity/non_ascii; unsloth@2026.11.10 (cpu) t1.template_parity/non_ascii (non_thinking); unsloth@2026.11.10 (cpu) t1.template_parity/reasoning; unsloth@2026.11.10 (cpu) t1.template_parity/single_turn; unsloth@2026.11.10 (cpu) t1.template_parity/single_turn (non_thinking); unsloth@2026.11.10 (cpu) t1.template_parity/system_prompt; unsloth@2026.11.10 (cpu) t1.template_parity/system_prompt (non_thinking); unsloth@2026.11.10 (cpu) t1.template_parity/tool_call; unsloth@2026.11.10 (cpu) t1.template_parity/tool_call (non_thinking) |
