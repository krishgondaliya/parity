# Tier 1 summary

## Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca

### Template properties

<a id="qwen-qwen3-0-6b-c1899de289a0-prefix-breaks-a55ee1b16601"></a>
**Qwen/Qwen3-0.6B@c1899de289a0#prefix-breaks@a55ee1b16601** (prefix_breaks; template `model`, sha256 `a55ee1b16601`)

Not prefix-preserving: at 5 boundary(ies) across 4 fixture(s), rendering a conversation up to an assistant turn is not the start of rendering the whole conversation. Frameworks that use this template inherit these breaks.

- `multi_turn`, boundary 2: removes `"<think>\n\n</think>\n\n"`, inserts nothing
- `multi_turn`, boundary 4: removes `"<think>\n\n</think>\n\n"`, inserts nothing
- `tool_call`, boundary 2: removes `"<think>\n\n</think>\n\n"`, inserts nothing
- `reasoning`, boundary 2: removes `"<think>\n91 is odd and its digits sum to 10, so 2 and 3 do not divide it. 91 / 7 = 13, so 91 = 7 * 13.\n</think>\n\n"`, inserts nothing
- `empty_assistant`, boundary 2: removes `"<think>\n\n</think>\n\n"`, inserts nothing

### Results

| Framework | Check | Result | Ran on | Notes |
| --- | --- | --- | --- | --- |
| axolotl@0.21.1 | t1.assistant_mask | fail | 8 of 8 |  |
| axolotl@0.21.1 | t1.end_of_turn | fail | 8 of 8 |  |
| axolotl@0.21.1 | t1.multiturn_prefix | inconclusive | 4 of 8 | prefix not preserved; the breaks are the template's (see [Qwen/Qwen3-0.6B@c1899de289a0#prefix-breaks@a55ee1b16601](#qwen-qwen3-0-6b-c1899de289a0-prefix-breaks-a55ee1b16601)); not applicable on 4 fixture(s) |
| axolotl@0.21.1 | t1.special_tokens | fail | 8 of 8 |  |
| axolotl@0.21.1 | t1.template_parity | pass | 8 of 8 |  |
| axolotl@0.21.1 | t1.truncation | fail | 8 of 8 |  |
