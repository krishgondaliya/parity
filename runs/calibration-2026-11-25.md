# Calibration, 2026-11-25 (M4)

- **Model:** `Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca`, all 8 fixtures, both serving modes, every planted fault.
- **Checks:** as of commit `8a00284f5889`, including `t1.special_tokens`' split padding rules and the serving-mode applicability rule (a fixture whose assistant turn carries reasoning is not applicable in non-thinking mode).
- **Green means:** no untriaged fail, no harness-caused inconclusive, every planted fault caught (`parity.engine.calibration`). Fails linked to triaged findings or known differences are allowed, and listed.
- **Overall: GREEN.**

## trl@1.18.2: green

- Results: 7 fail, 14 not_applicable, 75 pass
- Linked fails, KD-0001: 7 (non_thinking t1.template_parity/empty_assistant; non_thinking t1.template_parity/multi_turn; non_thinking t1.template_parity/tool_call; thinking t1.template_parity/empty_assistant; thinking t1.template_parity/multi_turn; thinking t1.template_parity/reasoning; thinking t1.template_parity/tool_call)
- Untriaged fails: 0
- Inconclusive, the framework's (a linked template parity fail explains it): 0
- **Inconclusive, harness-caused**: 0
- **Planted faults missed**: 0

## axolotl@0.21.1: green

- Results: 21 fail, 14 not_applicable, 61 pass
- Linked fails, P-0001: 13 (thinking t1.assistant_mask/long_answer; thinking t1.assistant_mask/multi_turn; thinking t1.assistant_mask/non_ascii; thinking t1.assistant_mask/reasoning; thinking t1.assistant_mask/single_turn; thinking t1.assistant_mask/system_prompt; thinking t1.assistant_mask/tool_call; thinking t1.truncation/multi_turn; thinking t1.truncation/non_ascii; thinking t1.truncation/reasoning; thinking t1.truncation/single_turn; thinking t1.truncation/system_prompt; thinking t1.truncation/tool_call)
- Linked fails, P-0002: 8 (non_thinking t1.assistant_mask/empty_assistant; non_thinking t1.end_of_turn/empty_assistant; non_thinking t1.special_tokens/empty_assistant; non_thinking t1.truncation/empty_assistant; thinking t1.assistant_mask/empty_assistant; thinking t1.end_of_turn/empty_assistant; thinking t1.special_tokens/empty_assistant; thinking t1.truncation/empty_assistant)
- Untriaged fails: 0
- Inconclusive, the framework's (a linked template parity fail explains it): 0
- **Inconclusive, harness-caused**: 0
- **Planted faults missed**: 0

## unsloth@2026.11.10 (gpu): green

GPU run `parity-gpu-results-b73da059b614-20261125T221035Z.zip` (commit `b73da059b614`, Tesla T4), verified (`unsloth/provenance.json`).

**Unsloth's calibration run is its GPU route** (Colab kit, RUN = "full"; verified in `provenance.json`). Non-thinking mode is not configurable for Unsloth (P-0006, the human's decision). The kit was built at commit `b73da05`, before the serving-mode applicability rule (2026-11-25), and that rule is the only code change since. So the six `reasoning` × non-thinking cells record `not_configurable`, where today's runner would record `not_applicable`. Neither is judged, and neither counts toward green. Every judged cell is unaffected.

- Results: 8 fail, 8 not_applicable, 44 not_configurable, 36 pass
- Linked fails, P-0005: 8 (thinking t1.special_tokens/empty_assistant; thinking t1.special_tokens/long_answer; thinking t1.special_tokens/multi_turn; thinking t1.special_tokens/non_ascii; thinking t1.special_tokens/reasoning; thinking t1.special_tokens/single_turn; thinking t1.special_tokens/system_prompt; thinking t1.special_tokens/tool_call)
- Untriaged fails: 0
- Inconclusive, the framework's (a linked template parity fail explains it): 0
- **Inconclusive, harness-caused**: 0
- **Planted faults missed**: 0

## Framework x mode x check

| Framework | Mode | Check | Clean result (over 8 fixtures) | Planted faults | Notes |
| --- | --- | --- | --- | --- | --- |
| trl@1.18.2 | non_thinking | t1.assistant_mask | 1 n/a, 7 pass | 45 of 45 caught |  |
| trl@1.18.2 | non_thinking | t1.end_of_turn | 1 n/a, 7 pass | 14 of 14 caught |  |
| trl@1.18.2 | non_thinking | t1.multiturn_prefix | 5 n/a, 3 pass | 5 of 5 caught |  |
| trl@1.18.2 | non_thinking | t1.special_tokens | 1 n/a, 7 pass | 28 of 28 caught |  |
| trl@1.18.2 | non_thinking | t1.template_parity | 3 difference, 1 n/a, 4 pass | 21 of 21 caught | root cause KD-0001 |
| trl@1.18.2 | non_thinking | t1.truncation | 1 n/a, 7 pass | 9 of 9 caught |  |
| trl@1.18.2 | thinking | t1.assistant_mask | 8 pass | 52 of 52 caught |  |
| trl@1.18.2 | thinking | t1.end_of_turn | 8 pass | 16 of 16 caught |  |
| trl@1.18.2 | thinking | t1.multiturn_prefix | 4 n/a, 4 pass | 6 of 6 caught |  |
| trl@1.18.2 | thinking | t1.special_tokens | 8 pass | 32 of 32 caught |  |
| trl@1.18.2 | thinking | t1.template_parity | 4 difference, 4 pass | 24 of 24 caught | root cause KD-0001 |
| trl@1.18.2 | thinking | t1.truncation | 8 pass | 10 of 10 caught |  |
| axolotl@0.21.1 | non_thinking | t1.assistant_mask | 1 fail, 1 n/a, 6 pass | 45 of 45 caught | root cause P-0002 |
| axolotl@0.21.1 | non_thinking | t1.end_of_turn | 1 fail, 1 n/a, 6 pass | 14 of 14 caught | root cause P-0002 |
| axolotl@0.21.1 | non_thinking | t1.multiturn_prefix | 5 n/a, 3 pass | 4 of 4 caught |  |
| axolotl@0.21.1 | non_thinking | t1.special_tokens | 1 fail, 1 n/a, 6 pass | 28 of 28 caught | root cause P-0002 |
| axolotl@0.21.1 | non_thinking | t1.template_parity | 1 n/a, 7 pass | 21 of 21 caught |  |
| axolotl@0.21.1 | non_thinking | t1.truncation | 1 fail, 1 n/a, 6 pass | 7 of 7 caught | root cause P-0002 |
| axolotl@0.21.1 | thinking | t1.assistant_mask | 7 difference, 1 fail | 52 of 52 caught | root cause P-0001; root cause P-0002 |
| axolotl@0.21.1 | thinking | t1.end_of_turn | 1 fail, 7 pass | 16 of 16 caught | root cause P-0002 |
| axolotl@0.21.1 | thinking | t1.multiturn_prefix | 4 n/a, 4 pass | 5 of 5 caught |  |
| axolotl@0.21.1 | thinking | t1.special_tokens | 1 fail, 7 pass | 32 of 32 caught | root cause P-0002 |
| axolotl@0.21.1 | thinking | t1.template_parity | 8 pass | 24 of 24 caught |  |
| axolotl@0.21.1 | thinking | t1.truncation | 6 difference, 1 fail, 1 pass | 8 of 8 caught | root cause P-0001; root cause P-0002 |
| unsloth@2026.11.10 (gpu) | non_thinking | t1.assistant_mask | 8 not_configurable | none run |  |
| unsloth@2026.11.10 (gpu) | non_thinking | t1.end_of_turn | 8 not_configurable | none run |  |
| unsloth@2026.11.10 (gpu) | non_thinking | t1.multiturn_prefix | 4 n/a, 4 not_configurable | none run |  |
| unsloth@2026.11.10 (gpu) | non_thinking | t1.special_tokens | 8 not_configurable | none run |  |
| unsloth@2026.11.10 (gpu) | non_thinking | t1.template_parity | 8 not_configurable | none run |  |
| unsloth@2026.11.10 (gpu) | non_thinking | t1.truncation | 8 not_configurable | none run |  |
| unsloth@2026.11.10 (gpu) | thinking | t1.assistant_mask | 8 pass | 52 of 52 caught |  |
| unsloth@2026.11.10 (gpu) | thinking | t1.end_of_turn | 8 pass | 16 of 16 caught |  |
| unsloth@2026.11.10 (gpu) | thinking | t1.multiturn_prefix | 4 n/a, 4 pass | 6 of 6 caught |  |
| unsloth@2026.11.10 (gpu) | thinking | t1.special_tokens | 8 fail | 32 of 32 caught | root cause P-0005 |
| unsloth@2026.11.10 (gpu) | thinking | t1.template_parity | 8 pass | 24 of 24 caught |  |
| unsloth@2026.11.10 (gpu) | thinking | t1.truncation | 8 pass | 10 of 10 caught |  |
