# Decision log

One entry per decision that changes scope, a pass rule, a tolerance, a target list, or a gate outcome. Newest first.

Format:

```
## YYYY-MM-DD: <short title>
- Decision:
- Why:
- Evidence: <file paths, run ids, links>
- Reversible? yes/no
```

## 2026-11-23: Phase 0 targets and calibration model pinned (M0)
- Decision: run the kill test on these 3 targets plus 1 calibration model, each pinned to an exact Hugging Face commit:

  | Role | Model | Revision (commit SHA) | Released | Reasoning | Tool calls | `{% generation %}` markers | License |
  | --- | --- | --- | --- | --- | --- | --- | --- |
  | Target | `thinkingmachines/Inkling-2-Mini` | `7a67ae2314b44ef05c87fd1d4d311c58aaeb18f6` | 2026-11-20 | yes | yes | yes (ships its own) | Apache-2.0 |
  | Target | `tencent/Hy4-flash` | `eccd3123eec73f2040f4c9ed0de1883593470510` | 2026-11-16 | yes | yes | no | Tencent custom (`tencent-hy-license`) |
  | Target | `deepseek-ai/DeepSeek-V4.1-Flash-1017` | `08c8657f02291e11b6f6b63e0e16014f8e748518` | 2026-10-16 | yes | yes | no | MIT |
  | Calibration | `Qwen/Qwen3-0.6B` | `c1899de289a04d12100db370d81485cdf75e47ca` | 2025-04-26 | yes | yes | no (TRL ships `qwen3_training.jinja`) | Apache-2.0 |

- Why:
  - All three targets are major-lab releases from the last ~5 weeks. All are ungated, and all load with a standard tokenizer class, so no repo code runs.
  - All have both reasoning blocks and tool calls, so they exercise all 8 fixtures. All appear to strip `<think>` from earlier turns, which exercises the multi-turn prefix check.
  - They cover two mask cases. Inkling ships `{% generation %}` markers, so we test whether those markers are right. Hy4 and DeepSeek ship none, so each framework has to find the assistant spans on its own.
  - DeepSeek V4.1 Flash is the most widely fine-tuned candidate.
  - Qwen3 is the known-good calibration family, because TRL ships a training template for it (TRL #5471).
  - Candidates I rejected:
    - `mistralai/Mistral-Large-4-1T-A52B`: TRL already ships a training template for it, so TRL bugs there are less likely.
    - `microsoft/Aion-1.0-Instruct-2.3B`: no reasoning blocks.
    - `moonshotai/Kimi-K3.1-Pro`: its tokenizer is custom repo code (`tokenization_kimi.py` + tiktoken).
    - Muse Spark Open and GLM-5.3: released before the ~2-month window.
- Evidence:
  - Hugging Face Hub API, queried 2026-11-23. Each SHA resolves, and each was the repo's latest commit when pinned.
  - At each pinned revision, `tokenizer.json` and `tokenizer_config.json` are present, and the targets also have `chat_template.jinja`.
  - Template features were read from the template text at the pinned SHA. None contain CR line endings.
  - The TRL template list was read from `trl/chat_templates/` on main.
- Notes to carry forward:
  - Inkling and Hy4 were both updated on the pin date (2026-11-23). If they fix their templates later, diff the new version against these pins.
  - Hy4 uses the `TokenizersBackend` tokenizer class, which needs transformers v5. If a framework env pins transformers v4, it may not load the tokenizer. M1 will show this.
  - The Hy4 license hasn't been reviewed yet. The human should read it before any Hy4 results are published.
- Reversible? yes (re-pin and re-run; earlier verdicts keep their recorded revisions)

## 2026-11-23: Parity chosen as the career-first project
- Decision: build Parity (Day-0 correctness lab) with a self-improving check suite and a two-question evaluation as core deliverables.
- Why: highest portfolio ceiling, objectively checkable, CPU-first. See `docs/plan/04-decision-history.md`.
- Evidence: plan doc https://claude.ai/artifact/LuCqsapFCxPhWiaD5RJzcu
- Reversible? yes (fallback: Crashproof Rollouts if Gate A fails)
