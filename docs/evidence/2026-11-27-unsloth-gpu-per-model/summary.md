# Kit 4b4de273e093's per-model zips (2026-11-26 Colab run)

The combined results zip didn't download, so each per-model zip is verified and kept on its own (`ingest.py`).

- **Qwen3 (calibration):** `parity-gpu-model-4b4de273e093-Qwen__Qwen3-0.6B-20261126T122230Z.zip`, 96 verdicts, commit `4b4de273e093`, Tesla T4 (driver 580.82.07), torch 2.14.1+cu130, Unsloth's CUDA route.
  - Against M4's GPU run, field by field: 90 of 96 verdicts identical apart from `declaration.model_build` (recorded since 2026-11-26).
  - 6 differ only as 8a00284 says (after M4's kit commit b73da05): non-thinking `reasoning`, not_configurable -> not_applicable, in result, reason and declaration.error. The CPU backend's re-calibration already shows them not_applicable (`docs/evidence/2026-11-26-m4-recalibration/`).
  - Unexplained differences: none.
  - Calibration green rule: green. Results {'fail': 8, 'not_applicable': 14, 'not_configurable': 38, 'pass': 36}; fails linked {'P-0005': 8}, untriaged 0; harness inconclusives 0; missed plants 0; not_supported 0.
- **Models in the run:** Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca, deepseek-ai/DeepSeek-V4.1-Flash-1017@08c8657f02291e11b6f6b63e0e16014f8e748518. No zip for the others has come back.
