# Vendored from TRL

`qwen3_training.jinja` is copied byte for byte from TRL. It is used only in tests, as a second, independent reference: its `{% generation %}` markers give an assistant mask that the prefix-diff method in `src/parity/references/template.py` must reproduce. No adapter uses it.

- Source: https://github.com/huggingface/trl/blob/v1.18.2/trl/chat_templates/qwen3_training.jinja
- TRL version: 1.18.2 (latest on PyPI on 2026-11-23). Tag object `039e3aa9f0f19ae98d35048c33cc76cf24e6e833`, commit `c7caa02b6e162034d5736939cab49ebda4b4a080`
- SHA-256: `1907918feb7afcf18fac8db1a419985b4773ac434e353cfae91b1949dcbf592b`. `tests/stubs.py` checks this, so a line-ending change fails loudly.
- License: Apache-2.0, copyright the Hugging Face team (TRL).
