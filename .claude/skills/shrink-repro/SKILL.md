---
name: shrink-repro
description: Shrink a confirmed Parity candidate finding to the smallest standalone script that still shows the problem.
argument-hint: "[finding-id, e.g. P-0001]"
---

Shrink finding $0 to a standalone repro at `findings/$0/repro.py`.

Requirements:

- Runs with only the framework, transformers and the standard library, on CPU, in seconds.
- No full model weights: tokenizer and config only, or a tiny random-weight model.
- No Parity imports. A maintainer must be able to paste it into their own environment.
- Aim for 30 lines or fewer. It prints expected vs actual and exits non-zero on failure.
- Exact versions in a comment at the top. Copy the lockfile to `findings/$0/`.

Verify:

1. Run it 3 times in a fresh env with the pinned versions. It must fail every time.
2. If a fixed version exists (newer release or main branch), run it there and record whether it passes.
3. Update `findings/$0/record.md` with the run results.
