---
name: wrap-session
description: End a Parity work session. Updates STATUS.md and DECISIONS.md, runs tests, and commits.
disable-model-invocation: true
---

Wrap up this session:

1. Run `uv run pytest -q`. If anything fails, say so first in the summary and don't commit broken code.
2. Update `docs/STATUS.md`:
   - The current phase and today's date.
   - What got done this session (2-5 concrete bullets).
   - The next 1-3 actions, specific enough to start tomorrow without re-reading everything.
   - Open questions.
   - The findings list (id, status).
3. If a decision was made (scope, tolerance, target list, gate), add it to `docs/DECISIONS.md`.
4. If a command in CLAUDE.md's Commands section changed or now exists, update it.
5. Commit with a clear message. Don't push unless I ask.
6. Give me a 3-line summary: done, next, blocked.
