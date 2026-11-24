---
name: draft-issue
description: Draft (never post) an upstream issue for a Parity finding.
argument-hint: "[finding-id]"
disable-model-invocation: true
---

Draft the upstream issue for finding $0. **Do not post it anywhere.** The human posts it by hand.

1. Read `findings/$0/record.md` and `findings/$0/repro.py`. If the skeptic's verdict isn't "real candidate", stop.
2. Search for duplicates one more time (read-only `gh search issues`). If one exists, draft a short comment for that issue instead.
3. Write `findings/$0/issue.md` using the issue template in `findings/_TEMPLATE.md`:
   - Title: `[<area>] <one-line symptom> for <model> (<framework> <version>)`.
   - Polite, short, factual. No blame, and no claims beyond the evidence.
   - The repro inline, expected vs actual, the reference used, the environment.
4. Ask the `issue-reviewer` subagent to review the draft, and apply its fixes.
5. Tell me the draft is ready, which repo it targets, and anything to double-check before I post it.
