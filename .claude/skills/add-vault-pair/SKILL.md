---
name: add-vault-pair
description: Turn a fixed silent training bug into a reproducible buggy/fixed pair for the Parity bug vault (Phase 3+).
argument-hint: "[issue or fix URL]"
---

Build a vault pair from: $ARGUMENTS

Follow `vault/README.md` and `docs/plan/02-detailed-blueprint.md` (section "Bug vault format").

1. Find the buggy version and the fixed version (release tags or commits), and pin each in its own lockfile.
2. Write a repro that fails on buggy and passes on fixed: CPU-only, no full weights.
3. Run both sides 5 times. Admit the pair only if buggy fails 5/5 **and** fixed passes 5/5. Otherwise log it as dropped, with the reason.
4. Save it to `vault/staging/<V-id>/`: `record.md` (all vault fields filled), the repro, and both lockfiles.
5. **Don't decide the split.** Tell me the pair is staged, with its fix date and framework. I decide open vs hidden:
   - Open: I'll ask you to move it to `vault/open/`.
   - Hidden: I move it by hand to the hidden location outside this repo, and you never see it again.
