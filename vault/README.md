# Bug vault

The vault is Parity's scoreboard: real silent training bugs, each stored as a buggy/fixed version pair that reproduces every time. It's how we prove the check suite is actually getting better.

## Layout

```
vault/staging/<V-id>/   new pairs waiting for the human to pick a split
vault/open/<V-id>/      open split: agents and Claude Code may read these
~/parity-vault-hidden/  hidden split: OUTSIDE this repo, never read by Claude Code or loop agents
```

## Rules

1. **Reproduce or drop.** Admit a pair only if the buggy side fails 5/5 and the fixed side passes 5/5, on CPU, without full weights.
2. **The human picks the split.** Claude Code stages pairs; you move hidden pairs by hand.
3. **Hidden by date:** every pair whose fix is dated after the cutoff in `docs/DECISIONS.md` goes hidden.
4. **Hidden by framework:** one whole framework (chosen in Phase 3, recorded in `docs/DECISIONS.md`) goes hidden for the transfer test. The loop never touches that framework.
5. **Aggregates only.** The evaluation harness reports recall, intervals and counts, never per-pair details.

## Record template (`record.md` in each pair folder)

```markdown
# V-000N

- Source: <issue or fix URL>
- Bug class: <e.g. end-of-turn mask, packing leak, resume step count>
- Framework: <name>
- Buggy version: <tag/commit>  (lockfile: buggy.lock)
- Fixed version: <tag/commit>  (lockfile: fixed.lock)
- Trigger: <model, template, config, fixture that expose it>
- Repro: repro.py (fails on buggy, passes on fixed)
- Reproduced: buggy fails x/5, fixed passes x/5
- Date fixed: YYYY-MM-DD
- Split: staging | open | hidden
```
