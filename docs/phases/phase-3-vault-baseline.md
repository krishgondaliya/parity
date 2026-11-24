# Phase 3: Bug vault and frozen baseline

**Goal:** build the scoreboard before the loop exists: a vault of real buggy/fixed pairs (part of it hidden), an evaluation harness that reports aggregates only, and a frozen, scored hand-written suite, v0.

**Ends with:** suite v0 frozen as a git tag and scored on the hidden vault, with intervals.

## Prerequisite: isolate the hidden vault (before any pair enters it)

The guard hook only prevents accidents. Before the first pair goes into the hidden vault:

1. Move the vault to a separate Linux user, with `700` permissions, so the account Claude Code runs as can't read it at the OS level.
2. Close the routes around file permissions for Claude Code's account: no passwordless sudo, no `docker` group membership, WSL interop off. (`Bash(sudo *)` is also denied in `.claude/settings.json`.)
3. Hidden-vault evaluation runs as the vault user, through one wrapper that prints only aggregate numbers. **You** run it, not Claude Code. Every run is logged.
4. Check it from Claude Code's account: reading or listing the vault fails with "Permission denied", and none of the routes in step 2 works. Record the check in `docs/DECISIONS.md`.

Details and reasons: `docs/DECISIONS.md`, "Phase 3 prerequisite: isolate the hidden vault at the OS level".

## Milestones

### M1. Find candidate bugs
Done when there's a list of fixed, silent training-data bugs with links, in `vault/candidates.md`.

```
/start-phase 3
```
After approval:
```
Search (read-only) the issue trackers of TRL, Unsloth, Axolotl, verl, SkyRL and transformers for FIXED bugs that silently
changed what gets trained on: chat templates, loss masks, end-of-turn tokens, special tokens, truncation, packing,
gradient accumulation, resume, checkpoint conversion. Include our own fixed findings and the 2024 gradient-accumulation bug.
List each with its issue/fix link, framework, bug class, and fix date in vault/candidates.md. Don't build pairs yet.
```

### M2. Build pairs
Done when the admitted pairs are staged, each reproducing 5/5 on both sides. Dropped candidates are logged with a reason.

```
/add-vault-pair <issue-or-fix-url>
```
Suggested target: 30+ admitted pairs, about half of them hidden. Be honest that a small hidden set gives wide intervals (with ~15 hidden pairs, a Wilson interval is roughly ±20-25 points), and plan to grow the vault over time.

### M3. You split the vault
- Pick the **date cutoff**: pairs fixed after it go hidden. Record it in `docs/DECISIONS.md`.
- The **transfer framework** (chosen in Phase 2): all of its pairs go hidden.
- Move the hidden pairs to `~/parity-vault-hidden/` **by hand**, and move the open pairs to `vault/open/`.
- Since the prerequisite above, the hidden pairs go into the vault user's directory, not this account's home. The path in the line above predates it.

### M4. Evaluation harness (aggregates only)
Done when `uv run parity eval --suite <version>` reads `PARITY_HIDDEN_VAULT` and prints recall with a 95% Wilson interval, recall at equal suite size k, the false-alarm count, the fixed-mutant score and the suite size. Nothing per-pair. Tested on `vault/open/` first. On the hidden vault it runs only inside the wrapper from the prerequisite, as the vault user, and only you start it.

```
Build src/parity/eval/: score a suite version on a vault directory given by env var.
Caught = check fails on buggy AND passes on fixed. Output aggregates only (recall, Wilson 95% interval, recall@k averaged
over random k-subsets, false alarms on the calibration set, fixed-mutant score, suite size). Never print per-pair details.
Test it end to end on vault/open.
```

### M5. Fixed mutant set and calibration set
- Generate a **fixed mutant set once**, with a separate script (not the future breaker agent), and store it with a hash.
- Freeze the **calibration set** (known-good setups) that false alarms are measured on.

### M6. Freeze and score v0
```
git tag suite-v0
```
Then **you** run the hidden-vault wrapper on `suite-v0`. Claude Code works only from the aggregate numbers it prints.

## Definition of done

- [ ] Vault admitted pairs, each 5/5 on both sides; split recorded (date cutoff + transfer framework)
- [ ] The hidden split is outside the repo and owned by the vault user (`700`); Claude Code's account can't read it at the OS level, and the guard hook blocks it too (both tested)
- [ ] Hidden-vault evaluation runs only through the wrapper, started by you, and every run is logged
- [ ] The eval harness prints aggregates only (tested on the open vault)
- [ ] The fixed mutant set and calibration set are frozen with hashes
- [ ] `suite-v0` is tagged and scored, with the result in `runs/eval/history.csv`

## Pitfalls

- **Expect to drop a lot of candidates.** Deep-learning bugs often fail to reproduce on modern versions.
- **Never peek at the hidden split** "just to check one thing". If you do, move those pairs to open and pick replacements.
