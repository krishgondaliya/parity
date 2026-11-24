# Phase 1: Walking skeleton

**Goal:** turn the Phase 0 scripts into a clean package where one model × one framework × one check runs end to end, from the CLI to a public report card.

**Ends with:** the public repo, a CI badge, and a report card with one honest row.

## Milestones

### M1. Package and engine
Done when `src/parity/` follows the CLAUDE.md layout, the engine writes `parity.verdict/v0` JSON, and a missed self-test forces `inconclusive`.

```
/start-phase 1
```
After approval:
```
Refactor the Phase 0 code into src/parity/ (adapters, references, checks, fixtures, engine).
The engine runs check + planted fault together and writes one verdict JSON per (check, model, framework, fixture).
A missed planted fault must force result=inconclusive. Keep every Phase 0 test passing; add engine tests.
```

### M2. CLI
Done when `uv run parity run --model <org/name@rev> --framework trl --tier 1 --check t1.end_of_turn` works and writes to `runs/<run-id>/`.

### M3. Report card
Done when `uv run parity card` builds `site/index.html` from verdict files: rows are models (newest first), columns are frameworks, and each cell shows counts and links to its verdicts and lockfile. The card states its scope (tiers run).

Each model also gets a **template properties** section, shown once per model: facts about the chat templates its frameworks use, such as Qwen3's prefix breaks (`parity.references.properties`). A cell whose verdicts inherit a property links to it, so a pass that inherits prefix breaks never reads as "prefix preserved". Each cell also shows how many fixtures each check ran on, and which planted faults were real-world or synthetic. `parity.engine.report.render_summary` already does the Markdown version.

```
Build the static report card generator (plain HTML + a little CSS, no framework).
Every cell links to its verdict JSON and lockfile. Add a scope line: which tiers ran, and that Tier 2 uses tiny models.
```

### M4. CI and going public
Done when GitHub Actions runs the tests on every push, the card publishes to GitHub Pages, and the README explains Parity in 30 seconds.

```
Add a GitHub Actions workflow that runs uv sync + pytest on push.
Add a second workflow that builds the card and publishes site/ to GitHub Pages on main.
Write README.md: what Parity is, the report card link, how a check works (with the end-of-turn example), the findings ledger, how to reproduce a verdict.
```
Then flip the repo to public (`gh repo edit --visibility public`; run it yourself).

### M5. Findings ledger page
Done when the card links to a ledger page listing every finding: id, status, upstream link, fixed-in. Retracted findings stay visible.

## Definition of done

- [ ] One end-to-end path through the CLI writes verdicts and builds the card
- [ ] A missed self-test forces `inconclusive` (tested)
- [ ] CI is green, the card is live on GitHub Pages, the repo is public
- [ ] The README and the ledger page exist
- [ ] CLAUDE.md's Commands section is updated

## Pitfalls

- Don't add frameworks or checks here. That's Phase 2. The skeleton should be thin but complete.
- Keep run outputs out of git, except the verdicts the card publishes.
