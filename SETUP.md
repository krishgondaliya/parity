# Setting up Parity in Claude Code (from zero)

Follow this once, top to bottom. It takes about an hour. After that, every work session follows the loop at the bottom.

## 0. What you need

| Thing | Why | Notes |
| --- | --- | --- |
| macOS, Linux, or Windows with WSL | Claude Code and the tooling | Any laptop is fine. Tiers 1 and 2 are CPU-only |
| A Claude plan (Pro, Max, Team, Enterprise) or a Claude Console account | Claude Code needs one to log in | |
| GitHub account + `gh` CLI | Your public repo; read-only searches of framework issue trackers | Posting upstream stays manual |
| Hugging Face account + read token | Downloading tokenizers and configs | Some models (e.g. Llama, Gemma) need you to accept a license on their page first |
| `uv` | Python versions, environments, lockfiles | One tool for everything Python |
| Docker | Only from Phase 7 (Tier 3) | Skip for now |

## 1. Install the tools

```bash
# Claude Code (native installer: macOS, Linux, WSL)
curl -fsSL https://claude.ai/install.sh | bash
# or on macOS: brew install --cask claude-code

# uv (Python + env manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# GitHub CLI (macOS shown; see cli.github.com for Linux)
brew install gh
gh auth login
```

Check:

```bash
claude --version
uv --version
gh --version
python3 --version   # the safety hook uses python3
```

## 2. Create the repo

```bash
mkdir parity && cd parity
unzip ~/Downloads/parity-claude-code-starter.zip -d .   # kit files land in the repo root
git init
uv init --package --python 3.12                        # adds pyproject.toml, src/parity/, README.md
```

Check that `CLAUDE.md`, `.claude/`, `docs/`, `findings/` and `vault/` now sit next to `pyproject.toml` (`ls -a`).

Then:

```bash
cp .env.example .env        # put your HF_TOKEN in .env
git add -A
git commit -m "chore: bootstrap Parity with Claude Code starter kit"
gh repo create parity --private --source=. --push   # make it public later, at Phase 1
```

Keep the repo private until Phase 1. Then make it public: the public repo is part of the portfolio.

## 3. Create the hidden vault location (now, even though you need it in Phase 3)

The hidden half of the bug vault must live outside this repo, so Claude Code and the loop agents never see it.

```bash
mkdir -p ~/parity-vault-hidden
echo 'PARITY_HIDDEN_VAULT=~/parity-vault-hidden' >> .env
```

The guard hook blocks any tool call that mentions `parity-vault-hidden`. Only the evaluation harness reads it, through the environment variable.

**You run hidden-vault evaluation, not Claude Code.** Before any pair enters the hidden vault (Phase 3), the vault moves to a separate Linux user with `700` permissions, and evaluation runs as that user through one wrapper that prints only aggregate numbers. Every run is logged, and Claude Code works only from the numbers you paste back (`/run-eval`). See the Phase 3 prerequisite in `docs/phases/phase-3-vault-baseline.md` and `docs/DECISIONS.md`.

## 4. First Claude Code session

```bash
cd parity
claude
```

1. Log in when asked, and accept the workspace trust prompt (needed for project skills and hooks to load).
2. **Don't run `/init`.** The provided `CLAUDE.md` replaces it. Later, ask Claude to update the Commands section instead.
3. Type `/` and check that the project skills show up (`/start-phase`, `/triage`, `/wrap-session`, and the rest).
4. Paste this first prompt:

```
Read CLAUDE.md, docs/STATUS.md and docs/plan/01-overview.md.
In 10 bullets, tell me what Parity is, the hard rules, and what Phase 0 must prove.
Then list anything in SETUP.md I still haven't done (check the repo and .env, without printing secrets).
```

5. Test the guard hook. Ask Claude: `run gh issue create --title test --body test`. It should be blocked with a reason. If it isn't, fix the hook before going on.

6. Then: `/start-phase 0`

## 5. The everyday loop

```
claude
  -> /start-phase <n>     (first session of a phase) or "continue from STATUS.md"
  -> plan mode: Claude proposes, you approve
  -> build in small steps, tests green, commit
  -> any failing check: /triage -> /shrink-repro -> skeptic subagent -> /draft-issue
  -> /wrap-session        (updates STATUS.md and DECISIONS.md, commits)
```

Habits that keep it working:

- **One task per conversation.** `/clear` between unrelated tasks, so old context doesn't leak in.
- **Plan mode for anything bigger than a small fix.** Approve the plan before code is written.
- **You post upstream, always by hand.** Read the draft and the repro first. One issue per bug.
- **Gates are decisions you make.** `/gate-check` gathers the evidence and writes the record, but you decide. Log the outcome in `docs/DECISIONS.md`.

## 6. Where each phase is explained

| Phase | Playbook | Ends with |
| --- | --- | --- |
| 0 Kill test | `docs/phases/phase-0-kill-test.md` | Gate A (`docs/gates/gate-A.md`) |
| 1 Walking skeleton | `docs/phases/phase-1-walking-skeleton.md` | Public repo + 1-row card |
| 2 Tier 1 breadth | `docs/phases/phase-2-tier1-breadth.md` | Full Tier 1 matrix |
| 3 Bug vault + frozen baseline | `docs/phases/phase-3-vault-baseline.md` | v0 scored on the hidden vault |
| 4 Self-improving loop | `docs/phases/phase-4-self-improving-loop.md` | Gate B (`docs/gates/gate-B.md`) |
| 5 Tier 2 + Day-0 automation | `docs/phases/phase-5-numerics-automation.md` | 48-hour report cards |
| 6 Prospective + recursion tests | `docs/phases/phase-6-prospective-recursion.md` | Gate C (`docs/gates/gate-C.md`) |
| 7 RL seam, then go further | `docs/phases/phase-7-rl-seam.md` | Tier 3 on the card |

## 7. What's in this kit

```
CLAUDE.md                     project rules + imports docs/STATUS.md every session
SETUP.md                      this file
.env.example                  token placeholders
.gitignore
.claude/settings.json         permissions + the guard hook
.claude/hooks/guard.py        blocks upstream posting and hidden-vault access
.claude/skills/*/SKILL.md     project slash commands (see list below)
.claude/agents/*.md           skeptic, issue-reviewer, eval-auditor subagents
docs/STATUS.md                living state (current phase, next actions)
docs/DECISIONS.md             decision log
docs/plan/                    the full plan, worked example, decision history
docs/phases/                  one playbook per phase
docs/gates/                   gate checklists A, B, C
findings/_TEMPLATE.md         finding record + issue draft template
vault/README.md               vault rules and record template
```

**Skills (type `/name`):** `start-phase`, `new-adapter`, `new-check`, `calibrate`, `triage`, `shrink-repro`, `draft-issue`, `add-vault-pair`, `run-eval`, `gate-check`, `wrap-session`.

**Subagents:**
- `skeptic`: tries to prove a finding is NOT a real bug.
- `issue-reviewer`: checks that a draft is minimal, polite and reproducible.
- `eval-auditor`: looks for leakage and cheating in the evaluation.
