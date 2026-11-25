# Parity

Parity is a public Day-0 correctness lab for post-training stacks. For each open-weight model release, it checks what fine-tuning frameworks (TRL, Unsloth, Axolotl, verl, SkyRL) actually feed into training against an independent reference, and publishes a report card. Later, a gated self-improving loop grows the check suite, and a hidden bug vault proves whether it really improved.

This is a solo, career-first portfolio project. Correctness and honesty matter more than speed.

## Current state

@docs/STATUS.md

## Plan (read when you need detail, not every session)

- Overview: `docs/plan/01-overview.md`
- Detailed blueprint (interfaces, schemas, pass rules, loop, vault, evaluation): `docs/plan/02-detailed-blueprint.md`
- Phase playbooks: `docs/phases/` (one file per phase)
- Gate checklists: `docs/gates/`
- Decision log: `docs/DECISIONS.md`

## Hard rules (never break these)

1. **Never post upstream.** Do not create, comment on, or edit issues or PRs in any other repo. Draft issues go to `findings/<id>/issue.md` and the human posts them. A hook enforces this.
2. **Adapters call the framework's own public code path.** Never re-implement a framework's tokenization, masking, collation, or training logic inside an adapter. If you re-implement it, you are testing our code, not theirs.
3. **Every check ships with its planted fault.** A verdict counts only if the check caught its own planted fault in the same run. Otherwise the result is `inconclusive`, never `pass`.
4. **Assume our code is wrong first.** For any failing check, rule out the adapter, the fixture, and the reference before calling it a framework or model bug.
5. **Exact beats tolerance.** Use bit-exact comparison wherever possible. Numeric checks use the noise-floor rule (fail above 4x the measured floor). Never loosen a tolerance to make something pass; any tolerance change goes in `docs/DECISIONS.md` with a reason.
6. **No full model weights in Tiers 1 and 2.** Download only tokenizer, chat template and config files. Use tiny random-weight models built with `from_config` for numerics.
7. **Pin everything.** Every verdict records the exact model revision, framework version and lockfile.
8. **Language discipline.** In anything public, say "candidate finding" until a maintainer confirms. Never say "bug" in a draft title unless it is confirmed.
9. **Hidden vault is off limits** (Phase 3 onward). Never read, list, grep or reference the hidden vault path. Only the evaluation harness reads it, through an environment variable.
10. **Secrets stay local.** Tokens live in `.env` (gitignored). Never print or commit them.

## How to work in this repo

- Start a phase with `/start-phase <n>`. Work in plan mode first, get the plan approved, then build.
- One task per conversation. Use `/clear` between unrelated tasks.
- Small commits with clear messages. Run the tests before every commit.
- End every session with `/wrap-session` so `docs/STATUS.md` stays true.
- When a fail shows up: `/triage` first, then `/shrink-repro`, then `/draft-issue`. Use the `skeptic` subagent before any finding leaves the repo.
- Gates are decided by evidence files, not by feel: `/gate-check <A|B|C>`.
- GitHub lookups are read-only. Use `gh search ...`, or `gh api "search/issues?q=..."` with the query in the URL. Never use `gh api -f` / `-F` / `--field`: with fields, `gh api` sends a POST, and the guard hook rightly blocks it (rule 1).

## Repo layout (target, grows by phase)

```
src/parity/
  adapters/     one module per framework (public code path only)
  references/   independent answers: templates, loss, fp64 AdamW, unpacked forward
  checks/       tier1/, tier2/, tier3/  (check + planted fault + tests)
  fixtures/     the 8 test conversations
  tiny/         tiny random-weight model factory
  engine/       runner, pass rules, verdict writer
  shrink/       repro minimizer
  watcher/      release polling (Phase 5)
  loop/         writer, breaker, gates (Phase 4)
  eval/         scoring, baselines, Wilson intervals (Phase 3)
envs/           one uv lockfile per framework@version
runs/           verdict JSON per run (gitignored except published ones)
findings/       one folder per finding: record.md, repro.py, issue.md
vault/open/     open buggy/fixed pairs (hidden split lives OUTSIDE this repo)
site/           static report card
docs/           plan, phases, gates, status, decisions
```

## Commands

Fill these in as they come to exist. Keep this list accurate.

- Install / sync: `uv sync`
- Tests: `uv run pytest -q` (framework adapter tests skip here)
- TRL environment: `cd envs/trl-1.18.2 && uv sync`, then from the repo root `envs/trl-1.18.2/.venv/bin/python -m pytest -q` (all tests, including `tests/test_adapter_trl.py`)
- Six checks on the real TRL adapter (M3 evidence): `envs/trl-1.18.2/.venv/bin/python docs/evidence/2026-11-24-m3-trl/run.py`
- Golden review by eye: `uv run python docs/evidence/golden_review.py`
- Axolotl environment: `cd envs/axolotl-0.21.1 && uv sync`, then from the repo root `envs/axolotl-0.21.1/.venv/bin/python -m pytest -q` (includes `tests/test_adapter_axolotl.py`, a few minutes)
- Six checks on the real Axolotl adapter (M3 evidence, ~12 min): `envs/axolotl-0.21.1/.venv/bin/python docs/evidence/2026-11-24-m3-axolotl/run.py`
- Unsloth environment: `cd envs/unsloth-2026.11.10 && uv sync`, then from the repo root `envs/unsloth-2026.11.10/.venv/bin/python -m pytest -q` (includes `tests/test_adapter_unsloth.py`; CPU fallback, see `environment.json`)
- Six checks on the real Unsloth adapter, both modes (M3 evidence, CPU backend): `envs/unsloth-2026.11.10/.venv/bin/python docs/evidence/2026-11-24-m3-unsloth/run.py` (`--relink` re-links saved verdicts)
- Unsloth GPU run kit (Colab): `uv run python docs/gpu-run/build_kit.py` (clean tree; writes `dist/gpu-kit/<commit>/`), then follow `docs/gpu-run/GPU-RUN.md`. In the notebook, `RUN = "p0005"` skips the checks and runs only P-0005 (a few minutes).
- Ingest the GPU results zip (verify, keep, render GPU primary + CPU secondary): `uv run python docs/evidence/2026-11-25-m3-unsloth-gpu/ingest.py dist/gpu-results/<zip>` (`--relink` re-renders saved verdicts). A P-0005-only zip: `uv run python docs/evidence/2026-11-25-p0005-gpu/ingest.py dist/gpu-results/<zip>`. Both verify through `parity.gpu_results`.
- P-0005 repro, standalone (no Parity): `uv venv --python 3.12 && uv pip install -r findings/P-0005/requirements.txt`, then run `findings/P-0005/repro.py` from outside the repo (exits 1)
- Test the GPU kit's checks step without a GPU: `envs/unsloth-2026.11.10/.venv/bin/python docs/gpu-run/run_checks.py --models <org/name@rev> --out <dir> --env-lock envs/unsloth-2026.11.10/uv.lock --allow-cpu`
- Six checks in both serving modes, with root-cause linking (current evidence): `envs/trl-1.18.2/.venv/bin/python docs/evidence/2026-11-24-m3-rerun/run.py trl` (and `envs/axolotl-0.21.1/.venv/bin/python ... axolotl`, ~15 min). Add `--relink` to re-link saved verdicts to `findings/root_causes.json` without re-running the framework.
- Structural-method agreement check: `uv run python docs/evidence/2026-11-24-structural-added-tokens/validate.py`
- Run checks (planned): `uv run parity run --model <org/name@rev> --framework <name> --tier 1`
- Build report card (planned): `uv run parity card`

## Style

- Python 3.12, type hints, dataclasses for records.
- Verdicts and findings are JSON or Markdown files, never only console output.
- Prefer boring, readable code over clever code. This repo will be read by hiring managers.
