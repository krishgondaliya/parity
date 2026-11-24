# How we got to Parity: decision history

As of Nov 23, 2026. A condensed record of the idea hunt that led to Parity, so future work doesn't redo it.

## The criteria we hunted with

- Problem first, then a solution with revenue, then an existence check. Drop anything already fully built.
- Technical (not regulatory, not customer service), persistent problem, high odds of success.
- Premise: "steal incumbents' alpha". Motto: "stand on shoulders of giants".
- Deep (Makora-style custom tools), approachable solo, a big learning opportunity for applied AI roles at labs, not a feature.
- Success must be objectively checkable (no judgment calls that create false alarms).
- YC Fall 2026 principle: pick a business where operating it generates data nobody else can get.

## Ideas killed, and why

| Idea | Why it died |
| --- | --- |
| Verified GPU kernel engine | Makora already does a superset |
| Oracle migration, PLC, game porting, game remaster | Destination platforms give migration away, or weak fit |
| Foreman (agent supervisor) | False alarms too hard to control |
| AX tools, LLM cost optimizer, OpenClaw skill security, agent provenance, behavioral diff / traffic replay | Already shipped by funded startups or open specs |
| Time-travel debugger for agents (Python/TS) | [pyflight](https://pypi.org/project/pyflight/) (MIT, July 2026) already does Python with agent tools; [Replay MCP](https://docs.replay.io/basics/replay-devtools/mcp) covers web; [Undo](https://docs.undo.io/java/UndoAI.html) covers C/C++/Java |
| Mutation testing for agent-written tests | Commoditized: dedicated products, many free agent skills, ThoughtWorks "Adopt" |
| Silent-bug preflight for training data | TrainSight, dropoutt, trainproof are free; TrainCheck (OSDI '25) and Amazon's [Plumbline](https://arxiv.org/abs/2611.13393) cover the deep version |
| Job-description infra round: tool caching, sandbox scale, sandbox broker, rollout service, zero-advantage task selection, RL observability | Owned by research and big players: [TVCache](https://arxiv.org/abs/2602.10986), Morph, GKE Agent Sandbox, Modal, ComputeSDK, NVIDIA [ProRL Agent](https://arxiv.org/html/2603.18815), GRESO / Pilot-Commit in verl, [NeMo RL observability](https://docs.nvidia.com/nemo/rl/latest/observability/index.html), [verl RL-Insight](https://verl.readthedocs.io/en/latest/advance/rl_insight.html) |

**The pattern:** in Nov 2026, single "checker tools" for agent code or RL infra get built within weeks, often by solo devs under MIT. A tool alone has no moat.

## Saved for later

- **Goodhart** (reward-hack auditor for RL environments): on hold. Doc: https://claude.ai/artifact/6G1ir3887pn6fxEEnEQQaM. Its "Goodhart-proofing" ideas now live inside Parity's loop design.

## Why Parity was picked (career first)

Startup potential and portfolio potential pulled apart, so the goal became a public, career-first project aimed at post-training / RL infra roles.

| Idea | Portfolio ceiling | Startup potential | Cost | Odds of a real result |
|---|---|---|---|---|
| Parity (Day-0 correctness lab) | Very high | Medium (contested) | ~$0, CPU-first | Medium |
| Goodhart | High | Medium | Low | Medium-high |
| Crashproof Rollouts | Medium-high | Low | ~$100-200 | High |

- **Externally verifiable:** bugs that maintainers confirm and fix can't be discounted as toy results.
- **Proven path:** Unsloth built its reputation (and a YC spot) on fixing bugs in major model releases and training code.
- **Objective:** deterministic, reference-comparison checks with planted-fault self-tests.
- **Steal incumbents' alpha:** Amazon runs this internally (Plumbline); Parity does it in public for the long tail of frameworks.

**Known weaknesses (from the pressure test):** platforms absorbing template/mask fixes, Plumbline going open source, a narrow paying market, and weak fit for "agent-written code". Accepted because the goal is career-first.

## Fallback: Crashproof Rollouts

If Parity's kill test finds zero confirmed bugs, switch to Crashproof Rollouts:

- Break a small agentic RL run on purpose (kill sandboxes, hangs, OOM, restarts) with a seeded chaos tool.
- Measure the damage: GPU time wasted, rollouts lost, infra errors leaking into observations or rewards, time to target reward.
- Build the fixes: infra-vs-model failure classification, turn-level journaling and resume, retry/timeout policies, exact trainer resume.
- Publish before/after numbers at 0%, 5% and 15% failure rates. One coding-model team saw ~16% of early trajectories hit sandbox-caused failures ([KAT-Coder-V2.5](https://arxiv.org/pdf/2607.05471)).
- Setup: Qwen3 0.6B/1.7B, GRPO, multi-turn text-to-SQL in Docker, SkyRL (verl as fallback). Related work: [Belayer](https://arxiv.org/pdf/2608.14635), [SPROUT](https://arxiv.org/pdf/2610.14405).

## Added on Nov 23, 2026

- **Self-improving check suite:** writer and breaker agents, three executed gates, hidden bug vault.
- **Two-question evaluation as a core deliverable:** (1) is the suite getting better? (2) is the loop the reason? Both are now built into the roadmap (Phase 3 vault and baseline before Phase 4 loop; Gates B and C).
