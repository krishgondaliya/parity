# Detailed blueprint

As of Nov 23, 2026 · Live doc: https://claude.ai/artifact/LuCqsapFCxPhWiaD5RJzcu#b103711c-4083 (tab: Detailed blueprint)

The build details behind the overview: interfaces, schemas, pass rules, the Phase 0 runbook, and the self-improving loop with its vault and evaluation protocol.

## Repo layout and stack

One Python repo, CPU-first. Docker only shows up in Tier 3.

```
parity/
  watcher/      release polling: Hugging Face Hub orgs, GitHub releases
  envs/         one uv lockfile per framework@version
  fixtures/     the 8 test conversations (below)
  adapters/     trl.py, unsloth.py, axolotl.py, verl.py, skyrl.py
  references/   template.py, loss.py, adamw_fp64.py, unpacked.py
  tiny/         builds a random-weight model from a release's config
  checks/       tier1/, tier2/, tier3/  (each check ships with its planted fault)
  shrink/       repro minimizer
  findings/     the ledger: one folder per finding (record + repro)
  agents/       writer/, breaker/ (prompts, tools, archived versions)
  gates/        the three executed gates, no LLM inside
  vault/        open/ only; the hidden split lives in a separate private repo
  eval/         scoring, baselines, Wilson intervals, the improvement chart
  site/         static report card generated from verdict JSON
```

**Stack:** Python, CPU builds of PyTorch, transformers, uv, a pytest-style runner, GitHub Actions for scheduled runs, plain HTML generated from JSON for the card.

**The 8 fixtures** (each check runs on all of them):

1. Single turn
2. Multi-turn, 3+ turns
3. With a system prompt
4. Tool call plus tool result
5. Reasoning (think) block in the assistant turn
6. Empty assistant turn
7. Answer long enough to hit max length
8. Non-ASCII text (accents, CJK, emoji)

## Adapter interface

An adapter calls the framework's own public data path and training loop and reports exactly what came out. It never re-implements them. Otherwise you'd be testing your own code.

```python
@dataclass
class TrainCfg:
    objective: str          # "assistant_only" | "completion_only" | "all" (the framework's own setting)
    max_len: int
    packing: bool
    grad_accum: int
    pad_to: int | None      # encode_batch only: pad every row to exactly this length
    serving_mode: str | None  # "thinking" | "non_thinking"; picks the reference only, never passed to the framework

@dataclass
class Encoded:
    input_ids: list[int]
    labels: list[int]               # -100 where no loss
    attention_mask: list[int]
    position_ids: list[int] | None  # set when packing

class Adapter(Protocol):
    name: str       # "trl", "unsloth", ...
    version: str    # pinned in envs/

    def encode(self, conv, cfg: TrainCfg) -> Encoded: ...
    def train_steps(self, model_dir, batches, cfg: TrainCfg, n: int) -> StepLog: ...  # loss, grads, params per step
    def save(self, path) -> None: ...
    def resume(self, path) -> None: ...
    def export(self, path, fmt: str) -> str: ...
```

- **Tier 1** uses only `encode` with the release's real tokenizer. If a framework insists on a model object, pass the tiny one.
- **Tier 2** uses `train_steps`, `save`, `resume` and `export` on the tiny model.
- **Map each framework's own option names onto `objective`** (for TRL: `assistant_only_loss`, `completion_only_loss`). The adapter file documents that mapping, so a "difference" verdict can be traced to a setting.

## Check contract and verdict schema

Every check has two methods: one runs the comparison, one returns a copy of the adapter with that check's own bug planted. A verdict counts only if the planted copy was caught.

```python
class Check(Protocol):
    id: str     # "t1.end_of_turn"
    tier: int
    def run(self, adapter, release, fixture) -> Verdict: ...
    def plant(self, adapter) -> Adapter: ...   # same adapter, this check's bug injected
```

One verdict file per (check, model, framework, fixture):

```json
{
  "schema": "parity.verdict/v0",
  "check": "t1.end_of_turn",
  "model": "<org>/<model>@<revision>",
  "framework": "trl@<version>",
  "fixture": "multi_turn",
  "result": "fail",
  "kind": "bug",
  "self_test": "caught",
  "evidence": {"first_divergence": 412, "expected": "...", "got": "..."},
  "env_lock": "envs/trl-<version>.lock",
  "repro": "findings/<id>/repro.py"
}
```

| Field | Values | Rule |
| --- | --- | --- |
| `result` | pass, fail, inconclusive | Can't finish (unsupported architecture, crash) means inconclusive, never pass |
| `kind` | bug, difference | Difference = diverges from the official template but matches the framework's documented setting |
| `self_test` | caught, missed | Missed turns the result into inconclusive, whatever the comparison said |

## Pass rules and tolerances

There are three kinds of pass rule, borrowed from Plumbline ([paper](https://arxiv.org/pdf/2611.13393)). Where a check can be exact, it is.

| Rule | Used by | How it decides |
| --- | --- | --- |
| Exact | All of Tier 1, save and resume, token in/out, serving parity | Token ids or tensors are equal bit for bit. Report the first divergence index plus 8 decoded tokens on each side |
| Rounding bound | Loss parity, optimizer step | Loss recomputed in fp32 and updates in fp64; the gap must stay within rounding |
| Noise floor | Gradient accumulation, packing, export round trip, logprob parity | Relative L2 error per layer group against the fp32 CPU reference, judged by the formula below |

```latex
\text{fail} \iff \frac{\lVert g_{\text{test}} - g_{\text{ref}} \rVert_2}{\lVert g_{\text{ref}} \rVert_2} > 4 \cdot \max(\sigma_{\text{run}},\ \delta_{\text{bf16}})
```

σ_run is the run-to-run variation of the unchanged recipe. δ_bf16 is the bf16-vs-fp32 precision gap.

- **The margin is thin.** In Plumbline, unchanged recipes stayed at or below 3.44x the floor and known defects reached at least 4.35x. So measure the floors yourself for each framework and architecture, and publish the ratio next to the verdict.
- **The optimizer check also reports stuck weights:** the share of parameters with a nonzero gradient whose value didn't change after a step. That's the bf16 in-place update failure Plumbline found.

## Planted faults

Each check ships with the bug it exists to catch. Before any real verdict counts, the check has to flag the planted version. Where possible, the planted fault is a real bug someone already hit.

| Check | Planted fault |
| --- | --- |
| Template parity | Drop one newline or swap one special token in the rendered string |
| Assistant mask | Unmask the user turn; separately, zero the whole mask |
| End of turn | Mask the closing token (the GLM-5.3-Flash bug Plumbline found) |
| Special tokens | Prepend a second BOS; set pad to EOS and mask all pads |
| Truncation | Truncate from the right, through the answer |
| Multi-turn prefix | Re-render history with `false` turned into `False` (Prime Intellect's example) |
| Loss parity | Divide by sequence count instead of token count |
| Gradient accumulation | Average the loss per micro-batch (the 2024 Transformers bug) |
| Packing isolation | Skip the position-id reset and use a plain causal mask |
| Optimizer step | Skip Adam's bias correction |
| Save and resume | Don't restore the step counter (Plumbline's resume bug) |
| Export round trip | Transpose one expert's weight (the gpt-oss-20b load bug) |
| Token in, token out | Retokenize the decoded text instead of reusing the sampled ids |
| Logprob parity | Compute trainer logprobs at the wrong temperature |
| Serving parity | Serve with a template missing the generation prompt |

## Kill-test runbook (Phase 0)

Phase 0 runs 432 check-fixture pairs (3 models x 3 frameworks x 6 Tier 1 checks x 8 fixtures), all on a laptop CPU, and ends with Gate A.

1. **Pick targets.** The 3 most recent major open-weight releases. Include at least one with reasoning blocks and one with tool calling.
2. **Calibrate on a known-good model first.** Run every check and planted fault on a family TRL already ships a training template for, such as Qwen3 ([TRL #5471](https://github.com/huggingface/trl/issues/5471)). Every planted fault must be caught and every clean run must pass before you touch a new release.
3. **Pin.** Lock the latest TRL, Unsloth and Axolotl releases, one uv environment each.
4. **Run** all 432 pairs and collect the verdict files.
5. **Triage every fail, assuming your own code is wrong first:** is it the framework, the model's template, or your adapter? Early on, most "bugs" will be adapter mistakes. Fix and re-run.
6. **Shrink and search.** Cut each surviving fail down to a short standalone script, then search the framework's issue tracker so you don't file a duplicate.
7. **File** one issue per bug, using the template below, and log each one in the ledger.
8. **Gate A.** At least one bug a maintainer confirms means you go to Phase 1. Zero means you stop and switch to Crashproof Rollouts.

## Finding record and issue template

Every finding gets a ledger record, whatever happens to it. Retracted findings stay visible too, which is what makes the lab trustworthy.

| Field | What goes in it |
| --- | --- |
| id, date found | `P-0001`, the date |
| model, framework | Exact revision and pinned version |
| check, fixture, kind | Which check fired, on what input, bug or difference |
| evidence | Expected vs got, first divergence |
| repro | Path to the standalone script |
| issue | Link to the upstream issue |
| status | open, confirmed, fixed, retracted |
| fixed in | The release that fixed it, after a re-run passes |

The issue template is short, polite and reproducible in one paste:

```markdown
**Title:** [<area>] <one-line symptom> for <model> (<framework> <version>)

**What happens**
With `<setting>`, <what goes wrong, in one or two sentences>.

**Repro** (CPU, no full weights downloaded)
<short standalone script>

**Expected vs actual**
Expected <tokens or labels at positions ...>. Actual: <...>.

**Reference used**
<e.g. the model's own chat template via apply_chat_template>

**Environment**
<framework version>, transformers <version>, lockfile attached.
```

## Report card format and automation

The card is a static page built from the verdict files. Merging a pull request is the human review gate: nothing reaches the card without your merge.

**The card:**

- **Layout:** rows are model releases (newest first), columns are frameworks, and each cell shows pass / fail / inconclusive counts. Clicking a cell opens the per-check, per-fixture verdicts.
- **Traceability:** each cell links to its lockfile and verdict JSON, so anyone can re-run it.
- **Scope:** every card states which tiers ran, and that Tier 2 used tiny random-weight models.

**The automation:**

1. A scheduled GitHub Actions job polls the watchlist (model orgs and framework repos) a few times a day.
2. A new release starts a matrix job, one runner per framework, running Tiers 1 and 2 on CPU.
3. The job opens a draft pull request in the Parity repo with the new verdict files and any candidate findings.
4. You triage the candidates (shrink, search, file) and merge. The merge publishes the card.
5. When a framework or model ships a fix, the next poll re-runs the affected checks and updates the ledger.

## Self-improving loop internals

Two LLM agents propose, plain code decides, and every proposal is logged so the recursion test has data later.

| Agent | Sees | Never sees | Produces |
| --- | --- | --- | --- |
| Writer | Public issues, changelogs, template diffs, the accepted suite, breaker misses, gate feedback | The hidden vault | A check plus its planted fault, tied to the bug it targets |
| Breaker | Framework source, the adapters, the accepted suite | The hidden vault, the writer's prompts | A breakage: a small patch to a framework's data path plus the symptom it should cause |
| Gatekeeper | Everything it needs to run | Nothing hidden; it has no LLM | Accept or reject, with the failing gate named |

**The three gates, cheapest first, then a redundancy filter:**

1. **No false alarm.** The check passes on every known-good setup, 3 runs each to catch flaky checks.
2. **Self-test.** It catches its own planted fault.
3. **Catches its target.** It fails on the buggy version and passes on the fixed one, using a real pair from the open vault, or it kills the breaker's breakage.
4. **Adds something.** It catches at least one bug or breakage the current suite misses. Otherwise it's rejected, which keeps suite size honest (see the size confound under Evaluation).

**One iteration:**

1. The breaker produces a batch of breakages.
2. The suite runs against them, and the misses go to the writer.
3. The writer proposes checks within a fixed dollar and time cap.
4. The gates run, and the accepted checks merge as suite version vN.

**Writer versions:** the writer may edit its own prompt and tools between iterations. Each edit is a new, archived writer version, like DGM's archive. That's what makes the recursion test possible.

**Logged per proposal:** writer version, model ID, cost, gate results with reasons, wall-clock time.

**Your role:** review a sample of accepted checks each week, and log any edit you make separately. A silent human edit would contaminate the loop-vs-human comparison.

## Bug vault format

The vault is the scoreboard, so it gets stricter rules than anything else in Parity. A pair counts only if it reproduces every time, on both sides.

**Where bugs come from:** Parity's own confirmed findings, fixed upstream issues describing silent training bugs (TRL, Unsloth, Axolotl, verl, SkyRL, transformers), and well-known public ones like the 2024 gradient-accumulation bug.

| Field | What goes in it |
| --- | --- |
| id, source | `V-0001`, link to the issue or fix |
| bug class | e.g. end-of-turn mask, packing leak, resume step count |
| framework | Name |
| buggy version, fixed version | Each pinned with its own lockfile |
| trigger | Model, template, config and fixture that expose it |
| repro | A script that fails on buggy and passes on fixed |
| reproduced | e.g. buggy fails 5/5, fixed passes 5/5. Anything less and the pair is dropped |
| date fixed | Used for the time split |
| split | open or hidden |

**Reproduction rule:** run both sides 5 times before admitting a pair. One deep-learning benchmark review re-ran buggy and fixed versions 20 times, and only 54% of faults reproduced ([Jahangirova et al.](https://arxiv.org/abs/2412.16336)). Expect to drop a lot.

**The hidden split:**

- **By date:** every bug fixed after a cutoff date goes hidden, so no agent could have learned from it.
- **By framework:** one whole framework (e.g. verl) is hidden, for the transfer test.
- **Storage:** hidden pairs live in a separate private repo that agents' tools can't reach. Only the evaluation harness reads them.

## Evaluation protocol

Every suite version and every baseline is scored the same way on the hidden vault, and the results go into one public chart: hidden-vault recall per iteration, with intervals.

**Suite versions:** v0 is the hand-written suite, frozen at the end of Phase 3. vN is the suite after N loop iterations.

**Question 1 metrics (is the suite better?):**

| Metric | Definition | Rule |
| --- | --- | --- |
| Hidden-vault recall | Hidden pairs caught / all hidden pairs. Caught = fails on buggy and passes on fixed | Report with a 95% Wilson interval |
| Recall at equal size | Recall of a random k-check subset, averaged over draws, for each k | Compare suites at the same k, never raw totals |
| False alarms | Checks that fire on the known-good calibration set | Must be 0 for a version to count |
| Fixed-mutant score | Share of a fixed mutant set caught, generated once by a separate process, not the breaker | A guide only; it never overrides the vault |
| Prospective catches | Confirmed bugs caught in releases after a frozen tag | Frozen tag vs v0, counted per release |

```latex
\text{Wilson 95\% interval:}\quad \frac{\hat p + \frac{z^2}{2n} \pm z\sqrt{\frac{\hat p(1-\hat p)}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}},\qquad \hat p = \frac{\text{caught}}{n},\ z = 1.96
```

**Question 2 tests (is the loop the reason?):**

| Test | Setup | Passes when |
| --- | --- | --- |
| Baselines | B1 frozen v0. B2 the loop with no breaker. B3 human-only checks written over the same calendar window (report your hours) | The loop's recall at equal size beats all three, and the intervals don't overlap |
| Seeds | 3 loop runs from v0 with different seeds | The gain shows up in every run, not just one |
| Recursion | A probe set of 10 bug classes held back from the open vault (descriptions only), given once to each archived writer version | Later writers produce more accepted checks per dollar. Retire the probe set after use |
| Transfer | One framework excluded from the loop and kept in the hidden split | Recall on it rises along with the rest |

**What you may claim:**

- **Gate B:** pass baselines and seeds, and you can say the loop improves the suite.
- **Gate C:** pass prospective catches and transfer, and you can say it generalizes.
- **Recursion:** pass it too, and you can say the improver improves. If you pass everything except recursion, the honest word is self-extending.
