# Decision log

One entry per decision that changes scope, a pass rule, a tolerance, a target list, or a gate outcome. Newest first.

Format:

```
## YYYY-MM-DD: <short title>
- Decision:
- Why:
- Evidence: <file paths, run ids, links>
- Reversible? yes/no
```

Milestone numbers follow `docs/phases/phase-0-kill-test.md`. Entries up to and including "Open items for M4 (adapters)" use an older Phase 0 numbering: old M1 (platform spike) = M3's install check, done early; old M2 = M1; old M3 = M2; old M4 = M3.

## 2026-11-24: M2 skeptic review: five bugs fixed
- Decision: an independent review (the `skeptic` subagent) of everything M2 added found these. Each fix has a test that failed before it.
  1. **A pass without its self-test could be counted.** `summarize`, `write_verdict` and `read_verdict` now refuse a pass or fail unless its self-test was caught and every planted-fault record says caught. A raw `check.run` result, or a hand-edited JSON file, can't count (rule 3 at the record level).
  2. **The reference cache ignored key order in dicts.** It hashed conversations with sorted keys, but transformers' `tojson` keeps key order, so two conversations that differ only in key order shared stale token ids. It now hashes in order.
  3. **The reference silently trusted templates that ignore `add_generation_prompt`.** The assistant header became must-train without any error. It now refuses when the generation prompt adds nothing and the template has no generation markers. Inkling still computes, because its markers agree.
  4. **The `false` → `False` planted fault crashed on templates that refuse string tool arguments.** It crashed on Inkling and Hy4, making their multi-turn prefix cells inconclusive. It is now skipped when the declared template can't render it (decided from the template, before any framework output). With stand-ins, all six checks now compute on Inkling and Hy4.
  5. **No planted fault guarded the padding part of `t1.special_tokens`.** The pad = EOS fault also masked each row, so the row rule caught it first.
     - The real-world fault now masks only at collation, as transformers' collator does, so it guards the padding part.
     - A new synthetic per-row variant guards the row rule, and it is the only pad fault when the framework packs.
     - Mutation tests confirm each part is guarded. There are now 15 planted faults; 3 are real-world.
- Also found and fixed while doing this: `test_template_breaks_match_the_references_prefix_breaks` was vacuous. It compared the keys of the dict `apply_chat_template` returns in transformers 5 with token ids, so it could never fail. It now compares real ids, and passes.
- Not fixed: the review's design questions are open below and in `docs/STATUS.md`.
- Evidence: `tests/test_engine.py` (record-level self-test), `tests/test_checks_common.py` (cache key), `tests/test_reference_template.py::test_template_that_ignores_add_generation_prompt_is_refused`, `tests/test_check_multiturn_prefix.py::test_a_plant_the_declared_template_cant_render_is_not_applied`, `tests/test_check_special_tokens.py` (the two mutation tests). The review's scratch scripts lived in the session scratchpad; the tests are the durable record.
- Open for the human (from the review):
  - **DeepSeek V4.1:** every cell is `inconclusive`. Its generation prompt pre-fills `<think>`, but history turns render `</think>` instead, so the reference refuses on 4 of the 8 fixtures. Either anchor content at the longest common prefix and record "prefill missing from history" as a template property, or keep refusing.
  - **Circularity:** the official stand-in is circular for all six checks, and the TRL stand-in is circular for every check except template parity. Independent evidence comes only from unit tests (golden spans, TRL markers, KD-0001) and from the planted faults.
  - **Rules no planted fault guards on Qwen3:** `end_of_turn`'s `missing_or_replaced` and `wrong_label` branches, `label_mismatch`, the leading-special and count rules. Most planted faults are blatant; the review proposed harder variants.
  - **Other design questions:**
    - A cut row with zero must-train tokens passes truncation.
    - `declare()` is read with `base_cfg`, not the config used to encode.
    - A `packed` declaration is never verified.
    - Marker disagreements are not a first-class report property.
    - Multi-turn prefix can blame prefix handling for a data bug (dropped reasoning) that template parity owns.
    - The `history_false_as_python` source matches Prime Intellect's symptom, not its mechanism, and has no URL.
- Reversible? yes

## 2026-11-23: M2 follow-ups: declarations are verified, planted faults are tagged, template properties are reported
- Decision:
  - **Declarations are claims, and checks verify them.** Every verdict records the adapter's declaration and, per field, whether behavior confirmed it.
    - `consistent`: the behavior confirms the declared value.
    - `contradicted`: the behavior is exactly what a different value gives. That is an adapter error until shown otherwise (rule 4), so the verdict is `inconclusive`.
    - `mismatch`: the behavior matches no value. It is judged normally, so it can fail.
    - `unverified`: the behavior can't show the field.
  - What the checks verify:
    - **Chat template** (every check): the tokens are exactly the model's own template's render while another template is declared.
    - **Truncation mode**: the row is exactly what another mode gives (right, left or drop). This is verified only when something is cut.
    - **Batching**: rows show packing (a different row count, or position ids restarting inside a row) when padding was declared.
    - **Pad token**: every pad holds one other token. Pads holding several different tokens fail.
    - The **label convention** is recorded but not verified. A shifted-label pattern must stay a `fail`, or the `shift_labels_left` planted fault could never be caught.
  - This refines "a framework fails only when it doesn't match its own declared mode" (truncation). Output that is exactly another mode is now `inconclusive` (adapter error). Output that matches no mode (off-by-one, lost loss, re-appended stop token) still fails.
  - **Planted faults carry an origin:** `real-world` (with a source) or `synthetic`. Three of the 14 are real-world:
    - `mask_closing_tokens`: Plumbline's GLM-5.3-Flash bug.
    - `pad_is_closing_token_masked`: pad = EOS with id-based pad masking. transformers 5.24.0's `DataCollatorForLanguageModeling(mlm=False)` does `labels[labels == pad_token_id] = -100`.
    - `history_false_as_python`: Prime Intellect, May 2026.
    - The other 11 are synthetic.
  - **Template properties are reported at the model level.**
    - A template's prefix breaks are computed once per model and template (`parity.references.properties`).
    - A multi-turn prefix pass that inherits them names the property (`Verdict.template_properties`) and records `prefix_preserved: false`.
    - The report shows each property once per model and links to it from the cell (`parity.engine.report`). The Phase 1 card does the same.
- Heads-up for M4: frameworks may skip `empty_assistant` on purpose. If that's documented, it becomes a known-differences entry; if not, it's a candidate finding.
- Why: the human's review of M2 (2026-11-23).
- Evidence: `tests/test_check_*.py` (the contradiction tests), `tests/test_report.py`, `tests/test_tier1_suite.py`.
- Reversible? yes

## 2026-11-23: Tier 1 pass rules and planted faults (M2)
- Decision: the six Tier 1 checks, as built. Every rule is exact; there are no numeric tolerances in Tier 1.

  | Check | Compares against | Fails when | Planted faults | Not applicable |
  | --- | --- | --- | --- | --- |
  | `t1.template_parity` | model's own template | any token differs, unless a registry entry matches exactly (then `difference`) | drop the first newline token; swap the first stop token for another special token | never |
  | `t1.assistant_mask` | declared template's zones | a must-train token has no loss, a must-not-train token has loss, or a trained label isn't its token | unmask the prompt; zero the mask; shift labels left by one; start each trained span one token late | never |
  | `t1.end_of_turn` | declared template | a turn's closing stop token is missing, replaced, untrained or mislabelled | take the loss off every trained stop token | never |
  | `t1.special_tokens` | declared template | leading specials or special-token counts differ; a must-train stop token loses its loss; in a padded batch from the framework's collator, padding isn't at one end, changes the row, uses another token, or has loss | an extra leading special token (second BOS, or the pad token when there's no BOS); pad with the turn-closing token and mask it everywhere | the padding part, when the framework packs |
  | `t1.truncation` | declared template, cut by the framework's declared mode | the row isn't the cut reference, its loss doesn't follow the cut zones, or it drops/keeps against its mode | off-by-one at max length; keep the cut answer but lose its loss; re-append the stop token after cutting | never |
  | `t1.multiturn_prefix` | declared template's prefix renders | at a boundary, the framework breaks the prefix where the template doesn't, or breaks it differently | re-render earlier tool-call arguments as a Python repr (`false` → `False`); re-render earlier assistant turns with one character changed | fixtures with fewer than two assistant turns (4 of 8) |

  - Details:
    - **Dropped with no length limit** is a `fail` in every check. The exception is the truncation check's no-limit step, where it is `inconclusive`, so truncation isn't blamed for it.
    - **Truncation:** `long_answer` runs at its own `max_len` (128). The other seven fixtures run at their exact reference length, the boundary where nothing may be cut.
    - **Special tokens:** each fixture is batched with the fixture whose length differs most, padded to the longer row + 8. A pad token that is also a stop token (Qwen3's `<|endoftext|>`) is recorded in the verdict, not failed.
    - **Multi-turn prefix**, refining "Template properties are judged before frameworks" below:
      - A break is anchored at both ends. It is recorded as its distance from the end of the prefix, the tokens removed, and the tokens inserted.
      - A framework passes a boundary only if it keeps the prefix there, or breaks it exactly as its declared template does. The old rule, "a subset of the template's breaks", would have let any break through at a boundary where the template already breaks (on Qwen3 that is every boundary).
      - A break inherited from the template is a `pass` with the template's breaks recorded (`template_breaks_at`), not a `difference`. `difference` stays reserved for registry entries.
    - **Synthetic plants:** `history_edit` (multi-turn prefix) is synthetic so that every applicable fixture has a plant. The real-world plant, `history_false_as_python`, applies only to `tool_call`.
- Why: hard rules 3 and 5, and the decisions of 2026-11-23 on truncation, `not_applicable`, marker cross-checks and padding.
- Evidence: `src/parity/checks/tier1/`, `tests/test_check_*.py`. On the calibration model with both test stand-ins, every applicable verdict passes (KD-0001 aside), and every planted fault is caught (`tests/test_tier1_suite.py`).
- Reversible? yes

## 2026-11-23: Check engine rules for Tier 1 (M2)
- Decision:
  - **Adapters declare before they encode.** From the framework's own config and defaults, and before encoding anything, an adapter reports: the chat template the framework will render with, its label convention (`aligned` or `pre_shifted`), its truncation mode (`right`, `left` or `drop`), whether it pads or packs, and its pad token. `encode` always returns labels aligned with the input ids; an adapter for a pre-shifting framework shifts them back, and the verdict records which convention the framework uses.
  - **`not_applicable` is a fourth result.** Applicability is decided from the fixture alone, before any framework output exists, never from what the adapter returns. It is never a pass. The summary shows how many fixtures each check ran on, and a check that ran on zero fixtures for a framework is `inconclusive`.
  - **Planted faults.** A check can ship several. Which ones apply to a fixture is decided from the fixture, the model's own tokenizer and template, and the framework's declaration, never from the adapter's output. A verdict counts only if at least one applies and every one that applies is caught (its run ends in a fail of kind `bug`). Otherwise the result is `inconclusive`, and the clean result stays in the evidence.
  - **Which reference.** Template parity compares against the model's own template. The other five checks compare against the template the framework declares, so they test masks, stop tokens, padding, truncation and prefixes, not template choice. If the framework's tokens don't match its own declared template's render, those checks return `inconclusive` and point to template parity.
  - **Verdict fields added to `parity.verdict/v0`** (nothing is published yet): `known_difference`, `reason`, `conventions` (the framework's recorded choices), `reference` (template source and sha256, the official template's sha256, the transformers, tokenizers and TRL versions), and `plants` (each planted fault and whether it was caught).
  - **Test stand-ins until M3.** The checks are tested against stand-in encoders in `tests/stubs.py`: Qwen3's own template with labels from our reference, and TRL's vendored training template with labels from its generation markers. They are not adapters, live only in `tests/`, and never produce published verdicts. Passing on them proves the checks' logic, not anything about a framework.
- Why: rule 3 (a check proves itself in the same run), rule 4 (keep our choices apart from the framework's), and the decisions of 2026-11-23 on `not_applicable`, label conventions, truncation and padding.
- Evidence: `src/parity/adapters/base.py`, `src/parity/engine/`, `tests/test_engine.py`, `tests/test_stubs.py`.
- Reversible? yes

## 2026-11-23: Mask reference: the prefix diff, cross-checked against generation markers (settles Inkling)
- Decision:
  - The prefix diff is the assistant-mask reference for every template, with or without `{% generation %}` markers. This revises the rule in "Tier 1 reference, and how assistant spans are defined" below, that a template's markers define the mask.
  - When a template has markers, the mask transformers reads from them is cross-checked against the prefix-diff zones. Neutral tokens never count as a disagreement. Any other disagreement (a must-train token left unmarked, or a must-not-train token marked) makes the reference refuse: the verdict is `inconclusive`, and the disagreement goes to `/triage` as a possible model-template finding.
  - This settles the Inkling question from "Mask zones" below: Inkling's zones come from the prefix diff like every other template, and its markers are cross-checked, not trusted.
- Why: markers are the template author's claim. Taking them as the reference would test whether frameworks follow the claim, never whether the claim is right, and M0 picked Inkling partly to test exactly that.
- Evidence: on TRL's Qwen3 training template the two agree on all 8 fixtures; they differ only on the neutral newline after `<|im_end|>` (`test_trl_template_reference_uses_prefix_diff_and_its_markers_agree`). Markers moved onto the assistant header are refused (`test_markers_that_disagree_on_a_non_neutral_token_are_refused`).
- Reversible? yes

## 2026-11-23: Phase 3 prerequisite: isolate the hidden vault at the OS level
- Decision:
  - The guard hook only prevents accidents. It matches a path string in tool calls, so it can't stop a command that reaches the vault some other way.
  - Before any pair enters the hidden vault, the vault moves to a separate Linux user with `700` permissions, so this account (the one Claude Code runs as) can't read it at the OS level.
  - Hidden-vault evaluation runs as that user, through one wrapper that prints only aggregate numbers. The human runs it, not Claude Code. Every run is logged, and the log holds only what the wrapper prints.
  - `Bash(sudo *)` is in the deny list in `.claude/settings.json` from today.
- Why: Gates B and C rest on hidden-vault recall. If an agent could read the vault, that number would prove nothing. File permissions are a boundary; a string-matching hook is not.
- Found while logging this (checked 2026-11-23 with `id` and the WSL interop entry in `binfmt_misc`): this account has three ways around file permissions today. Each must be closed before the move counts.
  - It is in the `sudo` group. The deny-list entry is a guardrail; the boundary is that sudo asks for a password Claude Code never has.
  - It is in the `docker` group, which is root-equivalent without a password (a container can mount `/`). Remove the account from the group, or switch to rootless Docker.
  - WSL interop is on (`wsl.exe` is on the PATH), and `wsl.exe -u root` opens a root shell without a password. Turn interop off in `/etc/wsl.conf` (`[interop]` `enabled = false`, then `wsl --shutdown`).
- Done when, as this account, reading or listing the vault fails with "Permission denied" and none of the three routes above works. Record that check here.
- Follow-ups before Phase 3, not done yet:
  - `/run-eval`, Phase 4's `/run-eval` steps and SETUP.md (which puts `PARITY_HIDDEN_VAULT` in this account's `.env`) still assume Claude Code runs the evaluation.
  - `vault/README.md`, SETUP.md and Phase 3 M3 still give the old location in this account's home. The guard hook blocks Claude Code from writing that path, so the human edits those lines.
  - The blueprint's "separate private repo" storage line is superseded by this entry.
  - Keep the marker name the guard hook matches in the new vault path, so the hook still catches accidents.
- Update (same day): the first follow-up is done. `/run-eval`, Phase 3 M6, Phase 4 M5 and SETUP.md now say the human runs hidden-vault evaluation and Claude Code only records the aggregates. The lines that name the vault path are left for the human.
- Reversible? yes, but don't loosen it once the vault holds pairs.

## 2026-11-23: Adapter environments install the way each framework's docs say
- Decision:
  - Each adapter environment is installed the way the framework's own docs say: the documented source (PyPI, or the repo at the release tag), extras and build settings.
  - Any deviation is recorded in that environment's metadata (what differs from the docs, and why) and treated as a suspect in triage, to rule out before calling anything a framework bug. Running on CPU where the docs require a GPU counts as a deviation too.
  - SkyRL: install from its repo at the release tag, as its docs say (`uv sync --active --extra fsdp`), when its environment is first built in Phase 2, and the same in Phase 7. The PyPI install with custom `flash-attn` build settings that the spike used is not used for verdicts.
- Why: a verdict should describe what the framework's users actually run. An install route they don't use can resolve different versions (the spike's PyPI route got torch 2.15.0; SkyRL's repo install pins 2.14.0), and every undocumented choice of ours is one more thing that can pass for a framework bug (rule 4).
- Settles: open item 4 (SkyRL's install route) in "Open items for M4 (adapters)" below.
- Follow-up: the metadata format is set when the first adapter environment is built (Phase 0 M3), and `/triage` then lists that environment's recorded deviations as suspects.
- Reversible? yes

## 2026-11-23: Open items for M4 (adapters), from the platform spike
- No code yet. Each item becomes a decision when M4 settles it.
  1. **Reference per framework environment.** Each framework brings its own TRL and transformers (see the version table in the entry below). The reference runs inside each framework's own environment, and every verdict records the transformers, tokenizers and TRL versions it ran with (TRL is `none` when the env has none). The core env's transformers 5.24.0 is used only for the reference's own tests.
  2. **Registry entries only from evidence.** A known-differences entry is added only when calibration actually hits a difference and it has been checked against the framework's docs. No entries ahead of time for the TRL inside Unsloth (1.15.0) or Axolotl (1.14.1), since they may not use TRL's training-template swap at all. KD-0001 covers only TRL `==1.18.2` used directly.
  3. **Unsloth without a GPU.** Unsloth takes its CPU fallback: stock transformers + PEFT + TRL, per `unsloth/_cpu_init.py`. That's fine for Tier 1 data checks. Tier 2 numerics on Unsloth will need a GPU, to cover the code paths users actually run. The CPU fallback is confirmed in WSL (see the WSL spike entry below).
  4. **SkyRL's install route.** A plain PyPI install of `skyrl[skyrl-train]==0.5.0` fails on Linux: the `flash-attn` sdist won't build. One option is SkyRL's documented repo install (`uv sync --extra fsdp` at the tag), which uses SkyRL's prebuilt `flash-attn` for torch 2.14 + CUDA 13 and pins torch 2.14.0. The other is the PyPI install plus SkyRL's own `flash-attn` build settings, which the spike used (torch 2.15.0). Pick the route SkyRL users actually run and record it in the environment's lockfile. See the WSL spike entry below.
- Reversible? yes

## 2026-11-23: Platform spike re-run in WSL2: all five frameworks install and import their data path (M1)
- Decision:
  - The move to WSL2 holds. Every framework installs for real in WSL2 Ubuntu and imports its data path. M1 is done; M3 is next.
  - SkyRL is the one exception to a plain PyPI install. On Linux it needs SkyRL's own `flash-attn` build settings (below). Which SkyRL install route the M4 environments use is open item 4 in the entry above.
- Spike setup:
  - WSL2 Ubuntu 22.04.5 (kernel 6.6.87.2-microsoft-standard-WSL2), x86_64, 16 logical CPUs, no NVIDIA GPU (`torch.cuda.is_available()` is `False` in every environment), Python 3.12.13, uv 0.11.3.
  - Same framework versions as the Windows spike. They were still the latest on PyPI on 2026-11-23.
  - Each framework in its own environment: `uv venv --python 3.12 <env>`, then `uv pip install --python <env>/bin/python <pkg>==<version>`. Real installs, no dry runs.
  - `probe.py` imports only the data-path names listed below, from a neutral working directory, and writes the resolved versions to JSON.

  | Framework | Installs in WSL2 Ubuntu | Data path imports | Exact error | Docs on Linux |
  | --- | --- | --- | --- | --- |
  | TRL 1.18.2 | yes | yes: `SFTTrainer`, `SFTConfig`, `trl.data_utils`, `trl.chat_template_utils.get_training_chat_template` | — | no platform statement ([installation.md @ v1.18.2](https://github.com/huggingface/trl/blob/v1.18.2/docs/source/installation.md)) |
  | Axolotl 0.21.1 | yes (with `triton` 3.8.0, `xformers`, `bitsandbytes`) | yes: `axolotl.prompt_strategies.load`, `axolotl.prompt_strategies.chat_template` (`ChatTemplatePrompter`, `ChatTemplateStrategy`, `StrategyLoader`), `axolotl.utils.chat_templates.get_chat_template`, `axolotl.utils.collators` (`DataCollatorForSeq2Seq`, `V2BatchSamplerDataCollatorForSeq2Seq`), `axolotl.utils.data.sft.prepare_datasets` | — | requires an NVIDIA or AMD GPU, Python ≥ 3.12, PyTorch ≥ 2.13.0 ("2.13.0 recommended"); the documented install picks the CUDA build with `UV_TORCH_BACKEND` ([installation.qmd @ v0.21.1](https://github.com/axolotl-ai-cloud/axolotl/blob/v0.21.1/docs/installation.qmd)) |
  | Unsloth 2026.11.10 | yes (with `triton` 3.8.0, not `triton-windows`; `xformers`, `bitsandbytes`) | yes: `unsloth`, `unsloth.chat_templates` (`get_chat_template`, `train_on_responses_only`), `unsloth_zoo.dataset_utils`, all through its CPU fallback ("no GPU found, training on CPU") | — | "Unsloth works on **Windows, Linux, WSL** and **macOS**", CPUs included ([README on PyPI](https://pypi.org/project/unsloth/2026.11.10/)) |
  | verl 0.9.1 | yes | yes: `MultiTurnSFTDataset`, `RLHFDataset` | — | requires CUDA ≥ 12.8; "The uv workflow targets **Linux** with Python 3.12" ([install.rst @ v0.9.1](https://github.com/volcengine/verl/blob/v0.9.1/docs/start/install.rst)) |
  | SkyRL 0.5.0 `[skyrl-train]` | **plain PyPI install: no.** With SkyRL's own `flash-attn` build settings: yes | yes, with those settings: `skyrl.train.dataset.sft_dataset.SFTDataset`, `skyrl.train.dataset.dataset.PromptDataset`, `skyrl.train.dataset.collators.DefaultCollator`, `skyrl.train.dataset.preprocess.convert_prompts_responses_to_batch_tensors`, `skyrl.train.generators.utils.get_response_ids_and_loss_mask_from_messages` | Plain install: `Failed to build flash-attn==2.8.3.post1` … `ModuleNotFoundError: No module named 'torch'` | requires CUDA 13.0; the documented install is from the repo, `uv sync --active --extra fsdp` ([installation.mdx @ skyrl-v0.5.0](https://github.com/NovaSky-AI/SkyRL/blob/skyrl-v0.5.0/docs/content/docs/getting-started/installation.mdx)) |

- Versions each environment resolved to, and its size on disk:

  | Environment | TRL | transformers | tokenizers | torch | Size alone |
  | --- | --- | --- | --- | --- | --- |
  | trl 1.18.2 | 1.18.2 | 5.24.0 | 0.23.3 | 2.15.0 | 5.6 GiB |
  | axolotl 0.21.1 | 1.14.1 | 5.20.1 | 0.23.3 | 2.14.1 | 7.0 GiB |
  | unsloth 2026.11.10 | 1.15.0 | 5.17.0 | 0.23.3 | 2.14.1 | 6.1 GiB |
  | verl 0.9.1 | none | 5.10.4 | 0.22.2 | 2.15.0 | 6.0 GiB |
  | skyrl 0.5.0 [skyrl-train] | none | 5.16.1 | 0.23.3 | 2.15.0 | 6.9 GiB |

  - Every version matches the Windows resolutions in the entry below, and Axolotl matches its earlier Linux dry-run resolve. So M4 open item 1 holds on Linux.
  - **Total disk: 13.3 GiB (14.3 GB)** for all five environments. That is one `du -sb` over all five, which counts each file once: uv hardlinks identical files, so environments with the same torch share them. Each environment alone is 5.6 to 7.0 GiB.
  - Because of those hardlinks into uv's cache, the five environments added only 1.8 GB of new disk on this machine (`df` before and after), since the cache already held the wheels. On a fresh machine, budget about 14 GB; the cache and the environments share files when they're on the same filesystem.
- What changed from native Windows:
  - Axolotl installs. The Windows blocker (`triton` has no Windows wheels) doesn't apply on Linux.
  - Unsloth installs `triton`, not `triton-windows`, and leaves `sentencepiece` importable: `disable_sentencepiece_on_windows` acts only when `sys.platform == "win32"`. Checked: `import sentencepiece` (0.2.3) works after `import unsloth`. Neither Windows-only behavior from the entry below occurs.
  - SkyRL fails on Linux where Windows didn't. The `[skyrl-train]` extra requires `flash-attn==2.8.3.post1; sys_platform == "linux"`, so Windows never tried to install it. PyPI has that version only as an sdist, and its `setup.py` imports torch without declaring it as a build dependency.
    - SkyRL's own `pyproject.toml` @ skyrl-v0.5.0 avoids this in two ways. `[tool.uv.sources]` points `flash-attn` at SkyRL's prebuilt wheels (torch 2.14 + CUDA 13). `[tool.uv.extra-build-dependencies]` and `[tool.uv.extra-build-variables]` add torch as a build dependency and set `FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE`.
    - A PyPI install applies neither. The spike passed the second pair explicitly (`uv pip install --config-file skyrl-flash-attn-build.toml ...`). That builds `flash-attn` without its CUDA kernels, which the data path doesn't use. The resolution was otherwise unchanged (torch 2.15.0).
- Why: the spike checks the move before anything is built on it. Every framework's data path now imports on the operating system the verdicts will run on.
- Not covered: GPU code paths (no NVIDIA GPU here); SkyRL's documented repo install, which pins torch 2.14.0 and uses the prebuilt `flash-attn` (M4 open item 4); SkyRL without the `[skyrl-train]` extra (not re-run).
- Evidence: `docs/evidence/2026-11-23-m1-wsl-spike/` holds `probe.py`, the probe output per environment (`probe/*.json`), each environment's `uv pip freeze` (`freeze/*.txt`), the `flash-attn` build settings (`skyrl-flash-attn-build.toml`), and the failing plain SkyRL install log (`skyrl-plain-pypi-install.log`).
- Reversible? yes

## 2026-11-23: Develop and run Parity in WSL2 Ubuntu, not native Windows (M1 platform spike)
- Decision:
  - All development, framework environments, runs and Claude Code sessions move to WSL2 Ubuntu-22.04.
  - The repo lives in the Linux filesystem (`~/parity`), not under `/mnt/c`.
  - The Windows checkout stays only as a backup, and we never edit both copies.
  - Next step: re-run this spike inside WSL with real installs, then M3.
- Spike setup:
  - Native Windows 11, no NVIDIA GPU (Intel UHD), Python 3.12 via uv.
  - The latest PyPI release of each framework on 2026-11-23, each in its own environment (`uv venv`, then `uv pip install <pkg>==<version>`).
  - A probe then imported only the data-path modules an adapter needs.

  | Framework | Installs on native Windows | Data path imports | Exact error | Docs on Windows |
  | --- | --- | --- | --- | --- |
  | TRL 1.18.2 | yes | yes: `SFTTrainer`, `SFTConfig`, `trl.data_utils`, `trl.chat_template_utils.get_training_chat_template` | — | no platform statement ([installation.md @ v1.18.2](https://github.com/huggingface/trl/blob/v1.18.2/docs/source/installation.md)) |
  | Axolotl 0.21.1 | **no** | not reachable | `Because triton{sys_platform != 'darwin'}>=3.4.0 has no wheels with a matching platform tag (e.g., win_amd64) and axolotl>=0.21.1 depends on triton{sys_platform != 'darwin'}>=3.4.0, we can conclude that axolotl>=0.21.1 cannot be used.` | "### Windows: We recommend using WSL2 (Windows Subsystem for Linux) or Docker." ([installation.qmd @ v0.21.1](https://github.com/axolotl-ai-cloud/axolotl/blob/v0.21.1/docs/installation.qmd)). A dry-run resolve for Linux x86_64 succeeds (triton 3.8.0). |
  | Unsloth 2026.11.10 | yes (installs `triton-windows`, `xformers`, `bitsandbytes`) | yes: `unsloth`, `unsloth.chat_templates` (`get_chat_template`, `train_on_responses_only`), `unsloth_zoo.dataset_utils`, all through its CPU fallback | — | native Windows supported: PowerShell installer, winget, a `windows` extra ([README on PyPI](https://pypi.org/project/unsloth/2026.11.10/)) |
  | verl 0.9.1 | yes | yes: `MultiTurnSFTDataset`, `RLHFDataset` | — | requires CUDA ≥ 12.8; "The uv workflow targets **Linux** with Python 3.12" ([install.rst @ v0.9.1](https://github.com/volcengine/verl/blob/v0.9.1/docs/start/install.rst)) |
  | SkyRL 0.5.0 (`skyrl`, the successor of `skyrl-train`) | yes | only with the `[skyrl-train]` extra. Without it: `ModuleNotFoundError: No module named 'loguru'` | — | requires CUDA 13.0 and NVIDIA's Docker GPU images ([installation.mdx @ skyrl-v0.5.0](https://github.com/NovaSky-AI/SkyRL/blob/skyrl-v0.5.0/docs/content/docs/getting-started/installation.mdx)) |

- Versions each framework brought into its own environment. These are uv's resolutions on 2026-11-23; the real pins will come from each environment's lockfile in M4.

  | Environment | TRL | transformers | tokenizers | torch |
  | --- | --- | --- | --- | --- |
  | trl 1.18.2 | 1.18.2 | 5.24.0 | 0.23.3 | 2.15.0 |
  | unsloth 2026.11.10 | 1.15.0 | 5.17.0 | 0.23.3 | 2.14.1 |
  | axolotl 0.21.1 (Linux dry-run resolve, not installed) | 1.14.1 | 5.20.1 | 0.23.3 | 2.14.1 |
  | verl 0.9.1 | none | 5.10.4 | 0.22.2 | 2.15.0 |
  | skyrl 0.5.0 [skyrl-train] | none | 5.16.1 | 0.23.3 | 2.15.0 |

- Why:
  - Axolotl is one of the three Phase 0 frameworks, and it can't install on native Windows; its own docs recommend WSL2. A hybrid setup (Axolotl in WSL, the rest on Windows) would put verdicts on two operating systems, and triage would have to rule out the OS every time.
  - Unsloth runs Windows-only code. It installs `triton-windows` instead of `triton`. On Windows it also disables `sentencepiece` (`import_fixes.disable_sentencepiece_on_windows`), so transformers behaves as if sentencepiece weren't installed. Verdicts produced on Windows could differ from what Linux users get, and those would be our artifacts, not framework bugs (rule 4).
  - Everything later runs on Linux: GitHub Actions, verl and SkyRL (Linux plus CUDA), and the rented GPUs for Tier 3. SETUP.md already lists Windows with WSL as a supported setup.
  - It's cheapest now: there are no runs and no `envs/` yet, and the repo can be cloned straight from the Windows checkout.
  - It also removes Windows friction seen during setup: PowerShell 5.1 mangling quotes, the cp1252 console crashing on emoji, CRLF line endings.
- Also learned: the Phase 0 playbook says Unsloth needs an NVIDIA GPU just to import. That's out of date for 2026.11.10. With no accelerator it takes a CPU route (`unsloth/_cpu_init.py`) and prints "no GPU found, training on CPU".
- Not verified yet: nothing has been installed inside WSL. Axolotl was only dry-run resolved for Linux, and Unsloth's CPU route in WSL is assumed. (Verified later the same day: see the WSL spike entry above.)
- Evidence: the spike logs lived in the session scratchpad and were deleted afterwards. The commands above reproduce them.
- Reversible? yes

## 2026-11-23: Open note for Tier 3: TRL trains Qwen3 with think blocks that serving strips
- Note (not a decision, not a finding):
  - TRL 1.18.2 trains Qwen3 through `qwen3_training.jinja`, which keeps the think block in every assistant turn.
  - At inference, Qwen3's official template strips reasoning from assistant turns before the last user message.
  - So in multi-turn conversations, the history a served model sees differs from the history it was trained on. This is a train/serve mismatch.
- Follow-up: the Tier 3 serving-parity check must cover this case. Compare the training-format tokens with the served tokens on the `multi_turn`, `reasoning`, `tool_call` and `empty_assistant` fixtures, then decide whether it is a documented difference or a candidate finding.
- Evidence: `tests/test_reference_template.py::test_trl_template_renders_the_same_tokens_only_with_a_single_assistant_turn`; the TRL template header (link in the registry entry below).

## 2026-11-23: Template properties are judged before frameworks (multi-turn prefix)
- Decision:
  1. The multi-turn prefix check first tests the model's official template alone (`prefix_breaks` in the reference).
  2. If the official template isn't prefix-preserving, every framework using it inherits `difference (template)` at exactly those boundaries. The difference is attributed to the model, not the framework.
  3. A framework fails only if it breaks the prefix property beyond what its declared template does. Its breaks must be a subset of the breaks of the template it says it uses, whether that's the official template or its own training template.
  4. Known, documented template properties are not new findings: no finding record and no issue.
- Why: otherwise every framework would be blamed for the model's template. On Qwen3 that would mean 4 false fails out of 8 fixtures on the calibration model itself (rule 4: assume our code and our reference are wrong first).
- Evidence: Qwen3's official template breaks after assistant turns (`multi_turn` (2, 4); `tool_call`, `reasoning`, `empty_assistant` (2)), per `test_qwen3_template_is_not_prefix_preserving_after_assistant_turns`. The TRL training template's header says it was changed to be prefix-preserving.
- Reversible? yes

## 2026-11-23: Known-differences registry
- Decision:
  - A divergence counts as `difference` only if it matches a registry entry **exactly**. Everything else stays `fail`.
  - Each entry covers one documented, deliberate behavior and has: id, framework, version range, model family (plus the exact template it applies to), check id, exact divergence signature, and doc link.
  - No wildcard entries: no "any version", no "any model", no "any check". Widening an entry, for example to a new framework version, means verifying it again and logging it here.
  - Calibration is green when there are zero unexpected fails. Matched differences are listed by entry id in the calibration report.
  - The registry becomes a data file with the first check in M3. Adding or changing an entry needs an entry in this log.
- First entry, **KD-0001**:
  - Framework / versions: `trl ==1.18.2`, the only version verified so far.
  - Model family: Qwen3. It applies only when the model's chat template is byte-identical to TRL's `qwen3.jinja`, because that's the comparison TRL uses to swap in its training template. Verified for `Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca` (4,168 chars, identical). The training template used must have sha256 `1907918feb7afcf18fac8db1a419985b4773ac434e353cfae91b1949dcbf592b`.
  - Check: `t1.template_parity`.
  - Signature: start from the official render. For every assistant turn the official template renders without a think block, insert `<think>\n` + `reasoning_content.strip("\n")` + `\n</think>\n\n` immediately after that turn's `<|im_start|>assistant\n`. TRL's render must equal the result exactly; any other divergence is a `fail`.
  - Expected on the calibration fixtures: `multi_turn`, `tool_call`, `reasoning` and `empty_assistant` diverge in this way only; the other 4 fixtures don't diverge at all.
  - Docs:
    - The template header, lines 1–9 ("Removed the loop.index0 > ns.last_query_index conditional; always include thinking block"): https://github.com/huggingface/trl/blob/v1.18.2/trl/chat_templates/qwen3_training.jinja
    - TRL's chat templates README, lines 5–10 (training templates are swapped in by identity comparison, for prefix preservation and generation markers): https://github.com/huggingface/trl/blob/v1.18.2/trl/chat_templates/README.md
- Why: rule 4 and the zero-retractions goal. A deliberate, documented choice must not be filed as a bug. An exact, non-wildcard match makes sure the registry can't hide a real bug that happens to sit next to a documented behavior.
- Reversible? yes

## 2026-11-23: Mask zones for the assistant-mask check
- Decision: every reference token falls into exactly one zone.
  - **must-train:** assistant content plus its stop token. That's everything the template renders for the assistant turn after the generation prompt (think block, text, tool calls), through the stop token. In the reference this is `[token_start, token_end)`.
  - **must-not-train:** system, user and tool content, and every role header, including whatever the generation prompt pre-fills.
  - **neutral:** separator tokens after the stop token and before the next header, like the `\n` after `<|im_end|>`. In the reference this is `[token_end, token_message_end)`.
  - Pass rule: fail if any must-train token is unsupervised, or any must-not-train token is supervised. Neutral tokens never cause a fail either way. The verdict records the framework's neutral convention (`trained`, `not_trained` or `mixed`, with counts).
- Why: the newline after the stop token is never generated, so training on it is harmless, and frameworks differ on it. TRL's markers train it; our reference doesn't. Calling that a fail would be noise, but recording it keeps the difference visible.
- Open for M3: how the zones map onto templates that ship their own generation markers (Inkling). The likely rule: marked tokens through the stop token are must-train, and marked tokens after it are neutral. Decide and log this before running Inkling.
- Reversible? yes

## 2026-11-23: Tier 1 reference, and how assistant spans are defined
- Decision:
  - Reference tokens are `apply_chat_template(messages, tools=...)` with default options and the model's own template (transformers 5.24.0, pinned in `uv.lock`).
  - Templates with `{% generation %}` markers: the template's markers define the mask, read through `return_assistant_tokens_mask`.
  - Templates without markers: a prefix diff anchored in the full render. Message start = `len(render(messages[:k]))`. Content start = the same render with `add_generation_prompt=True`. Message end = found by cutting the message out of the full render and checking the result exactly equals `render(messages without k)`.
  - The supervised span runs from the content start through the turn's last stop token (tokenizer EOS plus the `eos_token_id`s in `generation_config.json`). The separator after the stop token (Qwen3: `\n`) is recorded as `token_message_end` but not supervised, because the model never generates it.
  - Any ambiguity raises `TemplateReferenceError`, which becomes `inconclusive`, never `pass`. Examples: a boundary inside a token, a prefix that doesn't match, or no stop token.
- Why:
  - The playbook's literal method (diff `render(messages[:k])` against `render(messages[:k+1])`) fails on the calibration model itself. Qwen3 renders a turn with an empty `<think></think>` when it is the last message, but without one once later messages follow.
  - Anchoring every boundary in the full render avoids that problem without re-implementing any template.
- Evidence:
  - `tests/test_reference_template.py`.
  - On TRL 1.18.2's `qwen3_training.jinja` (vendored in `tests/data/trl/`), the prefix-diff spans exactly equal the mask from TRL's generation markers on all 8 fixtures.
  - Hand-written golden spans match on 5 fixtures.
- Consequences to settle in M3, before calibration:
  1. **Trailing separator.** TRL's markers also supervise the `\n` after `<|im_end|>`, and ours don't. The assistant-mask check must decide whether a separator after the stop token is allowed. Log that tolerance here when it's decided.
  2. **TRL's training template differs from Qwen3's official one by design.** TRL always emits a think block, and its template header documents this. On `multi_turn`, `tool_call`, `reasoning` and `empty_assistant`, TRL's tokens differ from the official template's. If TRL uses that template, template parity on the calibration model will show a documented `difference`. Calibration must count it as expected, not as a fail.
  3. **Qwen3's own template isn't prefix-preserving after assistant turns.** `prefix_breaks` per fixture: `multi_turn` (2, 4); `tool_call`, `reasoning` and `empty_assistant` (2). The multi-turn prefix check must blame these on the template, not on a framework.
- Update (same day): all three are settled by the entries above. Item 1 by "Mask zones", item 2 by "Known-differences registry" (KD-0001), item 3 by "Template properties are judged before frameworks".
- Reversible? yes

## 2026-11-23: Phase 0 targets and calibration model pinned (M0)
- Decision: run the kill test on these 3 targets plus 1 calibration model, each pinned to an exact Hugging Face commit:

  | Role | Model | Revision (commit SHA) | Released | Reasoning | Tool calls | `{% generation %}` markers | License |
  | --- | --- | --- | --- | --- | --- | --- | --- |
  | Target | `thinkingmachines/Inkling-2-Mini` | `7a67ae2314b44ef05c87fd1d4d311c58aaeb18f6` | 2026-11-20 | yes | yes | yes (ships its own) | Apache-2.0 |
  | Target | `tencent/Hy4-flash` | `eccd3123eec73f2040f4c9ed0de1883593470510` | 2026-11-16 | yes | yes | no | Tencent custom (`tencent-hy-license`) |
  | Target | `deepseek-ai/DeepSeek-V4.1-Flash-1017` | `08c8657f02291e11b6f6b63e0e16014f8e748518` | 2026-10-16 | yes | yes | no | MIT |
  | Calibration | `Qwen/Qwen3-0.6B` | `c1899de289a04d12100db370d81485cdf75e47ca` | 2025-04-26 | yes | yes | no (TRL ships `qwen3_training.jinja`) | Apache-2.0 |

- Why:
  - All three targets are major-lab releases from the last ~5 weeks. All are ungated, and all load with a standard tokenizer class, so no repo code runs.
  - All have both reasoning blocks and tool calls, so they exercise all 8 fixtures. All appear to strip `<think>` from earlier turns, which exercises the multi-turn prefix check.
  - They cover two mask cases. Inkling ships `{% generation %}` markers, so we test whether those markers are right. Hy4 and DeepSeek ship none, so each framework has to find the assistant spans on its own.
  - DeepSeek V4.1 Flash is the most widely fine-tuned candidate.
  - Qwen3 is the known-good calibration family, because TRL ships a training template for it (TRL #5471).
  - Candidates I rejected:
    - `mistralai/Mistral-Large-4-1T-A52B`: TRL already ships a training template for it, so TRL bugs there are less likely.
    - `microsoft/Aion-1.0-Instruct-2.3B`: no reasoning blocks.
    - `moonshotai/Kimi-K3.1-Pro`: its tokenizer is custom repo code (`tokenization_kimi.py` + tiktoken).
    - Muse Spark Open and GLM-5.3: released before the ~2-month window.
- Evidence:
  - Hugging Face Hub API, queried 2026-11-23. Each SHA resolves, and each was the repo's latest commit when pinned.
  - At each pinned revision, `tokenizer.json` and `tokenizer_config.json` are present, and the targets also have `chat_template.jinja`.
  - Template features were read from the template text at the pinned SHA. None contain CR line endings.
  - The TRL template list was read from `trl/chat_templates/` on main.
- Notes to carry forward:
  - Inkling and Hy4 were both updated on the pin date (2026-11-23). If they fix their templates later, diff the new version against these pins.
  - Hy4 uses the `TokenizersBackend` tokenizer class, which needs transformers v5. If a framework env pins transformers v4, it may not load the tokenizer. M1 will show this.
  - The Hy4 license hasn't been reviewed yet. The human should read it before any Hy4 results are published.
- Reversible? yes (re-pin and re-run; earlier verdicts keep their recorded revisions)

## 2026-11-23: Parity chosen as the career-first project
- Decision: build Parity (Day-0 correctness lab) with a self-improving check suite and a two-question evaluation as core deliverables.
- Why: highest portfolio ceiling, objectively checkable, CPU-first. See `docs/plan/04-decision-history.md`.
- Evidence: plan doc https://claude.ai/artifact/LuCqsapFCxPhWiaD5RJzcu
- Reversible? yes (fallback: Crashproof Rollouts if Gate A fails)
