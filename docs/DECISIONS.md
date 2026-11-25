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

## 2026-11-25: Unsloth's GPU run is in: its primary result; the non-thinking table exploratory; P-0005 not yet attributed; H-10
- **The run** (Colab Tesla T4, driver 580.82.07 = CUDA 13.0, so the main lockfile; 109 packages as locked; Unsloth on `cuda`):
  - Verified by `docs/evidence/2026-11-25-m3-unsloth-gpu/ingest.py`: every file against its manifest, and `git archive` of `15ed6df` rebuilt here with the sha256 Colab recorded. The zip is kept byte for byte under `raw/`.
  - Results: thinking (primary) 36 pass, 8 inconclusive, 4 n/a, 0 fail; non-thinking 17 pass, 14 fail, 13 inconclusive, 4 n/a.
  - 81 of 96 cells as the docs-alone predictions said. All 15 misses are `t1.special_tokens`, the cells the predictions named in advance.
- **Decisions (human):**
  1. Unsloth's GPU results are its primary table, with CPU as the labeled secondary "CPU backend" table. The CPU summary now says so.
  2. The non-thinking table is exploratory. Its 14 fails (assistant mask and truncation on 7 fixtures) share one mechanism, now P-0006, status "exploratory": `train_on_responses_only` trains the last turn's empty think block, which the reference treats as the generation prompt's pre-fill. Pinned (backend gpu), so nothing is untriaged; not counted against Unsloth.
     - Skeptic, each point verified: Unsloth documents non-thinking training as the soft switch (the recommended notebook appends `/no_think`, and the empty block is model output). Our reference models the hard switch (`enable_thinking=False`; Qwen's model card and TRL's training template agree with that zone).
     - Its proposal, open for the human: reclassify as `not_configurable` (Unsloth documents no hard-switch training), add a known difference, or add a soft-switch reference.
  3. Rule 4 before linking `t1.special_tokens` to P-0005. The GPU evidence shows the swapped collator, `DataCollatorForSeq2Seq(pad_to_multiple_of=None)`, but not `trainer.args` or the collator before the swap. So the 15 cells aren't attributed yet.
     - The adapter's declaration now also records `trainer.args.pad_to_multiple_of`.
     - `findings/P-0005/adapter_check.py` (objective all vs assistant_only at the check's `pad_to`) settles it on the next trip.
  4. H-10 (ours): the P-0005 repro loaded 4-bit without LoRA, and crashed on GPU. Fixed to follow the notebook (`get_peft_model` first).
  5. The kit gains a P-0005-only mode (`RUN = "p0005"`, `colab_kit.py --only p0005`): install, smoke, then P-0005. The checks step it skips took 1147 s of the run's ~22 minutes. It runs:
     - the repro padded, with `padding_free` left to Unsloth, and with one real `trainer.train()` step that records the batch training draws;
     - then the adapter's `pad_to` check.
- **H-11 (ours, found by the skeptic):** on GPU the adapter read the collator in the inference state that Unsloth's `SFTTrainer.__init__` ends in (padding left); `train()` enters training mode first (padding right).
  - The adapter now makes the same `model.for_training(...)` call first.
  - CPU re-run: 96 of 96 verdicts identical. No GPU verdict was affected.
- **Stale report text (ours):** the secondary table's header still said "the same encoded rows… No extra framework runs", untrue since 2026-11-24. Fixed in `report.py`, and all four summaries re-rendered: only that line changed, plus the CPU banner.
- **Also seen, recorded, not judged:** xformers can't load with torch 2.14.1+cu130, so Unsloth uses PyTorch attention. A consequence of not running `install-kernels`; off the data path.
- **P-0004:** confirmed CPU-only. GPU rows equal the render in all 15 template-parity cells that run.
- Reversible? yes

## 2026-11-25: The P-0005 repro rides along on the GPU run, as an optional notebook cell (human)
- Decision (human): add `findings/P-0005/repro.py` to the GPU kit as an optional cell. Its output goes in its own folder inside the results zip, apart from the verdicts, and it must not affect the main run.
- How:
  - Notebook cell 6 runs `colab_kit.py --extra p0005` after the main run, in the environment the main run synced. It refuses unless that run's smoke step confirmed Unsloth's GPU route.
  - It writes only `P-0005/` (two logs, `run.json`, its own `MANIFEST.json`).
  - The zip is rebuilt with every main entry copied byte for byte, and swapped in only once it reads back clean. A failed repro is recorded, and the main results still zip; a failed rewrite leaves the original zip untouched (tested, and the failure path checked for real on CPU).
- Two variants:
  - padded (`packing=False, padding_free=False`), the repro as committed;
  - `--padding-free-unset`, new and opt-in: batching left to Unsloth's GPU defaults, with `max_steps=60` as the Thinking-2507 notebook sets.
  - Without `max_steps`, Unsloth's GPU route picks packing, by P-0005's code reading, and two short rows would pack into one.
- A crash also exits 1, the repro's "dropped" code, so a run counts as complete only if it printed its result line.
- Evidence: `docs/gpu-run/colab_kit.py` (`run_p0005`, `add_folder_to_zip`), `tests/test_gpu_kit.py`, `findings/P-0005/record.md` (Repro).
- Reversible? yes

## 2026-11-25: The Unsloth GPU run: predictions first, a backend record, a Colab kit, and a CUDA 12.6 lock variant
- **Predictions first** (`docs/evidence/2026-11-25-m3-unsloth-gpu/predictions.md`, commit `290eff2`): 96 cells (both modes) from Unsloth's docs and pinned notebooks alone, committed before any kit code existed.
  - They're not blind: the CPU run and P-0004/P-0005's code reading were already known, and the file says so.
  - It names in advance the cells where that code-based evidence disagrees with the docs. The main one is special tokens: the docs say pass; P-0005 says inconclusive.
  - The docs-alone score is the headline.
- **`Verdict.backend`** ("gpu" or "cpu", None for frameworks with no such choice):
  - The Unsloth adapter reads it from `unsloth.DEVICE_TYPE`.
  - Root causes carry a backend too, matched exactly. So P-0004 (`"backend": "cpu"`) never explains a GPU fail, even with identical evidence (tested on the real CPU verdicts relabeled as GPU: 15 untriaged, 0 linked).
  - Reports name the framework with its backend, e.g. "unsloth@2026.11.10 (gpu)". Once a framework has GPU results, its CPU results go in a separate "CPU backend (secondary)" table under each mode, as decided 2026-11-24.
  - The CPU run's saved verdicts predate the field. Its `--relink` records `backend: "cpu"` on them (the run was on CPU, per `environment.json`); only that line changed in the 96 files.
- **The kit** (`docs/gpu-run/`): a Colab notebook, plus `colab_kit.py` (standard library only) and `run_checks.py`, packaged by `build_kit.py` as `git archive` of a clean HEAD.
  - The commit id travels in the zip comment.
  - The kit refuses to run on CPU (`UNSLOTH_DISABLE_CPU_FALLBACK=1`, then a smoke step).
  - Subprocesses get no secret-looking variables and no `PYTHONPATH`, and Hugging Face's implicit token lookup is off.
  - Every step is logged, and a failed run still produces the zip.
  - The model list is a parameter, for M5.
- **CUDA 12.6 lock variant** (`envs/unsloth-2026.11.10/cu126/`):
  - The main lock's torch 2.14.1 is PyPI's CUDA 13.0 build, which needs a driver that supports CUDA 13. Colab's T4 runtime reported CUDA 12.4 in Unsloth's own saved notebook output (2025).
  - The variant takes torch 2.14.1+cu126 and torchvision 0.29.1+cu126 from PyTorch's CUDA 12.6 index, which has no 0.29.2 build.
  - On Linux x86_64 it is identical to the main lock otherwise (Unsloth, unsloth-zoo, TRL, transformers, tokenizers, datasets, PEFT, bitsandbytes, triton, xformers); `tests/test_gpu_kit.py` holds it to that.
  - The kit picks a lock from the driver's CUDA version (13.0 or newer: main; 12.x: the variant; older: stop) and records the variant as a deviation.
  - A committed lock, rather than an ad hoc `pip install` of another torch, so every version stays pinned (rule 7).
- **Deviations from Unsloth's documented Colab install, recorded in the run's `environment.json`:**
  - a locked uv environment instead of the notebook's `pip install --no-deps` cell on top of Colab's torch;
  - `unsloth install-kernels` not run: optional compute kernels, off the data path, and no forward pass runs;
  - TRL 1.15.0 and transformers 5.17.0, not the notebook's pins, as on CPU;
  - Python 3.12.
- **Not in the kit:** the P-0004 and P-0005 repros. The human scoped the kit to the adapter and the six checks. Changed the same day: P-0005's repro is now an optional cell (entry above).
- Evidence: `docs/gpu-run/GPU-RUN.md`, `tests/test_gpu_kit.py`, `tests/test_root_causes.py`, `tests/test_adapter_unsloth.py`.
  - Local check of `run_checks.py` on CPU (`--allow-cpu`, 2 min 43 s): it reproduces the CPU M3 counts exactly, and all 15 fails link to P-0004.
  - Also checked: `uv sync --locked --dry-run` of the variant, and the kit's freeze script on the main environment (109 packages, no mismatches).
- Reversible? yes

## 2026-11-24: After M3 Unsloth: P-0004 on the card as CPU-only, a GPU run next, P-0005 opened, predictions-first enforced
- Decisions (human):
  1. **P-0004:** shown on the report card as Unsloth CPU-backend behavior, labeled CPU-only, low impact (`card_label` in `findings/root_causes.json`, rendered next to the id).
     - Not shrunk.
     - Not a new report to TRL either: the underlying append is TRL's text-dataset behavior from TRL #3318.
     - Revisit after the GPU run.
  2. **Unsloth's labels:** no workaround in the adapter (rule 7). Next session is a GPU run. Unsloth's GPU route becomes its primary result; the CPU results stay as a labeled secondary "CPU backend" table. Until then, the Unsloth summary is headed "Backend: CPU".
  3. **P-0005**, triaged as its own finding, class "config value silently ignored": `train_on_responses_only` drops `SFTConfig.pad_to_multiple_of` when it swaps the collator, with no message.
     - It shows as inconclusive padding parts, not fails, so no verdict is pinned to it.
     - Skeptic: real candidate, silent (checked at DEBUG during a real training step), not fixed on unsloth-zoo main.
     - It only happens with padded batches. That is the CPU route's default and what the recommended Qwen3 notebook sets (`padding_free = False`); on GPU, Unsloth often chooses packing or padding-free instead.
     - To be confirmed on the GPU run before any draft. The repro now sets padded batches explicitly, so it can't give a false negative there.
  4. **Predictions first, made structural.** The guard hook (`.claude/hooks/guard.py`, rule 3) refuses any shell command that creates or runs code in `envs/<framework>-<version>` until a committed `docs/evidence/*/predictions.md` names that framework and version in its first line.
     - Uncommitted predictions, or predictions for another version, don't count.
     - `trl-1.18.2` predates the rule and is grandfathered by name; the list is pinned by a test to that one entry.
     - The `/new-adapter` skill now starts with step 0: pin the version, write predictions from the docs alone (reading docs, notebooks and GitHub source is fine; installing, importing or running anything is not), commit them on their own, and only then create the environment.
     - Anything learned later goes in a dated note below the predictions, never into them.
- Why (4): in M3 Unsloth, a data-path probe ran before the predictions were written, so some predictions weren't blind. A written rule didn't prevent it; a guard does.
- Known cost of (4): the guard reads command text. A shell command that only writes text naming an environment with no committed predictions is refused too; write such files with the Edit/Write tools.
- Evidence: `.claude/hooks/guard.py`, `tests/test_guard.py`, `.claude/skills/new-adapter/SKILL.md`, `findings/P-0004/record.md`, `findings/P-0005/record.md`, `findings/LEDGER.md`.
- Reversible? yes

## 2026-11-24: M3 Unsloth adapter, the run on Qwen3, and one candidate (P-0004, CPU backend only)
- **Adapter** (`src/parity/adapters/unsloth.py`): Unsloth's recommended Qwen3 notebook, step by step, plus `train_on_responses_only` with the explicit ChatML parts (the recipe choice is logged below).
  - It loads a local tiny `from_config` Qwen3 plus the real Qwen3-0.6B tokenizer files through `FastLanguageModel.from_pretrained`, as the human asked; Unsloth accepted it.
  - `declare()` reads the live tokenizer, trainer args and collator. The serving mode reaches Unsloth as `enable_thinking=False` in the formatting `apply_chat_template` (citation level: docs page, inference example).
  - Tests: `tests/test_adapter_unsloth.py`, 19 pass and 1 expected failure (strict). The contract "rows equal the declared render" fails, by P-0004.
- **Environment:** `envs/unsloth-2026.11.10`, the spike's freeze. Deviations in `environment.json`:
  - CPU fallback; the GPU loader steps are not exercised;
  - 4-bit becomes 16-bit on CPU;
  - Python 3.12;
  - the notebook's own TRL/transformers pins differ.
  - The statistics opt-out (`UNSLOTH_DISABLE_STATISTICS=1`) is set.
- **Predictions first, with a disclosed slip:** `docs/evidence/2026-11-24-m3-unsloth/predictions.md` was committed before any check ran (`daba17e`), but after a data-path probe; it says which cells rest on the probe. 48 of 48 primary cells came out as predicted.
- **Results** (`docs/evidence/2026-11-24-m3-unsloth/`):
  - primary: 8 pass (`t1.end_of_turn`), 8 fail (`t1.template_parity`), 28 inconclusive, 4 n/a;
  - secondary: 7 pass, 7 fail, 30 inconclusive, 4 n/a;
  - 15 fail verdicts, 1 root cause, 0 untriaged.
- **P-0004** (triaged, skeptic: real candidate, CPU backend only).
  - On CPU, Unsloth runs stock TRL's dataset preparation, which appends a second `<|im_end|>` after Qwen3's `<|im_end|>\n`. Unsloth's GPU-route preparation (`unsloth_zoo.sft_prepare_dataset`, intent stated in unsloth-zoo PR #2558) leaves chat-formatted text alone: 16/16 equal the render.
  - The extra token is the last one. Under assistant-only it gets no loss; under the notebook's all-token training it does, but generation never reaches it.
  - Every other check is inconclusive or blocked because rows don't align, so **Unsloth's labels are not judged yet**.
- **Also seen, untriaged:** `train_on_responses_only` drops `SFTConfig.pad_to_multiple_of` when it swaps the collator. By code reading this happens on both backends. The `special_tokens` padding part is inconclusive.
- Reversible? yes

## 2026-11-24: Unsloth's assistant-only recipe: no original-Qwen3 notebook has one; the choice made
- Finding while preparing M3 Unsloth: **Unsloth's Qwen3 docs and its notebooks for the original Qwen3 models show no assistant-only recipe.**
  - The Qwen3 page marks "Qwen3 (14B) Reasoning + Conversational" as the recommended notebook (`unslothai/notebooks@ebe88117`). It trains on every token: it formats text with `tokenizer.apply_chat_template(tokenize=False)`, and `SFTTrainer(dataset_text_field="text")` labels the whole text.
  - The same holds for the plain 14B, 32B and 0.6B ExecuTorch Qwen3 notebooks.
  - `train_on_responses_only` appears only in the Qwen3-2507 notebooks and Qwen3.8. Thinking-2507 and Tool-Calling pass the explicit ChatML parts; Instruct-2507, QAT and Qwen3.8 use auto-detection.
- **Decision (mine; the human's tie-break, "the one in the Qwen3 notebook", had no assistant-only recipe to pick):**
  - the recommended notebook's loader, formatting and `SFTTrainer`;
  - plus `train_on_responses_only(trainer, instruction_part="<|im_start|>user\n", response_part="<|im_start|>assistant\n")` from the Qwen3-4B-Thinking-2507 notebook (`unslothai/notebooks@0c6268b8`). Qwen3-0.6B is a hybrid thinking model, closest to the Thinking notebook.
  - Probe: auto-detection picks the same parts and gives identical labels on 4 fixtures, so the choice of call doesn't move results.
- **Mode setting:** `enable_thinking=False` in `tokenizer.apply_chat_template` while formatting (the Qwen3 page). The recommended notebook also appends " /no_think" to user turns of non-reasoning data; that changes content, so it is not applied.
- **Loader:** `FastLanguageModel.from_pretrained` accepts a local folder with a tiny `from_config` Qwen3 and the real tokenizer files. On CPU this is `unsloth_zoo.cpu.FastCPUModel`; the deviations are in `envs/unsloth-2026.11.10/environment.json`.
  - It leaves the template and pad token unchanged and sets `model_max_length` to `max_seq_length`.
  - `load_in_4bit=True` loads 16-bit on CPU, by Unsloth's own choice.
- **Process slip, disclosed:** a data-path probe ran before `predictions.md` was written, so some predictions rest on observed behavior. `predictions.md` says which.
- Reversible? yes

## 2026-11-24: TRL's per-row mode setting counts as documented; citation level "docstring only"; SFTConfig docs gap
- Decision (human): TRL's per-row `chat_template_kwargs` key counts as a documented way to set the serving mode. The function is public, TRL tests it, and it works through SFTTrainer.
- **Citation level: docstring only.** At v1.18.2, TRL's rendered docs have no API-reference entry for `trl.apply_chat_template`: `docs/source/data_utils.md` autodocs only `is_conversational`, `maybe_convert_to_chatml`, `extract_prompt` and `unpair_preference_dataset`. The function appears only in a usage example in `dataset_formats.md` (L302-313), which doesn't mention the key. The key itself is described only in the docstring (`trl/data_utils.py#L325-L327`). Recorded in the adapter's citation.
- **Possible docs gap, not a finding:** `SFTConfig` has no `chat_template_kwargs` field (passing one raises `TypeError`). `GRPOConfig` and others have one, and `grpo_trainer.md` shows it. TRL's SFT docs never say how to pass chat-template arguments.
- Reversible? yes

## 2026-11-24: Non-default serving modes get each framework's documented mode setting; P-0003 is a harness error
- Decision (human): "Same rows" assumed a framework's output doesn't depend on the mode, and that was wrong. This revises the earlier serving-mode entry.
  - **Primary (the model's default mode, thinking):** frameworks still get nothing, so they encode with their own defaults. The kill test stays at 432 runs.
  - **Secondary (any other mode):** the adapter gives the framework its documented setting for the mode, with the model's chat-template arguments (`parity.serving.framework_mode`, `template_kwargs`). These are extra runs, for the secondary table only.
    - TRL 1.18.2: a `chat_template_kwargs` key on every dataset row. It is documented in the public `trl.apply_chat_template` docstring (`trl/data_utils.py#L325-L327`), read by SFTTrainer (`sft_trainer.py#L1363-L1367`), and tested (`tests/test_sft_trainer.py#L1366`). It isn't on the SFT docs page; `SFTConfig` has no field for it.
    - Axolotl 0.21.1: the top-level `chat_template_kwargs` config option (`utils/schemas/config.py#L1497-L1502`, the generated config reference).
  - **A framework with no documented way to set the mode** raises `ModeNotConfigurable`. The verdict is `not_configurable`, with the reason, never a fail, and the summary shows "not configurable" cells.
  - Each declaration records how the mode reached the framework (`serving_mode_setting`).
- **P-0003 stays in the ledger as a harness error** (H-9 in `findings/LEDGER.md`), never counted against a framework.
  - `root_causes.json` marks it `harness error`. The report shows harness-error fails separately and never counts them as framework root causes.
  - `findings/LEDGER.md` keeps the running count of harness errors caught. It is 9, counted from this decision log.
- **Result of the re-run:** TRL's secondary table now has 0 bug fails (35 pass, 3 KD-0001, 6 inconclusive on `reasoning`, 4 n/a). Its primary table is unchanged: 40 pass, 4 KD-0001, 4 n/a.
- The report's fail line now splits bugs from known differences and primary from secondary. The earlier "21 fail verdicts" for TRL counted 7 KD-0001 differences as fails, which read as if TRL's primary table had fails. It had none.
- Evidence: `src/parity/serving.py`, `src/parity/adapters/{trl,axolotl}.py`, `tests/test_serving_mode.py`, `tests/test_root_causes.py`, `docs/evidence/2026-11-24-m3-rerun/`.
- Reversible? yes

## 2026-11-24: Re-run in both modes; P-0003 is ours (secondary rows never encoded in that mode); open decision for the human
- What the re-run shows (`docs/evidence/2026-11-24-m3-rerun/`):
  - **TRL primary (thinking):** 40 pass, 4 KD-0001, 4 not applicable. Unchanged.
  - **TRL secondary (non-thinking):** 21 pass, 3 KD-0001, 6 inconclusive (`reasoning`, our reference refuses), 4 not applicable, and 14 fails, all P-0003.
  - **Axolotl primary:** 27 pass, 13 KD-0002/0003 (P-0001), 4 P-0002, 4 not applicable, 0 inconclusive.
  - **Axolotl secondary:** 34 pass, 4 P-0002, 6 inconclusive (`reasoning`), 4 not applicable.
  - Every fail is linked: TRL has 21 fail verdicts from 2 root causes, Axolotl 21 from 2. 0 untriaged.
- **P-0003 is ours.** The secondary judges the same rows, so TRL was never told the mode and encoded in thinking mode, its default. It trains the empty think block that non-thinking serving pre-fills.
  - TRL labels correctly when given `enable_thinking: false` (per-row `chat_template_kwargs`; documented in `chat_templates.md` L115, tested, and itself the fix for TRL #8071).
  - Skeptic: our error, high confidence. With the column set, all 14 cells pass.
  - The 14 cells are pinned to P-0003 (status "ours: evaluation design") so they count once and never read as a TRL defect.
- **Open for the human:** in secondary runs, give each framework its documented mode setting (preferred by the skeptic and me, about 2 extra minutes). That reverses part of the "same encoded rows, no extra framework runs" decision above. Or keep same rows and label such cells "framework not configured for this mode". No change is made until you decide.
- Also fixed (`001bfbe`): a loss-only `t1.truncation` fail no longer records the truncation mode as "matches no mode".
- Reversible? yes

## 2026-11-24: Added tokens are accepted as structural headers; DeepSeek's history turns with text now compute
- Decision (human rule, applied by evidence):
  - **New guard.** The header and the stop token must each be a single token the tokenizer declares, special or added, and neither token's text may appear in any message's content (every string in every message, tool calls included). The other guards are unchanged.
  - **Rule from the human:** use the structural method for DeepSeek only if it agrees with the cut-out everywhere both compute, and that agreement covers more than one model family. Otherwise keep refusing.
- **Re-validation** (`docs/evidence/2026-11-24-structural-added-tokens/`): every turn of every fixture, on Qwen3, Inkling, Hy4 and DeepSeek, in both serving modes, compared token for token wherever both methods compute.
  - **0 disagreements.** Both compute and agree on 36 turns across 3 families: Qwen3 (3 thinking + 7 non-thinking), Hy4 (11 non-thinking) and DeepSeek (8 non-thinking). Inkling: 0, because its generation prompt adds nothing after the header.
  - **So the rule is met, and the structural method is used for DeepSeek.**
  - **Caveat, stated plainly:** the part that changed (an *added* header, not a special one) is exercised only on DeepSeek, on 8 non-thinking turns where the cut-out also computes, all agreeing. On Qwen3 and Hy4 the headers are special, so the old and new rules are identical there. The structural method as a whole covers three families; the widening covers one. If that isn't enough, reverting is one line (the default `header_ids`).
- **Result on DeepSeek (thinking mode, primary):**
  - `multi_turn` (turns 1 and 3) and `reasoning` (turn 1) now compute: spans `Paris.<eos>`, `Tokyo.<eos>`, `No. 91 = 7 * 13, so it is not prime.<eos>`, with `</think>` prompt side.
  - `empty_assistant` and `tool_call` still refuse: their first assistant turn has no text content to locate.
  - So 2 of DeepSeek's 4 multi-assistant fixtures move from refused to computed. `docs/evidence/golden-review.md` now shows the real reference for `multi_turn`.
- Evidence: `docs/evidence/2026-11-24-structural-added-tokens/results.md`, `tests/test_reference_deepseek.py`.
- Reversible? yes

## 2026-11-24: Token-level false→False plant; "tool arguments" is a declared, behavior-verified framework property
- Decision:
  - **New plant `history_false_as_python_tokens`** (synthetic) on `t1.multiturn_prefix`. In the framework's encoded output, inside every assistant turn that isn't the conversation's last message, tokens spelling `false` are swapped for their single-token `False` counterpart, labels included. The prefix encoding, where that turn is the latest, is untouched. So every framework keeps a guard on the history re-render rule, whatever it does with string arguments. 21 plants now.
  - **The input-level `history_false_as_python` is skipped where the framework declares `parsed_as_json`.** Such a framework refuses a Python repr, so this fault can't arise there, and the planted run only crashed: that is what made Axolotl's `tool_call` cell inconclusive.
  - **`Declaration.tool_arguments`**: `parsed_as_json` or `passed_through`. Neither framework has an option for it, so it is declared from a version-pinned source link:
    - TRL 1.18.2: `passed_through` (`trl/trainer/sft_trainer.py#L1360-L1381`: messages go to `apply_chat_template` unchanged; the installed file is identical to the tag).
    - Axolotl 0.21.1: `parsed_as_json` (`prompt_strategies/chat_template.py#L1402-L1415`: `json.loads`, raises on failure).
  - **Verified by behavior** on fixtures with dict tool arguments. The check also encodes the conversation with every argument given as a compact JSON string (`{"celsius":false}`). Passing it through renders it verbatim; parsing re-serializes it as the dict renders (`{"celsius": false}`). So the two behaviors give different tokens and neither needs an error path. A contradicted declaration makes the verdict inconclusive with `declared_actual_mismatch`.
  - **On the report card**: a "Framework properties" section shows the declared value and the behavior result for each framework.
- Why: the human's decision on the open question from the Axolotl run.
- Evidence: `src/parity/checks/tier1/multiturn_prefix.py`, `src/parity/adapters/base.py`, `tests/test_tool_arguments.py`.
- Reversible? yes.

## 2026-11-24: Shared root causes count once; every fail links to its root cause
- Decision:
  - Every cell keeps its own verdict. Each fail gets `Verdict.root_cause`, and the report card and Gate A count root causes, not cells (`src/parity/engine/root_causes.py`).
  - **Differences** link through their known-differences entry: `finding` names the finding it came from (KD-0002 and KD-0003 → P-0001); an entry from no finding counts as its own root cause (KD-0001).
  - **Bugs** link only by an exact pin in `findings/root_causes.json`: the same framework version, model, check, fixture and serving mode, and the same evidence (sha256 of its canonical JSON). Pins are written after /triage and the skeptic, from the verdicts themselves. No wildcards: if the failure changes in any way, the link breaks and the cell is untriaged again.
  - **Untriaged fails count one each**, so grouping can never hide a fail. Gate A's A4 now also needs zero untriaged fails.
  - The summary gains a "Fails by root cause" section per model, and result rows name their root causes.
  - A fail that a known difference explains only in part keeps the leftover in `known_difference_residual`, to be triaged and linked like any other fail.
- Why: one root cause fails several cells. P-0001 failed `t1.assistant_mask` and `t1.truncation`; P-0002 now fails up to four checks in both modes. "Blocked by t1.template_parity" only covered template parity.
- Evidence: `src/parity/engine/root_causes.py`, `tests/test_root_causes.py`, `docs/gates/gate-A.md`, `.claude/skills/triage/SKILL.md`.
- Reversible? yes.

## 2026-11-24: P-0001 is a known difference (KD-0002, KD-0003), thinking mode only; mask-level signatures
- Decision:
  - **First, the token answer** (re-read from Axolotl's prepared row on `reasoning`, every assistant token printed): only `<think>` (index 51) and the `\n` after it (52) are untrained. The reasoning text (53-91), `</think>`, the `\n\n` after it, the answer and `<|im_end|>` are all trained. On the 7 fixtures without reasoning, the untrained tokens are the empty block `<think>`, `\n\n`, `</think>`, `\n\n`. So it is only think tags and blank lines; no reasoning text is lost.
  - **Registry.** KD-0002 (`t1.assistant_mask`) and KD-0003 (`t1.truncation`): Axolotl 0.21.1, Qwen3's own template on both sides (sha256 `a55ee1b1…`), `serving_modes: ["thinking"]`, signature `think_opener_untrained`. Two entries because an entry names one check.
  - **Pinned to Axolotl's own test**: `tests/prompt_strategies/test_chat_templates_thinking.py#L143-L177` @ v0.21.1 asserts `trained == "greet</think>hello<|im_end|>part</think>see you<|im_end|>"`, i.e. the opener untrained and `</think>` trained. The mechanism is `chat_template.py#L728-L756` @ v0.21.1 (`find_turn` gives the field span; only `labels[turn_start:turn_end]` is set).
  - **Docs: a possible docs gap, not a finding.** Axolotl's user docs (v0.21.1 `docs/`, README, FAQS, Qwen3 examples) never say that think tags go untrained by default. The closest is `conversation.qmd` L445: `generation_tags` "also trains template structure inside them", which implies it without naming think tags. Recorded in the entries' `docs_note`.
  - **Mask-level (zone) signatures.** The registry rule is extended for mask checks: a zone signature moves exactly named reference tokens to the zone the documented behavior gives them, and the framework's labels must pass the check's own rule against those zones with nothing left over. Here it moves only the think opener at the start of a turn; every other token and turn keeps its zone. Token signatures (KD-0001) are unchanged.
  - **Every entry names its serving modes.** KD-0001 covers both, since Qwen3's tokens are the same in both modes. KD-0002/0003 cover thinking only; in non-thinking mode the opener is prompt side, and Axolotl needs no entry there.
  - **Partial explanations are shown, never absorbed.** When an in-scope entry leaves violations over, the verdict stays a bug and its evidence lists what is unexplained (`known_difference_residual`). On `empty_assistant`, KD-0002 explains the think opener, and P-0002's `<|im_end|>` at 14 is left over.
- Why: the human's decision on P-0001. The skeptic found it documented by test and partly our reference's mode assumption; the serving-mode decision settles the second part.
- Evidence: `src/parity/checks/known_differences.json`, `src/parity/checks/known_differences.py`, `tests/test_known_difference_axolotl.py`, `findings/P-0001/record.md`.
- Reversible? yes. Delete the entries and the 15 cells go back to fails.

## 2026-11-24: The serving mode is declared; mode-dependent tokens take their zone from it (from P-0001, after results)
- Decision:
  - **Provenance, stated plainly: this reference change was made after seeing the Axolotl results** (P-0001), not before. P-0001 showed the reference had silently assumed thinking mode for Qwen3. The fix is to make the assumption explicit, not to change which result counts.
  - **No neutral zone.** `TrainCfg.serving_mode` is `thinking` or `non_thinking`. Left unset it means the model's documented default (`src/parity/serving.py`), which is thinking for every Phase 0 model. Every verdict records the mode (`Verdict.serving_mode`), and the reference records the chat-template arguments it rendered with (`reference.template_kwargs`).
  - **Every render uses the mode's arguments**: the full render, the generation prompts, the cut-out, the structural method, the marker cross-check, prefix breaks and template properties. So a mode-dependent token gets its zone from the declared mode.
    - Qwen3 `enable_thinking` true/false; DeepSeek V4.1 `thinking` true/false; Hy4 `reasoning_effort` `high`/`no_think`.
    - Inkling documents no setting that turns thinking off (its `reasoning_effort` is a number in [0, 0.99] written as a system line), so it has no non-thinking mode and the reference refuses there (`ServingModeError`, inconclusive). Not checked against model cards.
  - **Primary and secondary results.** The kill test's primary results stay in thinking mode. Every run is also judged in non-thinking mode as a clearly labeled secondary result: the same encoded rows (adapters drop the mode before the framework sees the config, so caches are shared and no extra framework runs happen), a second reference, and its own table in the summary.
  - Frameworks are not given the mode in Phase 0; they encode with their own default chat-template arguments, as their users do by default.
- What the mode changes on Qwen3 (`tests/test_serving_mode.py`):
  - On 7 fixtures the tokens are identical in both modes. The only change is the last turn's empty think block (`<think>`, `\n\n`, `</think>`, `\n\n`): must-train in thinking mode, prompt side in non-thinking mode, where the generation prompt pre-fills it.
  - History turns keep their zones. In non-thinking mode they lack the generation prompt's pre-fill, so they are anchored by the narrow pre-fill rule, and the "pre-fill missing from history" property appears for Qwen3 in that mode.
  - `reasoning` has no non-thinking reading: its last turn reasons, but the non-thinking generation prompt closes the think block empty. The reference refuses, so those secondary cells are inconclusive.
- Why: P-0001's skeptic showed the reference judged no-reasoning data as if the model wrote the empty think block, which only holds in thinking mode. A neutral zone would hide that choice; a declared mode makes it visible and testable.
- Evidence: `src/parity/serving.py`, `src/parity/references/template.py`, `tests/test_serving_mode.py`, `findings/P-0001/record.md`.
- Reversible? yes. Removing a mode only drops the secondary table.

## 2026-11-24: M3 Axolotl adapter, the run on Qwen3, and two candidate findings
- Decision:
  - **Adapter.** `src/parity/adapters/axolotl.py` drives Axolotl from a YAML config through exactly what `axolotl preprocess` runs (`load_cfg`, then `load_datasets` inside `disable_datasets_caching()`).
    - The config is `type: chat_template`, `roles_to_train: ["assistant"]`, `train_on_eos: turn`, with `chat_template` left at its documented default `tokenizer_default`.
    - Single rows are read from the prepared dataset. Batches go through the collator `setup_trainer` builds, with a tiny `from_config` Qwen3.
    - `declare()` reads the loaded config and the live collator. The unset-means-`drop` over-length rule is declared from a version-pinned code link (`utils/data/utils.py#L347` @ v0.21.1, checked).
  - **Environment.** `envs/axolotl-0.21.1` is pinned to the WSL spike's freeze. Deviations from Axolotl's documented install are listed in `environment.json`: no `[deepspeed]` extra, the default torch backend, CPU only.
    - Axolotl's telemetry is opt-out; the adapter sets `AXOLOTL_DO_NOT_TRACK=1` and `DO_NOT_TRACK=1` before importing it.
    - The core package's lower bounds are loosened to `transformers>=5.10.4` and `huggingface-hub>=1.33.0`, what all five framework environments resolve, so the reference can run inside each. The core lock is unchanged, and the full suite passes inside Axolotl's environment (transformers 5.20.1) too.
  - **The adapter is checked against the real CLI.** `axolotl preprocess config.yml --debug-num-examples 0` wrote rows identical to the adapter's, labels included, on `single_turn` and `empty_assistant`.
- **Predictions first.** `docs/evidence/2026-11-24-m3-axolotl/predictions.md` was committed from Axolotl's docs alone before the adapter existed (`68c315f`).
  - The run matched it on **45 of 48** cells: 26 pass, 17 fail, 4 not applicable, 1 inconclusive (`compare.md`).
  - Not as predicted:
    - `t1.end_of_turn` and `t1.special_tokens` on `empty_assistant` fail. That is P-0002.
    - `t1.multiturn_prefix` on `tool_call` is inconclusive, because a planted fault crashed (below).
- **Triage** (`/triage`; records in `findings/`):
  - **P-0001, the think block** (15 fails: `t1.assistant_mask` ×8, `t1.truncation` ×7). With Qwen3's own template, Axolotl's default labeling never trains the think block that opens the last assistant turn. It labels each turn's located fields only.
    - Partly documented: the `generation_tags` description says the default "locates each turn's fields". But that remedy can't apply to Qwen3's own template, which has no generation markers.
    - Triage leans `difference`; the human decides.
  - **P-0002, the empty turn** (2 fails). The closing `<|im_end|>` of an empty assistant turn is never trained.
    - `find_turn` returns `(-1, -1)` for an empty turn, and the EOS search from −1 never labels it. This contradicts the documented `train_on_eos: turn`. Candidate.
  - The 7 truncation fails duplicate P-0001's root cause. "Blocked by t1.template_parity" doesn't cover a shared root cause between other checks. Design question for the human.
  - **Skeptic review** (verdicts in both records):
    - **P-0001: documented difference.** Axolotl's own test (`tests/prompt_strategies/test_chat_templates_thinking.py:177` @ v0.21.1) asserts the opening `<think>` is untrained and `</think>` trained.
      - Part of it is **our reference's thinking-mode assumption** (rule 4). Qwen3's own template renders a no-reasoning last turn identically in both modes, and in non-thinking mode the empty think block is pre-filled by the generation prompt, so it's prompt side. Axolotl's labels match that.
      - Proposed: a known-differences entry, and a reference decision on mode ambiguity. Not an upstream issue.
    - **P-0002: real candidate, low severity.** The skeptic found the path is completely silent (no warning on the default path; confirmed from the repro's log).
      - An empty *last* turn loses all of its loss.
      - Next step: `/shrink-repro P-0002`.
  - **The inconclusive `tool_call` verdict is our planted fault's limit, not an Axolotl fail.**
    - Axolotl parses string tool arguments as JSON, so `history_false_as_python`'s Python repr raised `JSONDecodeError`, and the planted run ended inconclusive.
    - Axolotl refusing non-JSON argument strings is a guard against exactly that bug class.
    - The fixture's other planted fault (`history_edit`) was caught. Open: a token-level variant, or a declared "arguments are parsed as JSON" property.
- **Also observed, not a Tier 1 finding:** `axolotl preprocess`'s default debug sample calls `random.randrange(0, len(dataset) - 1)`. It crashes on a one-row dataset and never samples the last row (`axolotl/common/datasets.py`, line 36 @ v0.21.1). It's a crash, not a silent training bug, and the adapter avoids it with `--debug-num-examples 0`.
- **What Axolotl's own tests guard** (tests/prompt_strategies @ v0.21.1): `roles_to_train`, `train_on_inputs`, the `train_on_eos`/`train_on_eot` variants, EOT tokens, tool calling, splitting thinking for Qwen3, and generation-marker labeling. The `generation_tags` tests note that markers are needed so "the empty analysis block [is] generated, not just field contents".
  - Not guarded: the default labeling of Qwen3's own template against an independent reference, and empty assistant turns.
- Evidence: `docs/evidence/2026-11-24-m3-axolotl/` (`run.py`, `predictions.md`, `compare.md`, verdicts, `summary.md`), `findings/P-0001/`, `findings/P-0002/`, `tests/test_adapter_axolotl.py` (21 pass in Axolotl's environment).
- Reversible? yes

## 2026-11-24: The kill test is frozen; Phase 2 deferrals; declaring behavior that has no config option
- Decision (human):
  - **The kill test is frozen at 8 fixtures and 432 runs.** Real results exist now, so changing the design would be post-hoc. Two items move to Phase 2 (listed in `docs/phases/phase-2-tier1-breadth.md`, "Deferred from Phase 0"):
    - gpt-oss-20b as a calibration tokenizer for per-turn closing tokens, with the open `<|end|>` question. Likely rule: a closing token is valid if it's a declared stop token or a documented history-form replacement of one, with the doc link.
    - A truncation fixture where modes disagree beyond `long_answer`.
    - No kill-test target reaches either rule, so neither can change Gate A.
  - **Behavior with no config option may be declared from a source link pinned to the exact version** (code or test, e.g. `.../blob/v1.18.2/...#L1502-L1508`), and must be re-verified on every version bump.
    - First case, logged now and implemented when a fixture first hits it: TRL 1.18.2 drops examples left with no trainable token after `keep_start` truncation. Source: `trl/trainer/sft_trainer.py` (`_prepare_dataset`, "Drop examples left fully masked by truncation") and `tests/test_sft_trainer.py::test_fully_masked_examples_dropped_after_truncation`, both at v1.18.2.
    - When implemented, TRL's declaration will carry it as its truncation behavior ("right, then drop a row with nothing to train") instead of letting the truncation check call it a declared/actual mismatch.
- Reversible? yes

## 2026-11-24: Structural delimiting of history turns, with guards; DeepSeek still refuses on the header guard
- Decision (human):
  - Where the cut-out can't delimit an assistant turn, the reference tries a structural method (`structural_span`). It is used only if all of these hold:
    - the turn starts with a header token the tokenizer declares special, and closes with a stop token that is also declared special;
    - the text between the header and the content is exactly what a generation prompt adds after that header. The default prompt is tried first, then the thinking-off one; each prompt's tail is taken against the same mode's render without it. That text must also be identical at every structurally delimited turn;
    - the message's content appears exactly once between the header and the stop token, right after that text.
  - The span runs from the content through the stop token, and whitespace right after the stop token is neutral.
  - **`</think>` in DeepSeek history turns is decided by evidence.** DeepSeek's generation prompt with thinking off ends in `<｜Assistant｜></think>`, the same text its history turns carry after the header. So `</think>` is a pre-fill: prompt side, never trained.
- **Validation** (`docs/evidence/2026-11-24-structural-validation/`): the structural method ran on every turn of every fixture on Qwen3, Inkling, Hy4 and DeepSeek, and was compared token for token with the cut-out wherever both compute.
  - There are **0 disagreements**. But that's thin evidence: both compute on only 3 turns, all Qwen3 history turns. The guards rarely let the structural method compute where the cut-out works:
    - last turns carry a think block before the content;
    - Inkling's generation prompt is empty;
    - Hy4 puts text between header and content that no generation prompt adds.
  - It's wired in under the guard as decided, and changes no result today.
- **Result on DeepSeek: still refuses.** DeepSeek declares `<｜Assistant｜>` as an added token, not a special one (`special: false`), so the header guard fails on every history turn.
  - If added tokens were accepted as headers, `multi_turn` (turns 1 and 3) and `reasoning` (turn 1) would compute, with `</think>` prompt side and spans `Paris.<eos>`, `Tokyo.<eos>`, `No. 91 = 7 * 13, so it is not prime.<eos>`. This is shown in `docs/evidence/golden-review.md`, marked NOT IN USE.
  - `empty_assistant` and `tool_call` would still refuse: their first assistant turn has no text content to locate.
- Open for the human: accept tokenizer-declared added tokens (special or not) as headers for the structural method?
- Evidence: `tests/test_reference_deepseek.py` (header guard, thinking-off evidence, what added-token headers would give), `tests/test_reference_template.py::test_the_structural_method_agrees_with_the_cut_out_where_both_compute`.
- Reversible? yes

## 2026-11-24: M3 TRL adapter, and the first run on a real framework
- Decision:
  - **Adapter.** `src/parity/adapters/trl.py` builds an `SFTTrainer` the way a user would and reads `trainer.train_dataset` through `trainer.data_collator`. It never tokenizes or applies a template itself.
    - The trainer gets a tiny Qwen3 built with `from_config` from the release's config.json, a fresh tokenizer, `SFTConfig(assistant_only_loss=True)`, and a `datasets.Dataset` of `messages` (plus `tools` where present).
    - `TrainCfg.objective` maps as `assistant_only` → `assistant_only_loss=True`, which is TRL's documented recipe (sft_trainer.md, "Train on assistant messages only"). `all` → loss on everything, TRL's documented default.
    - `declare()` reads `trainer.chat_template`, `trainer.args`, `trainer.padding_free`, `trainer.data_collator.pad_token_id` and `trainer.model.loss_function`.
  - **Environment.** `envs/trl-1.18.2` is installed the way TRL's docs say, with every version held at the WSL spike's freeze (`environment.json`: no deviations). Every verdict records the lockfile and its sha256.
- **Run** (`docs/evidence/2026-11-24-m3-trl/`): Qwen3-0.6B, 6 checks × 8 fixtures, through `evaluate_suite`.
  - 40 pass, 4 `not_applicable` (multi-turn prefix on single-assistant fixtures), and 4 `difference` (KD-0001 on exactly `multi_turn`, `tool_call`, `reasoning` and `empty_assistant`).
  - All 131 planted-fault runs caught, no flags.
  - As predicted, so nothing went to `/triage`.
- **Behavior worth knowing (not a finding):**
  - TRL trains the neutral newline after `<|im_end|>`. It's recorded as a convention.
  - The `datasets` round trip, which fills missing message keys with None, changed no token on the fixtures.
  - TRL drops any example left with no trainable token after `keep_start` truncation (in its code and its tests). Our fixtures don't reach that: `long_answer` keeps 105 answer tokens at 128.
  - If a future fixture does, our truncation check would see a drop under a declared right cut. It would call that a declared/actual mismatch (inconclusive) instead of the documented behavior. Settle it before such a fixture is added; probably as a TRL-specific truncation mode ("right, then drop if nothing trains").
- **Adapter tests** (run in TRL's environment, skipped in the core one): `declare()` matches TRL's documented defaults for our config. `encode` and `encode_batch` run on all 8 fixtures, and the rows equal TRL's declared template's render with loss exactly where its generation markers put it.
- Reversible? yes

## 2026-11-24: Narrow pre-fill anchoring in the reference; DeepSeek V4.1 still refuses on history turns
- Decision (human): anchor assistant content where the generation-prompt render and the full render stop agreeing, only when the difference is exactly the generation prompt's trailing pre-fill. Anything else still refuses.
  - The prompt must be the tokens it shares with the full render, followed by the pre-fill's own tokens, token for token.
  - The shared tokens must cover messages[:k] and at least one token of the generation prompt, and end on a token boundary.
  - One guard added while building it: the same generation prompt must appear verbatim at another assistant turn in the full render. That proves the template really emits the pre-fill there; a merely different header (e.g. a capitalized role) is not a pre-fill and still refuses.
  - The span records the missing pre-fill. Verdicts that judge such spans name the model-level property "pre-fill missing from history".
- Result on DeepSeek V4.1:
  - The generation-prompt step now passes on its history turns. But they still refuse at the next step: when an assistant turn is removed, DeepSeek merges the user messages around it ("...France?\n\nAnd of Japan?"), so no cut-out delimits the turn.
  - That is a different rule from the one approved, so it is left refusing. DeepSeek stays `inconclusive` on `multi_turn`, `reasoning`, `empty_assistant` and `tool_call`, and computes on the other four.
  - Its "pre-fill missing from history" property is still recorded, because it is detected from the generation-prompt renders alone (`prefill_missing_turns`).
- Where the anchoring does apply today: Qwen3 in non-thinking mode, whose generation prompt pre-fills an empty think block that history turns lack. Golden: `tests/test_reference_deepseek.py::test_golden_anchored_spans_qwen3_non_thinking`.
- Golden for DeepSeek: `single_turn`. The header `<｜Assistant｜><think>` is prompt side; `</think>Red, yellow and blue.<｜end▁of▁sentence｜>` is must-train; no separator follows the EOS. The template also injects a default system prompt ("Reasoning Effort: 75 ...") when there is none.
- Open for the human:
  - **How to delimit DeepSeek history turns.** For example, end the turn at its stop token and check that the message's content string appears exactly once in between.
  - **Which zone `</think>` belongs to in DeepSeek history turns.** The template emits it together with the `<｜Assistant｜>` header, which suggests it is prompt side. Anchoring where the renders stop agreeing would make it must-train.
- Reversible? yes

## 2026-11-24: Tier 3 note: pre-fill missing from history is a train/serve mismatch
- Note (not a decision, not a finding): a template whose generation prompt ends in a pre-fill that history turns render without (DeepSeek V4.1: `<think>`; Qwen3 non-thinking: an empty think block) trains history turns in a form the served model never sees at the start of its own turns. The Tier 3 serving-parity check must cover it next to the TRL think-block note below (`docs/phases/phase-7-rl-seam.md`).

## 2026-11-24: Engine rules after the skeptic review (flags, blocking, declarations per config)
- Decision (human):
  - **Verdict flags**, shown on the report card even on a pass:
    - `zero_trainable_tokens`: a truncation pass whose kept row has no token with loss, e.g. a cut that left only the prompt. It is still a pass if it matches the reference.
    - `declared_actual_mismatch`: see below.
    - `marker_disagreement`: also a model-level template property on the card, not just reason text.
    - `blocked_by_template_parity`: see below.
  - **One root cause, one finding** (`engine.runner.evaluate_suite`):
    - When `t1.template_parity` fails (kind `bug`) on a fixture, the other checks' fails on that fixture become `inconclusive` with the `blocked_by_template_parity` flag, and their own result stays in the evidence.
    - Their passes stay passes.
    - A matched known difference (KD-0001) blocks nothing.
  - **`declare()` describes the exact config of each encode call.**
    - Truncation reads its mode from the declaration for the cut config.
    - Special tokens reads batching and the pad token from the declaration for the padded config.
    - Each verdict records the declaration for its main call (with the config), plus `other_calls` for any other config it encoded under.
    - Adapters must read every declared value from the framework's live objects, never hardcode it. This is written into `adapters/base.py` and `/new-adapter`.
  - **A packed declaration makes the padding part `inconclusive`**, not `not_applicable`. Phase 0 configs never ask for packing, and packing isolation is a Phase 2 check.
  - **A declared/actual mismatch stays `inconclusive`, but is routed to `/triage`.** It means behavior that exactly matches a different declared value. It may be the adapter misreading the config, or the framework ignoring its own config (a real bug class). Such verdicts carry the flag and `evidence.triage = "declared/actual mismatch"`, and `/triage` now says how to decide.
- Evidence: `tests/test_tier1_suite.py::test_one_root_cause_one_finding`, `tests/test_check_truncation.py` (zero trainable flag, per-config mode, contradiction flag), `tests/test_check_special_tokens.py` (packing, per-call records), `tests/test_report.py::test_marker_disagreements_go_on_the_report_card`.
- Reversible? yes

## 2026-11-24: Harder planted faults; real-world needs a URL and the same mechanism
- Decision (human, after the skeptic review):
  - **Five harder planted faults**, all synthetic:
    - `t1.template_parity`: `rstrip_render`, which drops only the final separator.
    - `t1.assistant_mask`:
      - `train_only_last_turn`;
      - `span_starts_at_header`, which also trains `<|im_start|>assistant\n`;
      - `wrong_label_values`, which changes label values but not which tokens get loss. It guards the label-value rule, which no other planted fault reached on its own.
    - `t1.end_of_turn`: `wrong_closing_token`, which closes each assistant turn with a different stop token, trained.
  - **A `real-world` tag needs a public URL and a matching mechanism.** `Plant` refuses one without a URL. Two faults qualify:
    - `mask_closing_tokens`: https://arxiv.org/pdf/2611.13393.
    - `pad_is_closing_token_masked`: https://github.com/huggingface/transformers/blob/v5.24.0/src/transformers/data/data_collator.py#L798-L800. Lines checked by a read-only `gh api` lookup at the tag.
  - **`history_false_as_python` is now synthetic.** It shares Prime Intellect's symptom, not their mechanism.
  - That makes 20 planted faults, 2 of them real-world.
  - **Planted faults read the declaration for their own config.** `plants()` now gets the adapter's `declare`, so truncation reads the mode under the cut config, and special tokens reads batching under the padded config.
- Evidence: `tests/test_check_*.py` (one test per new fault, each showing the rule that catches it), `tests/test_tier1_suite.py`.
- Reversible? yes

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
