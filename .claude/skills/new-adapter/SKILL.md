---
name: new-adapter
description: Write or update a Parity framework adapter (TRL, Unsloth, Axolotl, verl, SkyRL) that reports exactly what the framework would train on. Use when adding a framework or bumping a framework version.
argument-hint: "[framework-name]"
---

Write the adapter for `$0`. Follow the contract in `docs/plan/02-detailed-blueprint.md` (section "Adapter interface").

Rules:

1. **Call the framework's own public data path**, exactly as a user would: its dataset preparation, collator and trainer entry points. Never re-implement its tokenization, masking, packing or loss.
2. Create a locked environment for this framework version under `envs/` (one uv project or lockfile per framework@version). Record the exact version.
3. Map the framework's own option names onto `TrainCfg.objective` (`assistant_only`, `completion_only`, `all`). Document the mapping at the top of the adapter file, with links to the framework docs you used.
4. Tier 1 needs only `encode()`. If the framework insists on a model object, pass a tiny random-weight model. Never download full weights.
5. Read the framework's own tests for its data path first, and note what they already guard. That shows where its blind spots might be.
6. **`declare(cfg)` describes exactly that config**, and every value comes from the framework's live objects (its config, trainer, collator, tokenizer), never hardcoded in the adapter. `Declaration.sources` names the object and attribute for each value, plus the doc link for the option that maps `TrainCfg.objective`.
7. **The adapter never fixes framework output.** It reads the processed dataset and the collator's batches as they are. The only allowed conversion is turning pre-shifted labels into aligned ones, declared as `label_convention`.
8. **Pin the environment** with a uv lockfile under `envs/<framework>-<version>/`, installed the way the framework's docs say. The adapter exposes `env_lock` and `env_fingerprint` (the lockfile's sha256), and every verdict records both.

Deliverables:

- `src/parity/adapters/$0.py`
- Tests (run inside the framework's environment, skipped elsewhere): on the known-good calibration model (Qwen3 family), `declare()` matches the framework's documented defaults for our config, and `encode()`/`encode_batch()` run on every fixture. The rows' input_ids must match `apply_chat_template` exactly with the template the framework *declares*: a framework may swap in its own training template (TRL does; KD-0001). Comparing against the model's own template is `t1.template_parity`'s job. If they don't match, stop and report: it's either our bug or a finding, and `/triage` decides which.
- A short note in `docs/DECISIONS.md` if anything about the framework's behavior surprised you.
