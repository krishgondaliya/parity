# The DeepSeek V4.1 GPU run's load failure: our tiny checkpoint (harness error H-13)

The DeepSeek-only GPU run of kit `55be6c81deb3` (2026-11-27; its zip is kept byte for byte in `raw/`. `verify_zip` passes the manifest and the rebuilt archive, then refuses it at the status check, "the run didn't finish") stopped after 23 s, in the adapter's first `FastLanguageModel.from_pretrained`, before any check:

> RuntimeError: Unsloth: could not find frozen table model.engram_tables.1 (plane 'weight') in the checkpoint. Refusing to load it into memory; set UNSLOTH_FROZEN_TABLE=gpu for a normal load.

`diagnose.py` (Unsloth environment, CPU) builds and saves the tiny model as the adapter does, compares it with the release's weight index (`model.safetensors.index.json`, names only, downloaded into its own folder), and runs Unsloth's own bind step (`unsloth_zoo.frozen_table_store.bind_frozen_tables`) on the saved checkpoint.

**Before the fix** (`diagnose-before-fix.json`): the same error, on CPU.
- Unsloth builds each engram table as a parameter-free frozen table (`UnslothFrozenTableEmbedding`), bound to the checkpoint's rows after loading. Built from config, it has no rows, so `save_pretrained` wrote nothing for `model.engram_tables.1`. The checkpoint had only `model.layers.1.engram.{k_weight, q_weight, wkv.weight}`.
- The release has `layers.1.engram.embed.weight` and `.scale` (and the same at layer 14): fp8 rows with e8m0 scales, one per 32 channels (`quantization_config`: `weight_block_size` [32, 32], `scale_fmt` ue8m0).
- The shrink also set `engram_head_dim` to 16, which isn't a whole 32-channel block; Unsloth's fp8 table decode needs whole blocks.
- The shrunk config keeps one engram table (layer 1 of 4) where the release has two (layers 1 and 14 of 40). That is by design, and not the cause: table 1 existed in the model; its rows were never saved.

**The fix** (`parity.tiny`): rows for every frozen table, in the release's form (fp8 e4m3 rows, e8m0 scales), under the release's plane names; `engram_head_dim` 32.

**After the fix** (`diagnose.json`):
- the checkpoint has `model.layers.1.engram.embed.weight` (F8_E4M3, [4096, 32]) and `.scale` (F8_E8M0, [4096, 1]): the release's names and dtypes under the checkpoint's `model.` prefix;
- against the release's index: no name differs on layer 1 or outside the layers (expert indices collapsed; the release's mtp, vision, aligner and image tensors aren't part of the text model; the release's other `.scale` tensors belong to fp8/fp4 linears, which a bf16 tiny model doesn't have);
- `bind_frozen_tables` finds every table (`model.engram_tables.1`), and a lookup through Unsloth's own fp8 decode equals the rows written;
- the adapter's whole path, `FastLanguageModel.from_pretrained` through an encode of `single_turn`, runs on the CPU backend.

Not used: `UNSLOTH_FROZEN_TABLE=gpu`, Unsloth's non-default loading path. With our checkpoint right, the default path loads it.
