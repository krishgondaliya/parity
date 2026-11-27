"""Why the DeepSeek V4.1 GPU run (kit 55be6c81deb3, 2026-11-27) stopped in FastLanguageModel.from_pretrained:
"could not find frozen table model.engram_tables.1 (plane 'weight') in the checkpoint".

Run from the repo root, in the Unsloth environment (CPU is enough; nothing here trains):
    envs/unsloth-2026.11.10/.venv/bin/python docs/evidence/2026-11-27-unsloth-deepseek-engram/diagnose.py

Rule 4 first: our tiny checkpoint is the suspect. This builds and saves the tiny model exactly as the Unsloth adapter
does (parity.tiny build + save, after importing the adapter module, which imports Unsloth), then:
  1. lists every engram tensor in the saved checkpoint, and the type and store of each engram table in the model;
  2. reads the release's weight index (model.safetensors.index.json: tensor names and file names, no weights),
     downloaded into its own folder (not the release cache, which holds tokenizer, template and config files
     only), and lists the release's engram tensors, with the release config's engram fields;
  3. compares our checkpoint's names with the release's own (see compare_names): on layer 1, an engram layer in
     both, and outside the layers. The release has 40 layers and engram tables at layers 1 and 14; the tiny model has
     4 layers and one at layer 1, by design;
  4. calls Unsloth's own bind step, `unsloth_zoo.frozen_table_store.bind_frozen_tables`, on the saved checkpoint,
     with a model built from the saved config, as from_pretrained does, and reports what it finds or raises; if it
     binds, looks up rows through the bound table (Unsloth's own fp8 decode) and compares them with the rows written.
Run before the factory fix (2026-11-27): bind raised the Colab error. Writes diagnose.json next to this file.
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENGRAM = re.compile(r"engram")


def checkpoint_header(folder: Path) -> dict[str, dict]:
    from unsloth_zoo.frozen_table_store import read_safetensors_header

    out = {}
    for path in sorted(folder.glob("*.safetensors")):
        header, _ = read_safetensors_header(str(path))
        out.update({k: v for k, v in header.items() if k != "__metadata__"})
    return out


def normalize(name: str) -> str:
    """One name per kind of tensor: the expert index collapsed (the tiny model has 4 experts, the release more)."""
    return re.sub(r"\.experts\.\d+\.", ".experts.<e>.", name)


def compare_names(tiny: set[str], release: set[str]) -> dict:
    """Our checkpoint's names without their `model.` prefix against the release's own names. The save writes the
    release's naming (attn.wkv, ffn.experts.<e>.w1, engram.embed); the release also stores a `.scale` next to each
    fp8/fp4 tensor, which a bf16 tiny model has no use for, so those are listed apart."""
    ours_all = {normalize(n[len("model."):] if n.startswith("model.") else n) for n in tiny}
    theirs_all = {normalize(n) for n in release}
    our_scales = {n for n in ours_all if n.endswith(".scale")}
    scales = {n for n in theirs_all if n.endswith(".scale")}
    ours, theirs = ours_all - our_scales, theirs_all - scales

    def layer(names, i):
        return {n.split(f"layers.{i}.", 1)[1] for n in names if n.startswith(f"layers.{i}.")}

    def outside(names):
        return {n for n in names if not n.startswith("layers.")}

    return {
        "layer_1_only_in_release": sorted(layer(theirs, 1) - layer(ours, 1)),
        "layer_1_only_in_tiny": sorted(layer(ours, 1) - layer(theirs, 1)),
        "layer_1_scales_in_tiny": sorted(layer(our_scales, 1)),
        "layer_1_scales_in_release": sorted(layer(scales, 1)),
        "layer_1_scales_only_in_tiny": sorted(layer(our_scales, 1) - layer(scales, 1)),
        "outside_layers_only_in_release": sorted(n for n in outside(theirs) if not n.startswith(("mtp.", "vision.", "aligner.", "image_")) and n not in outside(ours)),
        "outside_layers_only_in_tiny": sorted(outside(ours) - outside(theirs)),
        "release_outside_layers_not_in_the_text_model": sorted({n.split(".")[0] for n in outside(theirs) if n.startswith(("mtp.", "vision.", "aligner.", "image_"))}),
    }


def tables_of(model) -> dict:
    from unsloth.models.deepseek_v41.modeling import is_parameter_free_table

    out = {}
    for name, module in model.named_modules():
        tables = getattr(module, "engram_tables", None)
        if tables is None:
            continue
        for key, table in tables.items():
            store = getattr(table, "frozen_table_store", None)
            out[f"{name + '.' if name else ''}engram_tables.{key}"] = {
                "class": type(table).__name__,
                "parameter_free": is_parameter_free_table(table),
                "store": type(store).__name__ if store is not None else None,
                "parameters": [n for n, _ in table.named_parameters(recurse=False)],
            }
    return out


def release_index(spec: str) -> dict:
    from huggingface_hub import hf_hub_download

    repo, revision = spec.split("@")
    folder = Path(tempfile.mkdtemp(prefix="parity-ds-index-"))
    path = hf_hub_download(repo, "model.safetensors.index.json", revision=revision, local_dir=folder)
    return json.loads(Path(path).read_text(encoding="utf-8"))


def bind(folder: Path, written: dict) -> dict:
    """Unsloth's bind step on the saved checkpoint, with a model built from the saved config (as from_pretrained
    does before it binds); then a lookup through each bound table against the rows written."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    from unsloth_zoo.frozen_table_store import bind_frozen_tables

    model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(folder))
    try:
        bound = bind_frozen_tables(model, str(folder), mode="disk")
    except Exception as exc:  # the result to record
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    modules = dict(model.named_modules())
    lookups = {}
    for name in sorted(bound):
        table = modules[name]
        ids = torch.tensor([[0, 7, table.num_embeddings - 1]])
        got = table(ids).float()
        values, scales = written[f"{name}.weight"], written[f"{name}.scale"].float()
        block = int(getattr(table, "block_size", 32))
        expected = (values.float().unflatten(-1, (-1, block)) * scales.unsqueeze(-1)).flatten(-2)[ids[0]]
        lookups[name] = {"shape": list(got.shape), "equals_rows_written": bool(torch.equal(got.reshape(expected.shape), expected))}
    return {"ok": True, "bound": sorted(bound), "lookups": lookups}


def main() -> None:
    import parity.adapters.unsloth  # noqa: F401  (sets Unsloth's environment and imports it, as the adapter does)
    from parity.release import Release, fetch_release_files
    from parity.targets import PHASE0_TARGETS
    from parity.tiny import build, save

    pin = next(t for t in PHASE0_TARGETS if t.repo_id.startswith("deepseek-ai/"))
    source = fetch_release_files(Release.parse(pin.spec))
    tiny = build(source, "text")
    folder = save(tiny, source, Path(tempfile.mkdtemp(prefix="parity-ds-tiny-")) / "model")
    saved_config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    release_config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    index = release_index(pin.spec)
    release_names = sorted(index["weight_map"])
    header = checkpoint_header(folder)
    out = {
        "model_build": tiny.record,
        "tiny": {
            "engram_config": {k: saved_config.get(k) for k in ("engram_layer_ids", "engram_num_embeddings", "num_hidden_layers")},
            "engram_tensors_saved": {k: [v["dtype"], v["shape"]] for k, v in header.items() if ENGRAM.search(k)},
            "frozen_tables_recorded": tiny.record.get("frozen_tables"),
            "engram_tables_in_model": tables_of(tiny.model),
        },
        "release": {
            "engram_config": {k: release_config.get(k) for k in ("engram_layer_ids", "engram_num_embeddings", "num_hidden_layers")},
            "tensor_names": len(release_names),
            "engram_tensors": [k for k in release_names if ENGRAM.search(k)],
            "name_prefixes": sorted({k.split(".")[0] for k in release_names}),
        },
        "names_against_the_release": compare_names(set(header), set(release_names)),
        "bind_on_saved_checkpoint": bind(folder, tiny.frozen_table_rows),
    }
    (HERE / "diagnose.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=1, ensure_ascii=False)[:4000])


if __name__ == "__main__":
    sys.exit(main())
