"""Tiny random-weight models for the data path (rule 6), shared by every adapter.

Tier 1 never needs a model's weights, only a model the framework will accept, so each adapter builds a tiny one with
`from_config`. Which architecture (D1, the human's decision, 2026-11-25):

- **The release's own**, wherever the environment can build it (its transformers, plus anything the framework
  registers itself: Unsloth registers DeepSeek V4.1 at import).
- **Otherwise a stand-in** chosen in advance from the same family (`STAND_INS`), for the path the framework's docs use
  for a text-only chat fine-tune of that release: "text" (a causal LM) or "vision" (an image-text-to-text model). The
  stand-in takes the release's vocabulary size and special-token ids, and every verdict records it as a deviation
  (`TinyModel.record`). Evidence for the candidates: docs/evidence/2026-11-25-m5-env-support/.

Shrinking: each architecture keeps its own invariants, so the sizes come from a profile per model type, applied only
to fields the config has. Qwen3, the calibration model, keeps exactly the settings the adapters used before this
module existed (`QWEN3`), so calibration builds the same model. `quantization_config` is dropped: the weights are
random, and a quantization recipe for a checkpoint that was never loaded means nothing.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, PretrainedConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

PATHS = ("text", "vision")

# The settings every adapter applied before this module (hasattr-guarded), kept byte for byte for Qwen3.
QWEN3 = {"num_hidden_layers": 1, "hidden_size": 16, "intermediate_size": 32, "num_attention_heads": 2,
         "num_key_value_heads": 1, "head_dim": 8}  # fmt: skip

# Text decoders other than Qwen3. Applied only where the config has the field.
TEXT = {
    "num_hidden_layers": 2, "hidden_size": 64, "intermediate_size": 128, "moe_intermediate_size": 32,
    "num_attention_heads": 4, "head_dim": 16,
    "n_routed_experts": 4, "num_experts": 4, "num_local_experts": 4, "num_experts_per_tok": 2,
    "q_lora_rank": 32, "kv_lora_rank": 32, "o_lora_rank": 32, "o_groups": 2,
    "qk_rope_head_dim": 8, "qk_nope_head_dim": 8, "qk_head_dim": 16, "v_head_dim": 16,
    "index_head_dim": 16, "index_n_heads": 2,
    "swa_head_dim": 16, "swa_num_attention_heads": 4, "swa_num_key_value_heads": 2,
    "loop_gate_rank": 4, "num_nextn_predict_layers": 0,
}  # fmt: skip

# Per model type, on top of TEXT: fields that point at layer indices, or must agree with the layer count.
EXTRA: dict[str, dict[str, Any]] = {
    "inkling2_loop_text": {
        "num_hidden_layers": 3, "loop_layer_index_start": 1, "loop_layer_index_end": 2,
        "n_layers_per_attn_res_block": 1,
    },
    "deepseek_v41_text": {
        "num_hidden_layers": 4, "compress_ratios": [0, 0, 2, 1], "kv_source_layer_ids": [2],
        "index_source_layer_ids": [2], "candidate_source_layer_id": 2, "engram_layer_ids": [1],
        "engram_num_embeddings": [4096], "engram_vocab_size": 4096, "engram_n_heads": 2, "engram_head_dim": 16,
        "dspark_target_layer_ids": [1, 2, 3], "dspark_markov_rank": 8, "dspark_n_routed_experts": 4,
        "dspark_num_experts_per_tok": 2,
    },
}  # fmt: skip

# Vision towers (the vision path only). Hidden sizes that feed the text model are set to the text hidden size.
VISION = {"num_hidden_layers": 1, "n_layers": 1, "depth": 1, "hidden_size": 32, "intermediate_size": 64,
          "num_attention_heads": 2, "num_key_value_heads": 2, "projection_hidden_dim": 64}  # fmt: skip
VISION_TO_TEXT = ("out_hidden_size", "text_hidden_size")

# Per-layer lists cut to the new layer count.
LAYER_LISTS = ("layer_types", "mlp_layer_types", "indexer_types")

# D1 stand-ins, fixed before any run (docs/DECISIONS.md, 2026-11-25). None: no candidate for that path.
STAND_INS: dict[str, dict[str, str | None]] = {
    "deepseek_v41": {"text": "deepseek_v4", "vision": None},
    "hy_v4_flash": {"text": "hy_v4", "vision": "hy_v4_vl"},
    "inkling2_loop": {"text": "inkling_text", "vision": "inkling_mm_model"},
}
# Copied from the release's config onto a stand-in (the text part of a composite one).
FROM_RELEASE = ("vocab_size", "bos_token_id", "eos_token_id", "pad_token_id", "tie_word_embeddings")
UNKNOWN_ARCHITECTURE = "does not recognize this architecture"


class NoStandIn(RuntimeError):
    """The environment can't build the release's architecture, and no stand-in is set for this path."""


@dataclass(frozen=True)
class TinyModel:
    model: Any
    record: dict[str, Any]  # what was built and why; every verdict's declaration carries it


def build(local: Path, path: str = "text", seed: int = 0) -> TinyModel:
    """A tiny random-weight model for the release whose tokenizer and config files are in `local`."""
    if path not in PATHS:
        raise ValueError(f"path must be one of {PATHS}, got {path!r}")
    raw = json.loads((Path(local) / "config.json").read_text(encoding="utf-8"))
    release_type = raw.get("model_type")
    stand_in, why = None, None
    try:
        config = AutoConfig.from_pretrained(local)
    except ValueError as exc:
        if UNKNOWN_ARCHITECTURE not in str(exc):
            raise
        stand_in = STAND_INS.get(release_type, {}).get(path)
        if stand_in is None:
            raise NoStandIn(f"{release_type}: this environment can't build it, and no {path} stand-in is set") from exc
        if stand_in not in CONFIG_MAPPING:
            raise NoStandIn(f"{release_type}: its {path} stand-in {stand_in} is not registered in this environment") from exc
        why = str(exc).splitlines()[0]
        config = AutoConfig.for_model(stand_in)
        _copy_release_fields(config, raw)

    if getattr(config, "quantization_config", None) is not None:
        del config.quantization_config
    auto = AutoModelForImageTextToText if path == "vision" else AutoModelForCausalLM
    target = config
    if path == "text" and not _maps(auto, config) and _maps(auto, getattr(config, "text_config", None)):
        target = config.text_config  # e.g. Inkling 2: the causal LM is registered for its text config only
    if not _maps(auto, target):
        # e.g. DeepSeek V4.1 on Unsloth's own implementation: a causal LM only, no image-text class
        raise NoStandIn(f"{release_type}: this environment has no {path} model class for {type(target).__name__}")
    _shrink(target)

    torch.manual_seed(seed)
    model = auto.from_config(target)  # random weights; nothing downloaded
    record = {
        "release_model_type": release_type,
        "built_model_type": target.model_type,
        "class": type(model).__name__,
        "path": path,
        "stand_in": stand_in,
        "deviation": (
            f"stand-in: the model is {type(model).__name__} ({stand_in}), not the release's {release_type}, because "
            f"this environment can't build it ({why})"
            if stand_in
            else None
        ),
        "config_sha256": hashlib.sha256(model.config.to_json_string(use_diff=False).encode()).hexdigest(),
    }
    return TinyModel(model, record)


def save(tiny: TinyModel, local: Path, folder: Path) -> Path:
    """The tiny model in a folder a framework can load by path: its weights and config, plus the release's own
    tokenizer, template and other files (all but the release's config.json)."""
    tiny.model.save_pretrained(folder)
    for item in Path(local).iterdir():
        if item.name != "config.json":
            shutil.copy(item, folder / item.name)
    return folder


def _maps(auto: Any, config: PretrainedConfig | None) -> bool:
    return config is not None and type(config) in auto._model_mapping


def _copy_release_fields(config: PretrainedConfig, raw: dict[str, Any]) -> None:
    text = raw.get("text_config") or {}
    dest = getattr(config, "text_config", None) or config
    for key in FROM_RELEASE:
        value = text.get(key, raw.get(key))
        if value is not None:
            setattr(dest, key, value)
            if dest is not config and key != "vocab_size" and hasattr(config, key):
                setattr(config, key, value)


def _shrink(config: PretrainedConfig) -> None:
    text = getattr(config, "text_config", None)
    vision = getattr(config, "vision_config", None)
    for sub in {id(c): c for c in (config, text) if c is not None}.values():
        _shrink_text(sub)
    if vision is not None and vision is not config:
        hidden = getattr(text if text is not None else config, "hidden_size", None)
        for key, value in VISION.items():
            if hasattr(vision, key):
                setattr(vision, key, value)
        for key in VISION_TO_TEXT:
            if hasattr(vision, key) and hidden is not None:
                setattr(vision, key, hidden)


def _shrink_text(config: PretrainedConfig) -> None:
    if config.model_type == "qwen3":
        settings = QWEN3
    else:
        settings = {**TEXT, **EXTRA.get(config.model_type, {})}
        heads = getattr(config, "num_key_value_heads", None)
        if heads is not None:
            config.num_key_value_heads = 1 if heads == 1 else 2
        shared = getattr(config, "n_shared_experts", None)
        if shared:
            config.n_shared_experts = 1
    for key, value in settings.items():
        if hasattr(config, key):
            setattr(config, key, value)
    layers = getattr(config, "num_hidden_layers", None)
    for key in LAYER_LISTS:
        if getattr(config, key, None) and layers is not None:
            setattr(config, key, list(getattr(config, key))[:layers])
    if getattr(config, "local_layer_ids", None) and layers is not None:
        config.local_layer_ids = [i for i in config.local_layer_ids if i < layers]
