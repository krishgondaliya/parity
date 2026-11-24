"""Axolotl adapter: what Axolotl would train on, from a YAML config through its own preprocessing and collator.

Framework: axolotl==0.21.1 (checked at import). Environment: `envs/axolotl-0.21.1/`, every version held at the
2026-11-23 WSL spike's resolution. The deviations from Axolotl's documented install (no `[deepspeed]` extra, the
default torch backend, CPU only) are listed in `envs/axolotl-0.21.1/environment.json`.

Telemetry: Axolotl sends usage telemetry unless opted out. This module sets `AXOLOTL_DO_NOT_TRACK=1` and
`DO_NOT_TRACK=1` before importing Axolotl, so nothing about this machine or these runs leaves it.

How it works (hard rule 2: Axolotl's public path only)
    For each encode call the adapter writes the conversations to a jsonl file and a YAML config, as a user would,
    then runs exactly what `axolotl preprocess` runs: `load_cfg(config.yml)`, then `load_datasets(cfg=...)` inside
    `disable_datasets_caching()`. `encode` returns the prepared dataset's row as Axolotl wrote it.
    `encode_batch` also builds the trainer the way `axolotl train` does (`setup_trainer`, with a tiny random-weight
    model built with `from_config` from the release's config.json, rule 6) and runs the batch through
    `trainer.data_collator`. The adapter never tokenizes or applies a template itself, and never changes Axolotl's
    output.

Objective mapping (`TrainCfg.objective` -> the dataset entry), from Axolotl's docs
    - "assistant_only" -> `type: chat_template`, `roles_to_train: ["assistant"]`, `train_on_eos: turn`. Both are the
      documented defaults, set explicitly: the docs' legacy example "Training on last message" leaves them empty.
      https://github.com/axolotl-ai-cloud/axolotl/blob/v0.21.1/docs/dataset-formats/conversation.qmd
      `chat_template` stays unset: the documented default is `tokenizer_default`, the model's own template.
    - "all" -> `train_on_inputs: true`.
    - "completion_only" -> not mapped (the fixtures are conversations).

Other settings
    - `max_len` -> `sequence_len`. Axolotl has no "no limit"; `max_len=None` sets it to 8192, above every fixture.
    - Over-length rows follow `excess_length_strategy`, left unset. Axolotl's code treats unset as "drop".
    - `packing` -> `sample_packing`.
    - `pad_to` -> `pad_to_multiple_of` ("Pad each batch to a multiple of this value"). Every Parity row is
      shorter than `pad_to`, so each is padded to it.
    - Settings that don't touch the data, so this runs on a CPU in one process:
      `dataset_num_proc: 1`, `bf16`/`fp16` off, `val_set_size: 0`,
      `shuffle_merged_datasets: false` (keeps rows in order),
      and the `axolotl preprocess` flag `--debug-num-examples 0`.
      The last one is needed because the default debug sample crashes on a one-row dataset:
      `random.randrange(0, len(dataset) - 1)` in `axolotl/common/datasets.py`.

What Axolotl's own tests guard, and what they don't: see `docs/DECISIONS.md` (M3 Axolotl entry).
"""

from __future__ import annotations

import os

os.environ["AXOLOTL_DO_NOT_TRACK"] = "1"  # before any Axolotl import
os.environ["DO_NOT_TRACK"] = "1"

import copy  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import tempfile  # noqa: E402
from collections.abc import Sequence  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import axolotl  # noqa: E402
import datasets  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402
import yaml  # noqa: E402
from axolotl.cli.args import PreprocessCliArgs  # noqa: E402
from axolotl.cli.config import load_cfg  # noqa: E402
from axolotl.common.datasets import load_datasets  # noqa: E402
from axolotl.loaders import load_tokenizer  # noqa: E402
from axolotl.utils.chat_templates import get_chat_template_from_config  # noqa: E402
from axolotl.utils.trainer import disable_datasets_caching, setup_trainer  # noqa: E402
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # noqa: E402

import parity  # noqa: E402
from parity.adapters.base import Batch, Declaration, Dropped, Encoded, TrainCfg  # noqa: E402
from parity.fixtures import Conversation  # noqa: E402
from parity.release import LoadedRelease, fetch_release_files  # noqa: E402

VERSION = "0.21.1"
if axolotl.__version__ != VERSION:
    raise ImportError(f"this adapter is for axolotl=={VERSION}, found axolotl=={axolotl.__version__}")

REPO_ROOT = Path(parity.__file__).resolve().parents[2]
ENV_LOCK = f"envs/axolotl-{VERSION}/uv.lock"
BLOB = f"https://github.com/axolotl-ai-cloud/axolotl/blob/v{VERSION}"
DOCS = {
    "chat_template": f"{BLOB}/docs/dataset-formats/conversation.qmd",
    "schema": f"{BLOB}/src/axolotl/utils/schemas/datasets.py",
    "excess_length": f"{BLOB}/src/axolotl/utils/schemas/config.py",
    "excess_length_default": f"{BLOB}/src/axolotl/utils/data/utils.py#L347",
}
EXCESS_LENGTH_MODES = {"drop": "drop", "truncate": "right"}
NO_LIMIT = 8192
TINY = {"num_hidden_layers": 1, "hidden_size": 16, "intermediate_size": 32, "num_attention_heads": 2,
        "num_key_value_heads": 1, "head_dim": 8}  # fmt: skip
PROBE = [{"role": "user", "content": "Hi."}, {"role": "assistant", "content": "Hello."}]

datasets.disable_progress_bars()
transformers.logging.set_verbosity_error()
for name in ("axolotl", "datasets"):
    logging.getLogger(name).setLevel(logging.ERROR)


class AxolotlAdapter:
    """Axolotl 0.21.1's SFT data path, bound to one model release."""

    name = "axolotl"
    version = VERSION

    def __init__(self, release: LoadedRelease):
        self.release = release
        self.env_lock = ENV_LOCK
        self.env_fingerprint = hashlib.sha256((REPO_ROOT / ENV_LOCK).read_bytes()).hexdigest()
        self._local = fetch_release_files(release.release)  # tokenizer, template and config files only
        self._work = Path(tempfile.mkdtemp(prefix="parity-axolotl-"))
        self._runs = 0
        self._declarations: dict[TrainCfg, Declaration] = {}
        self._encoded: dict[tuple[str, str, TrainCfg], Encoded | Dropped] = {}
        self._batches: dict[tuple[tuple[str, ...], TrainCfg], Batch] = {}

    # The framework, driven the way a user drives it

    def _config(self, rows: list[dict[str, Any]], cfg: TrainCfg, micro_batch_size: int) -> Path:
        """Write the jsonl and the YAML config a user would write; return the config path."""
        if cfg.objective == "completion_only":
            raise NotImplementedError("the fixtures are conversations; completion-only isn't mapped")
        self._runs += 1
        run = self._work / f"run-{self._runs}"
        run.mkdir()
        (run / "data.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        dataset: dict[str, Any] = {"path": str(run / "data.jsonl"), "type": "chat_template"}
        config: dict[str, Any] = {
            "base_model": str(self._local),
            "datasets": [dataset],
            "sequence_len": cfg.max_len if cfg.max_len is not None else NO_LIMIT,
            "dataset_prepared_path": str(run / "prepared"),
            "output_dir": str(run / "out"),
            "shuffle_merged_datasets": False,
            "dataset_num_proc": 1,
            "val_set_size": 0,
            "micro_batch_size": micro_batch_size,
            "gradient_accumulation_steps": 1,
            "num_epochs": 1,
            "learning_rate": 1e-5,
            "bf16": False,
            "fp16": False,
        }
        if cfg.objective == "assistant_only":
            dataset.update({"roles_to_train": ["assistant"], "train_on_eos": "turn"})
        else:  # "all"
            config["train_on_inputs"] = True
        if cfg.packing:
            config["sample_packing"] = True
        if cfg.pad_to is not None:
            config["pad_to_multiple_of"] = cfg.pad_to
        (run / "config.yml").write_text(yaml.safe_dump(config, allow_unicode=True))
        return run / "config.yml"

    def _prepare(self, rows: list[dict[str, Any]], cfg: TrainCfg, micro_batch_size: int = 1):
        """Exactly what `axolotl preprocess` runs. Returns (loaded config, prepared train dataset, total steps)."""
        axolotl_cfg = load_cfg(str(self._config(rows, cfg, micro_batch_size)))
        with disable_datasets_caching():
            meta = load_datasets(cfg=axolotl_cfg, cli_args=PreprocessCliArgs(debug_num_examples=0))
        return axolotl_cfg, meta.train_dataset, meta.total_num_steps

    def _trainer(self, axolotl_cfg: Any, train_dataset: Any, total_num_steps: int):
        """The trainer `axolotl train` would build, with a tiny random-weight model."""
        config = AutoConfig.from_pretrained(self._local)
        for key, value in TINY.items():
            if hasattr(config, key):
                setattr(config, key, value)
        if getattr(config, "layer_types", None):
            config.layer_types = config.layer_types[: config.num_hidden_layers]
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_config(config)  # random weights; nothing downloaded
        tokenizer = load_tokenizer(axolotl_cfg)
        return setup_trainer(axolotl_cfg, train_dataset, None, model, tokenizer, None, total_num_steps)

    @staticmethod
    def _rows(convs: Sequence[Conversation]) -> list[dict[str, Any]]:
        with_tools = any(conv.tools for conv in convs)
        rows = []
        for conv in convs:
            row: dict[str, Any] = {"messages": copy.deepcopy(conv.messages)}
            if with_tools:
                row["tools"] = copy.deepcopy(conv.tools)
            rows.append(row)
        return rows

    # The adapter contract

    def declare(self, cfg: TrainCfg) -> Declaration:
        """Read Axolotl's loaded config and live objects for exactly this config. Nothing here is hardcoded."""
        cfg = cfg.for_framework()  # the serving mode only picks the reference (TrainCfg.serving_mode)
        if cfg not in self._declarations:
            axolotl_cfg, dataset, steps = self._prepare([{"messages": copy.deepcopy(PROBE)}], cfg)
            trainer = self._trainer(axolotl_cfg, dataset, steps)
            self._declarations[cfg] = _declaration(axolotl_cfg, trainer, self._local)
        return self._declarations[cfg]

    def encode(self, conv: Conversation, cfg: TrainCfg) -> Encoded | Dropped:
        cfg = cfg.for_framework()  # the serving mode only picks the reference (TrainCfg.serving_mode)
        key = (conv.id, conv.content_key, cfg)
        if key not in self._encoded:
            axolotl_cfg, dataset, _ = self._prepare(self._rows([conv]), cfg)
            if len(dataset) == 0:
                strategy = axolotl_cfg.excess_length_strategy or "drop (unset)"
                self._encoded[key] = Dropped(
                    f"Axolotl's dataset preparation removed the row (sequence_len={axolotl_cfg.sequence_len}, "
                    f"excess_length_strategy={strategy})"
                )
            else:
                self._encoded[key] = _row(dataset[0])
        return self._encoded[key]

    def encode_batch(self, convs: Sequence[Conversation], cfg: TrainCfg) -> Batch:
        cfg = cfg.for_framework()  # the serving mode only picks the reference (TrainCfg.serving_mode)
        key = (tuple(f"{conv.id}:{conv.content_key}" for conv in convs), cfg)
        if key not in self._batches:
            axolotl_cfg, dataset, steps = self._prepare(self._rows(convs), cfg, micro_batch_size=len(convs))
            if len(dataset) != len(convs):
                raise RuntimeError(f"Axolotl kept {len(dataset)} of {len(convs)} rows; can't line them up")
            trainer = self._trainer(axolotl_cfg, dataset, steps)
            batch = trainer.data_collator([dataset[i] for i in range(len(convs))])
            self._batches[key] = Batch(tuple(_batch_row(batch, i) for i in range(len(convs))))
        return self._batches[key]


def _row(example: dict[str, Any]) -> Encoded:
    """One row of Axolotl's prepared dataset, exactly as it wrote it."""
    ids = tuple(int(t) for t in example["input_ids"])
    labels = tuple(int(t) for t in example["labels"])
    attention = tuple(int(t) for t in example["attention_mask"]) if "attention_mask" in example else (1,) * len(ids)
    positions = tuple(int(t) for t in example["position_ids"]) if "position_ids" in example else None
    return Encoded(ids, labels, attention, positions)


def _batch_row(batch: dict[str, Any], i: int) -> Encoded:
    """One row of the collator's output, exactly as Axolotl's collator produced it."""
    ids = tuple(int(t) for t in batch["input_ids"][i].tolist())
    labels = tuple(int(t) for t in batch["labels"][i].tolist())
    attention = tuple(int(t) for t in batch["attention_mask"][i].tolist())
    positions = tuple(int(t) for t in batch["position_ids"][i].tolist()) if "position_ids" in batch else None
    return Encoded(ids, labels, attention, positions)


def _declaration(axolotl_cfg: Any, trainer: Any, local: Path) -> Declaration:
    dataset = axolotl_cfg.datasets[0]
    tokenizer = trainer.data_collator.tokenizer
    template = get_chat_template_from_config(cfg=axolotl_cfg, ds_cfg=dataset, tokenizer=tokenizer)
    own = AutoTokenizer.from_pretrained(local).chat_template
    choice = dataset.get("chat_template")
    source = "model" if template == own else f"axolotl@{VERSION}:{getattr(choice, 'value', choice)}"

    excess = axolotl_cfg.excess_length_strategy
    effective = excess or "drop"
    if effective not in EXCESS_LENGTH_MODES:
        raise ValueError(f"excess_length_strategy={effective!r} has no truncation mode (it raises)")

    loss = getattr(trainer.model, "loss_function", None)
    if getattr(loss, "__name__", "") != "ForCausalLMLoss":
        raise ValueError(f"can't tell the label convention: model.loss_function is {loss!r}")

    collator = trainer.data_collator
    pad = tokenizer.pad_token_id
    packed = bool(axolotl_cfg.sample_packing)
    return Declaration(
        chat_template=template,
        template_source=source,
        label_convention="aligned",
        truncation_mode=EXCESS_LENGTH_MODES[effective],
        batching="packed" if packed else "padded",
        pad_token_id=pad,
        sources=(
            f"chat_template: get_chat_template_from_config(cfg, datasets[0]) with chat_template={choice!r} "
            f"({DOCS['chat_template']})",
            f"objective: datasets[0].type={dataset.get('type')!r}, roles_to_train={dataset.get('roles_to_train')}, "
            f"train_on_eos={dataset.get('train_on_eos')!r}, train_on_eot={dataset.get('train_on_eot')!r}, "
            f"train_on_inputs={axolotl_cfg.train_on_inputs} ({DOCS['chat_template']}; {DOCS['schema']})",
            "label_convention: Axolotl's trainer passes the dataset labels to the model, and model.loss_function is "
            f"{loss.__name__}, which shifts labels itself",
            f"truncation_mode: sequence_len={axolotl_cfg.sequence_len}, excess_length_strategy={excess!r}; unset is "
            f"treated as 'drop' ({DOCS['excess_length']}; code: {DOCS['excess_length_default']})",
            f"batching: sample_packing={axolotl_cfg.sample_packing}",
            f"pad_token_id: trainer.data_collator.tokenizer.pad_token_id={pad}; collator "
            f"{type(collator).__name__}(pad_to_multiple_of={getattr(collator, 'pad_to_multiple_of', None)}, "
            f"label_pad_token_id={getattr(collator, 'label_pad_token_id', None)}), "
            f"padding_side={tokenizer.padding_side!r}",
        ),
    )
