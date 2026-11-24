"""TRL adapter: what TRL's SFTTrainer would train on, read from its own processed dataset and collator.

Framework: trl==1.18.2 (checked at import). Environment: `envs/trl-1.18.2/`, a uv project installed the way TRL's
docs say (`pip install trl`, https://github.com/huggingface/trl/blob/v1.18.2/docs/source/installation.md), every
transitive version held at the 2026-11-23 WSL spike's resolution. Deviations: none (`envs/trl-1.18.2/environment.json`).

How it works (hard rule 2: TRL's public path only)
    For each encode call the adapter builds an `SFTTrainer` the way a user would: a tiny random-weight model built
    with `from_config` from the release's own config.json (no weights downloaded, rule 6), the release's tokenizer
    loaded fresh, an `SFTConfig`, and a `datasets.Dataset` of conversations. Then it reads what TRL produced:
    rows of `trainer.train_dataset`, passed through `trainer.data_collator`. It never calls apply_chat_template or
    tokenizes anything itself, and never changes what TRL returns.

Objective mapping (`TrainCfg.objective` -> `SFTConfig`), from TRL's docs
    - "assistant_only" -> `assistant_only_loss=True` with a conversational dataset (a "messages" column).
      https://github.com/huggingface/trl/blob/v1.18.2/docs/source/sft_trainer.md#train-on-assistant-messages-only
      For known families TRL then swaps in its own training chat template (Qwen3: `qwen3_training.jinja`).
    - "all" -> `assistant_only_loss=False`, `completion_only_loss=False`: loss on the whole sequence, TRL's default
      for conversational language modeling. That default is documented behavior, not a bug.
    - "completion_only" -> not mapped: TRL supports it only for prompt-completion datasets, and the fixtures are
      conversations.

Other settings
    - `max_len` -> `SFTConfig.max_length` (None disables truncation, as documented).
    - `packing` -> `SFTConfig.packing`.
    - `pad_to` -> `SFTConfig.pad_to_multiple_of`. Every Parity row is shorter than `pad_to`, so each is padded to it.
    - Settings that only affect training, not the data path, are set so construction works on a CPU:
      `use_cpu=True`, `bf16=False`, `report_to="none"`, `output_dir` in a temporary directory.

Labels: TRL's `compute_loss` passes the dataset's labels to the model unchanged, and the model's loss function
(transformers' `ForCausalLMLoss`) shifts them itself, so they are aligned with input ids. `declare()` reads this from
`trainer.model.loss_function`; nothing is converted.

What TRL's own tests guard (tests/test_sft_trainer.py @ v1.18.2), and what they don't
    - Guarded: labels are input_ids with non-assistant tokens masked, some trained and some masked
      (`test_dataset_preparation_builds_labels_for_assistant_only_loss`); fully masked examples are dropped after
      truncation (`test_fully_masked_examples_dropped_after_truncation`); truncation to max_length; padding and
      pad_to_multiple_of in the collator; tool-call data trains (`test_train_toolcall_data`).
    - Not guarded: which tokens are trained, judged against an independent reference; whether the rendered tokens
      match the model's own template; what the datasets/Arrow round trip does to messages and tools. Those are what
      Parity's Tier 1 checks look at.
"""

from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import logging
import tempfile
from collections.abc import Sequence
from importlib.resources import files
from pathlib import Path
from typing import Any

import datasets
import torch
import transformers
import trl
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

import parity
from parity.adapters.base import Batch, Declaration, Dropped, Encoded, TrainCfg
from parity.serving import framework_mode, template_kwargs
from parity.fixtures import Conversation
from parity.release import LoadedRelease, fetch_release_files

VERSION = "1.18.2"
if trl.__version__ != VERSION:
    raise ImportError(f"this adapter is for trl=={VERSION}, found trl=={trl.__version__}")

REPO_ROOT = Path(parity.__file__).resolve().parents[2]
ENV_LOCK = f"envs/trl-{VERSION}/uv.lock"
DOCS = {
    "assistant_only_loss": (
        "https://github.com/huggingface/trl/blob/v1.18.2/docs/source/sft_trainer.md#train-on-assistant-messages-only"
    ),
    "sft_config": "https://github.com/huggingface/trl/blob/v1.18.2/trl/trainer/sft_config.py",
    "tool_arguments": "https://github.com/huggingface/trl/blob/v1.18.2/trl/trainer/sft_trainer.py#L1360-L1381",
    "chat_template_kwargs": (
        "citation level: docstring only (trl.apply_chat_template has no API-reference entry at v1.18.2; it appears "
        "only in a usage example in dataset_formats.md, which doesn't mention the key). "
        "Per-example key documented in trl.apply_chat_template's docstring: "
        "https://github.com/huggingface/trl/blob/v1.18.2/trl/data_utils.py#L325-L327; read by SFTTrainer: "
        "https://github.com/huggingface/trl/blob/v1.18.2/trl/trainer/sft_trainer.py#L1363-L1367; tested: "
        "https://github.com/huggingface/trl/blob/v1.18.2/tests/test_sft_trainer.py#L1366"
    ),
}
TRUNCATION_MODES = {"keep_start": "right", "keep_end": "left"}
TINY = {"num_hidden_layers": 1, "hidden_size": 16, "intermediate_size": 32, "num_attention_heads": 2,
        "num_key_value_heads": 1, "head_dim": 8}  # fmt: skip
PROBE = [{"role": "user", "content": "Hi."}, {"role": "assistant", "content": "Hello."}]

datasets.disable_progress_bars()
transformers.logging.set_verbosity_error()
logging.getLogger("trl").setLevel(logging.ERROR)


class TRLAdapter:
    """TRL 1.18.2's SFT data path, bound to one model release."""

    name = "trl"
    version = VERSION

    def __init__(self, release: LoadedRelease):
        self.release = release
        self.env_lock = ENV_LOCK
        self.env_fingerprint = hashlib.sha256((REPO_ROOT / ENV_LOCK).read_bytes()).hexdigest()
        self._local = fetch_release_files(release.release)  # tokenizer, template and config files only
        self._declarations: dict[TrainCfg, Declaration] = {}
        self._encoded: dict[tuple[str, str, TrainCfg], Encoded | Dropped] = {}
        self._batches: dict[tuple[tuple[str, ...], TrainCfg], Batch] = {}

    # The framework, built the way a user would build it

    def _trainer(self, rows: list[dict[str, Any]], cfg: TrainCfg) -> SFTTrainer:
        if cfg.objective == "completion_only":
            raise NotImplementedError("TRL's completion-only loss needs a prompt-completion dataset")
        config = AutoConfig.from_pretrained(self._local)
        for key, value in TINY.items():
            if hasattr(config, key):
                setattr(config, key, value)
        if getattr(config, "layer_types", None):
            config.layer_types = config.layer_types[: config.num_hidden_layers]
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_config(config)  # random weights; nothing downloaded
        tokenizer = AutoTokenizer.from_pretrained(self._local)  # fresh: SFTTrainer sets its pad token
        kwargs: dict[str, Any] = {
            "output_dir": tempfile.mkdtemp(prefix="parity-trl-"),
            "report_to": "none",
            "use_cpu": True,
            "bf16": False,
            "assistant_only_loss": cfg.objective == "assistant_only",
            "max_length": cfg.max_len,
            "packing": cfg.packing,
        }
        if cfg.objective == "all":
            kwargs["completion_only_loss"] = False
        if cfg.pad_to is not None:
            kwargs["pad_to_multiple_of"] = cfg.pad_to
        return SFTTrainer(
            model=model,
            args=SFTConfig(**kwargs),
            train_dataset=datasets.Dataset.from_list(rows),
            processing_class=tokenizer,
        )

    def _mode_setting(self, cfg: TrainCfg) -> str:
        if not cfg.serving_mode:
            return "framework defaults (the model's default mode)"
        kwargs = template_kwargs(self.release, cfg.serving_mode)
        return f"chat_template_kwargs={kwargs!r} on every dataset row ({DOCS['chat_template_kwargs']})"

    def _rows(self, convs: Sequence[Conversation], cfg: TrainCfg) -> list[dict[str, Any]]:
        """The dataset rows a user would write. In a non-default serving mode, every row carries TRL's documented
        per-example `chat_template_kwargs` (DOCS["chat_template_kwargs"]); TRL fails on mixed rows, so all or none."""
        with_tools = any(conv.tools for conv in convs)
        mode_kwargs = template_kwargs(self.release, cfg.serving_mode) if cfg.serving_mode else None
        rows = []
        for conv in convs:
            row: dict[str, Any] = {"messages": copy.deepcopy(conv.messages)}
            if with_tools:
                row["tools"] = copy.deepcopy(conv.tools)
            if mode_kwargs is not None:
                row["chat_template_kwargs"] = dict(mode_kwargs)
            rows.append(row)
        return rows

    # The adapter contract

    def declare(self, cfg: TrainCfg) -> Declaration:
        """Read TRL's live objects for exactly this config. Nothing here is hardcoded."""
        cfg = replace(cfg, serving_mode=framework_mode(self.release, cfg.serving_mode))  # None in the default mode
        if cfg not in self._declarations:
            probe = Conversation(id="probe", messages=copy.deepcopy(PROBE), tools=None)
            trainer = self._trainer(self._rows([probe], cfg), cfg)
            self._declarations[cfg] = _declaration(trainer, self._mode_setting(cfg))
        return self._declarations[cfg]

    def encode(self, conv: Conversation, cfg: TrainCfg) -> Encoded | Dropped:
        cfg = replace(cfg, serving_mode=framework_mode(self.release, cfg.serving_mode))  # None in the default mode
        key = (conv.id, conv.content_key, cfg)
        if key not in self._encoded:
            trainer = self._trainer(self._rows([conv], cfg), cfg)
            if len(trainer.train_dataset) == 0:
                self._encoded[key] = Dropped(
                    "SFTTrainer dropped the example while preparing the dataset (TRL drops examples with no "
                    "trainable token after truncation)"
                )
            else:
                batch = trainer.data_collator([trainer.train_dataset[0]])
                self._encoded[key] = _row(batch, 0)
        return self._encoded[key]

    def encode_batch(self, convs: Sequence[Conversation], cfg: TrainCfg) -> Batch:
        cfg = replace(cfg, serving_mode=framework_mode(self.release, cfg.serving_mode))  # None in the default mode
        key = (tuple(f"{conv.id}:{conv.content_key}" for conv in convs), cfg)
        if key not in self._batches:
            trainer = self._trainer(self._rows(convs, cfg), cfg)
            if len(trainer.train_dataset) != len(convs):
                raise RuntimeError(
                    f"SFTTrainer kept {len(trainer.train_dataset)} of {len(convs)} examples; can't line rows up"
                )
            batch = trainer.data_collator([trainer.train_dataset[i] for i in range(len(convs))])
            self._batches[key] = Batch(tuple(_row(batch, i) for i in range(len(convs))))
        return self._batches[key]


def _row(batch: dict[str, Any], i: int) -> Encoded:
    """One row of the collator's output, exactly as TRL produced it."""
    ids = tuple(int(t) for t in batch["input_ids"][i].tolist())
    labels = tuple(int(t) for t in batch["labels"][i].tolist())
    if "attention_mask" in batch:
        attention = tuple(int(t) for t in batch["attention_mask"][i].tolist())
    else:  # padding-free batches carry no attention mask
        attention = (1,) * len(ids)
    positions = tuple(int(t) for t in batch["position_ids"][i].tolist()) if "position_ids" in batch else None
    return Encoded(ids, labels, attention, positions)


def _declaration(trainer: SFTTrainer, mode_setting: str) -> Declaration:
    args = trainer.args
    if trainer.chat_template is not None:
        template = trainer.chat_template
        where = "SFTTrainer.chat_template (TRL's training template, set because assistant_only_loss=True)"
    else:
        template = trainer.processing_class.chat_template
        where = "SFTTrainer.processing_class.chat_template (the model's own; TRL swapped nothing in)"
    source = _template_source(template, trainer)

    if args.truncation_mode not in TRUNCATION_MODES:
        raise ValueError(f"unknown TRL truncation_mode {args.truncation_mode!r}")
    truncation = TRUNCATION_MODES[args.truncation_mode]

    loss = getattr(trainer.model, "loss_function", None)
    if getattr(loss, "__name__", "") != "ForCausalLMLoss":
        raise ValueError(f"can't tell the label convention: model.loss_function is {loss!r}")

    packed = bool(trainer.padding_free or args.packing)
    pad = trainer.data_collator.pad_token_id
    return Declaration(
        chat_template=template,
        template_source=source,
        label_convention="aligned",
        truncation_mode=truncation,
        batching="packed" if packed else "padded",
        pad_token_id=pad,
        tool_arguments="passed_through",
        serving_mode_setting=mode_setting,
        sources=(
            f"chat_template: {where}; template_source: matched against trl/chat_templates/*.jinja",
            f"objective: SFTConfig.assistant_only_loss={args.assistant_only_loss} ({DOCS['assistant_only_loss']})",
            "label_convention: SFTTrainer.compute_loss passes the dataset labels to the model unchanged and "
            f"model.loss_function is {loss.__name__}, which shifts labels itself",
            f"truncation_mode: SFTConfig.truncation_mode={args.truncation_mode!r}, max_length={args.max_length} "
            f"({DOCS['sft_config']}); TRL also drops examples left with no trainable token after truncation",
            f"batching: SFTTrainer.padding_free={trainer.padding_free}, SFTConfig.packing={args.packing}",
            f"pad_token_id: SFTTrainer.data_collator.pad_token_id={pad}; "
            f"pad_to_multiple_of={args.pad_to_multiple_of!r}",
            f"tool_arguments: no option; tokenize_fn passes the messages to apply_chat_template unchanged, so a "
            f"string is rendered verbatim ({DOCS['tool_arguments']})",
        ),
    )


def _template_source(template: str, trainer: SFTTrainer) -> str:
    """'model' for the model's own template, else the name of the TRL template file it matches."""
    own = AutoTokenizer.from_pretrained(trainer.processing_class.name_or_path).chat_template
    if template == own:
        return "model"
    for path in files("trl").joinpath("chat_templates").iterdir():
        if path.name.endswith(".jinja") and path.read_text(encoding="utf-8") == template:
            return f"trl@{VERSION}:{path.name}"
    return f"trl@{VERSION}:unknown"
