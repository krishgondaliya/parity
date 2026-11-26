"""Unsloth adapter: what Unsloth's Qwen3 recipe would train on, read from its trainer's dataset and collator.

Framework: unsloth==2026.11.10 (checked at import), with the TRL it runs (1.15.0). Environment:
`envs/unsloth-2026.11.10/`, every transitive version held at the 2026-11-23 WSL spike's resolution. Deviations from
Unsloth's documented install, including the CPU fallback, are in `envs/unsloth-2026.11.10/environment.json`.

Backend: Unsloth picks one at import, its GPU route when it finds an accelerator, otherwise its CPU fallback
(`unsloth/_cpu_init.py`: stock transformers + PEFT + TRL). The adapter reads which one Unsloth took
(`unsloth.DEVICE_TYPE`) and every verdict records it (`Verdict.backend`: "gpu" or "cpu"). The two routes prepare
datasets differently (P-0004), so their results are never merged. GPU runs use the Colab kit (`docs/gpu-run/`),
with the lockfile its CUDA driver allows: this environment's own, or the CUDA 12.6 variant in `cu126/`.

The recipe (hard rule 2: Unsloth's public path, exactly as its notebooks run it)
    Unsloth's Qwen3 page recommends the "Qwen3 (14B) Reasoning + Conversational" notebook (DOCS["notebook"]):
    1. `FastLanguageModel.from_pretrained(max_seq_length=2048, load_in_4bit=True, ...)`, then `get_peft_model(...)`.
       The adapter points it at a local folder holding a tiny `from_config` Qwen3 (random weights) next to the
       release's own tokenizer, template and config files, so the loader's tokenizer handling applies and no weights
       are downloaded (rule 6).
    2. Format each conversation with the returned tokenizer: `tokenizer.apply_chat_template(messages, tools=...,
       tokenize=False)`, as the notebooks do (tools as in DOCS["tool_notebook"]).
    3. `SFTTrainer(model, tokenizer=tokenizer, train_dataset=<"text" column>, args=SFTConfig(dataset_text_field=
       "text", <the notebook's settings>))`.
    4. Assistant-only: `train_on_responses_only(trainer, instruction_part="<|im_start|>user\\n",
       response_part="<|im_start|>assistant\\n")`, from the Qwen3-4B-Thinking-2507 notebook (DOCS["assistant_only"]).
       No notebook for the original Qwen3 models has an assistant-only step; the choice is logged in
       docs/DECISIONS.md (2026-11-24).
    Then the adapter puts the model in training mode the way Unsloth's train() does (`model.for_training`), and reads
    what Unsloth produced: rows of `trainer.train_dataset`, passed through `trainer.data_collator`. Formatting with the tokenizer is the recipe's own user code; the adapter never
    tokenizes, masks or pads anything itself, and never changes what Unsloth returns.

Objective mapping (`TrainCfg.objective`)
    - "assistant_only" -> step 4, `train_on_responses_only` with the explicit Qwen ChatML parts.
    - "all" -> steps 1-3 without step 4: the recommended notebook's own behavior (loss on the whole text).
    - "completion_only" -> not mapped: the fixtures are conversations.

Other settings
    - `max_len` -> `max_seq_length` in the loader (Unsloth's documented context-length knob) and `SFTConfig.max_length`
      (what TRL truncates by; on CPU the loader's value doesn't reach it). None: the notebook's 2048, and TRL's
      default `max_length`.
    - `packing` -> `SFTConfig.packing`. `pad_to` -> `SFTConfig.pad_to_multiple_of`.
    - Serving mode: any non-default mode raises ModeNotConfigurable, so those cells are not_configurable.
      - Unsloth documents non-thinking training only as the `/no_think` soft switch. It documents no way to train
        for the hard switch our reference models (`enable_thinking=False`).
      - `parity.adapters.unsloth_docs` has the text and the sources (the human's decision, 2026-11-25, P-0006).
      - Until then the adapter passed `enable_thinking=False` to the formatting call; on training text that changes
        nothing.
    - Training-only settings are the notebook's (`optim="adamw_8bit"`, ...); no training step ever runs.

Labels: aligned. `declare()` reads the model's loss function and refuses to guess if it isn't one that shifts labels.

What Unsloth's own tests guard (unsloth-zoo tests/test_responses_only_*.py, test_train_on_responses_*.py)
    - Guarded: response spans end at message boundaries, tool results stay masked, newline-merge edges at the
      markers, last-response-only, fully masked rows dropped by position, packed rows.
    - Not guarded: empty assistant turns, think blocks, truncation, padding against an independent reference, and
      whether the trained tokens equal the model's own template render (TRL appends an EOS to "text" rows).
"""

from __future__ import annotations

import os

os.environ.setdefault("UNSLOTH_DISABLE_STATISTICS", "1")  # Unsloth's documented opt-out; set before the import

import unsloth  # noqa: E402,I001  (Unsloth must be imported before transformers and TRL, as its notebooks do)
from unsloth import FastLanguageModel  # noqa: E402
from unsloth.chat_templates import train_on_responses_only  # noqa: E402

import copy  # noqa: E402
import hashlib  # noqa: E402
import logging  # noqa: E402
import tempfile  # noqa: E402
from collections.abc import Sequence  # noqa: E402
from dataclasses import dataclass, replace  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import datasets  # noqa: E402
import transformers  # noqa: E402
import trl  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402

import parity  # noqa: E402
from parity.adapters.base import Batch, Declaration, Dropped, Encoded, ModeNotConfigurable, TrainCfg  # noqa: E402
from parity.adapters.unsloth_docs import NOT_CONFIGURABLE, NOT_CONFIGURABLE_UNDOCUMENTED  # noqa: E402
from parity.fixtures import Conversation  # noqa: E402
from parity.release import LoadedRelease, fetch_release_files  # noqa: E402
from parity.serving import framework_mode  # noqa: E402
from parity.tiny import build as build_tiny, save as save_tiny  # noqa: E402

VERSION = "2026.11.10"
if unsloth.__version__ != VERSION:
    raise ImportError(f"this adapter is pinned to unsloth=={VERSION}, found {unsloth.__version__}")

DEVICE_TYPE = str(unsloth.DEVICE_TYPE)  # "cpu" on the CPU fallback; "cuda" (or another accelerator) on the GPU route
BACKEND = "cpu" if DEVICE_TYPE == "cpu" else "gpu"

REPO_ROOT = Path(parity.__file__).resolve().parents[2]
ENV_DIR = f"envs/unsloth-{VERSION}"
ENV_LOCK = f"{ENV_DIR}/uv.lock"
NOTEBOOKS = "https://github.com/unslothai/notebooks/blob"
DOCS = {
    "qwen3_docs": "https://unsloth.ai/docs/basics/qwen3-how-to-run-and-fine-tune",
    "notebook": f"{NOTEBOOKS}/ebe88117bb560d296910f6e733693294d630f024/nb/Qwen3_(14B)-Reasoning-Conversational.ipynb",
    "assistant_only": f"{NOTEBOOKS}/0c6268b8c23bb6e4377e0f8ccc4bbfce4876a3c4/nb/Qwen3_(4B)-Thinking.ipynb",
    "tool_notebook": f"{NOTEBOOKS}/0313a67513a87462ffdb78934f31d2181ba61b06/nb/Qwen3_(4B)-Tool_Calling.ipynb",
    "trl_add_eos": "https://github.com/huggingface/trl/blob/v1.15.0/trl/trainer/sft_trainer.py#L1284-L1306",
}
NOTEBOOK_MAX_SEQ_LENGTH = 2048
GENERAL = {
    "fine_tuning_guide": "https://unsloth.ai/docs/get-started/fine-tuning-llms-guide",
    "lora_guide": "https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide",
    "chat_templates": "https://unsloth.ai/docs/basics/chat-templates",
    "deepseek_v4": "https://unsloth.ai/docs/models/deepseek-v4",
}
DEEPSEEK_DOCS = (
    "https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/08c8657f02291e11b6f6b63e0e16014f8e748518/"
    "encoding/README.md"
)
DEFAULT_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


@dataclass(frozen=True)
class Recipe:
    """What Unsloth's docs say to run for one release: the loader's settings, the response parts, the trainer's
    documented settings, where they come from, and how sure that is (`citation`)."""

    path: str  # parity.tiny: "text" or "vision"
    instruction_part: str
    response_part: str
    trainer_args: dict[str, Any]  # SFTConfig settings the docs show; anything else is left at its default
    citation: str
    sources: tuple[str, ...]
    parts_source: str  # where the response parts come from, as the declaration records it
    not_configurable: str  # why a non-default serving mode is not_configurable for this release
    mode_setting: str


RECIPES: dict[str, Recipe] = {
    "Qwen/Qwen3-0.6B": Recipe(
        path="text",
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
        trainer_args={
            "per_device_train_batch_size": 2, "gradient_accumulation_steps": 4, "warmup_steps": 5, "max_steps": 30,
            "learning_rate": 2e-4, "logging_steps": 1, "optim": "adamw_8bit", "weight_decay": 0.001,
            "lr_scheduler_type": "linear", "seed": 3407, "padding_free": False,
        },  # fmt: skip
        citation="the model's own notebook",
        sources=(DOCS["notebook"], DOCS["assistant_only"]),
        parts_source=DOCS["assistant_only"],
        not_configurable=NOT_CONFIGURABLE,
        mode_setting="framework defaults (the model's default mode); other modes are not configurable (unsloth_docs)",
    ),
    # The human's decision (2026-11-26): Unsloth's DeepSeek-V4 guide documents running, not training, so its general
    # SFT docs apply: FastLanguageModel with max_seq_length 2048 and 4-bit loading, the default LoRA target list, the
    # guide's trainer settings (nothing else set, so padding_free is Unsloth's choice), and the template's own role
    # markers as the response parts. A finding here leaves the repo only if it reproduces with documented choices.
    "deepseek-ai/DeepSeek-V4.1-Flash-1017": Recipe(
        path="text",
        instruction_part="<｜User｜>",
        response_part="<｜Assistant｜>",
        trainer_args={"per_device_train_batch_size": 2, "gradient_accumulation_steps": 4, "max_steps": 60,
                      "learning_rate": 2e-4},  # fmt: skip
        citation="general docs; training this model is not documented",
        sources=(
            GENERAL["fine_tuning_guide"], GENERAL["lora_guide"], GENERAL["chat_templates"], GENERAL["deepseek_v4"],
            f"role markers: {DEEPSEEK_DOCS} (special tokens: <｜User｜> 'User turn prefix', <｜Assistant｜> "
            "'Assistant turn prefix'); checked on our reference render: docs/evidence/2026-11-26-unsloth-deepseek-parts/",
        ),  # fmt: skip
        parts_source=f"the template's own role markers ({DEEPSEEK_DOCS}); Unsloth's guidance: {GENERAL['lora_guide']}",
        not_configurable=NOT_CONFIGURABLE_UNDOCUMENTED,
        mode_setting="framework defaults (the model's default mode); other modes are not configurable (unsloth_docs)",
    ),
}
# Not judged (the human's decision, 2026-11-26), with the evidence in docs/evidence/2026-11-25-m5-env-support/:
# thinkingmachines/Inkling-2-Mini and tencent/Hy4-flash take FastModel's vision path (an image-text class declared,
# text_only left False), and their processors can't be built here (no Inkling2Processor or HYV4FlashProcessor in
# transformers 5.17.0). The Inkling vision stand-in also needs scipy, which the lock doesn't have.
NOT_JUDGED = {
    "thinkingmachines/Inkling-2-Mini": "can't build the release's processor: module transformers has no attribute "
    "Inkling2Processor",
    "tencent/Hy4-flash": "can't build the release's processor: module transformers has no attribute HYV4FlashProcessor",
}
PROBE = [{"role": "user", "content": "Hi."}, {"role": "assistant", "content": "Hello."}]
LABEL_SHIFTING_LOSSES = ("ForCausalLMLoss",)

datasets.disable_progress_bars()
transformers.logging.set_verbosity_error()
for name in ("trl", "unsloth", "unsloth_zoo", "datasets"):
    logging.getLogger(name).setLevel(logging.ERROR)


class UnslothAdapter:
    """Unsloth 2026.11.10's Qwen3 SFT recipe, bound to one model release."""

    name = "unsloth"
    version = VERSION
    backend = BACKEND

    def __init__(self, release: LoadedRelease, env_lock: str = ENV_LOCK):
        """`env_lock`: the lockfile this environment was built from (rule 7). The GPU kit passes the CUDA 12.6
        variant's lock when the machine's driver can't run this environment's own CUDA 13 torch."""
        if not (env_lock.startswith(f"{ENV_DIR}/") and env_lock.endswith("/uv.lock")):
            raise ValueError(f"env_lock must be a uv.lock under {ENV_DIR}/, got {env_lock!r}")
        self.release = release
        self.env_lock = env_lock
        self.env_fingerprint = hashlib.sha256((REPO_ROOT / env_lock).read_bytes()).hexdigest()
        self._source = fetch_release_files(release.release)  # tokenizer, template and config files only
        repo_id = release.release.repo_id
        if repo_id in NOT_JUDGED:
            raise NotImplementedError(f"{repo_id} on Unsloth is not judged: {NOT_JUDGED[repo_id]}")
        if repo_id not in RECIPES:
            raise NotImplementedError(f"no Unsloth recipe decided for {repo_id} (docs/DECISIONS.md)")
        self._recipe = RECIPES[repo_id]
        self._path, self._path_source = self._recipe.path, f"{self._recipe.citation}: " + "; ".join(self._recipe.sources)
        self._local = self._tiny_model_folder()
        self._declarations: dict[TrainCfg, Declaration] = {}
        self._encoded: dict[tuple[str, str, TrainCfg], Encoded | Dropped] = {}
        self._batches: dict[tuple[tuple[str, ...], TrainCfg], Batch] = {}

    def _tiny_model_folder(self) -> Path:
        """A local folder FastLanguageModel can load: a tiny random-weight model plus the real tokenizer files."""
        folder = Path(tempfile.mkdtemp(prefix="parity-unsloth-")) / "model"
        tiny = build_tiny(self._source, self._path)  # random weights; nothing downloaded (parity.tiny)
        self._model_build = {**tiny.record, "path_source": self._path_source}
        return save_tiny(tiny, self._source, folder)

    # The framework, built the way the notebook builds it

    def _trainer(self, convs: Sequence[Conversation], cfg: TrainCfg) -> SFTTrainer:
        if cfg.objective == "completion_only":
            raise NotImplementedError("the fixtures are conversations; completion-only isn't mapped")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(self._local),
            max_seq_length=cfg.max_len if cfg.max_len is not None else NOTEBOOK_MAX_SEQ_LENGTH,
            load_in_4bit=True,
            load_in_8bit=False,
            full_finetuning=False,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=32,
            target_modules=list(DEFAULT_TARGET_MODULES),  # the documented default list (lora_guide)
            lora_alpha=32,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )
        texts = [
            tokenizer.apply_chat_template(copy.deepcopy(conv.messages), tools=conv.tools, tokenize=False)
            for conv in convs
        ]
        args: dict[str, Any] = {
            "output_dir": tempfile.mkdtemp(prefix="parity-unsloth-out-"),
            "dataset_text_field": "text",
            **self._recipe.trainer_args,
            "report_to": "none",
            "packing": cfg.packing,
        }
        if cfg.max_len is not None:
            args["max_length"] = cfg.max_len
        if cfg.pad_to is not None:
            args["pad_to_multiple_of"] = cfg.pad_to
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=datasets.Dataset.from_dict({"text": texts}),
            eval_dataset=None,
            args=SFTConfig(**args),
        )
        if cfg.objective == "assistant_only":
            trainer = train_on_responses_only(
                trainer, instruction_part=self._recipe.instruction_part, response_part=self._recipe.response_part
            )
        _enter_training_mode(trainer)
        return trainer

    def _framework_cfg(self, cfg: TrainCfg) -> TrainCfg:
        """The config as Unsloth gets it: in the model's default mode it is told nothing; any other mode refuses."""
        cfg = replace(cfg, serving_mode=framework_mode(self.release, cfg.serving_mode))  # None in the default mode
        if cfg.serving_mode is not None:
            raise ModeNotConfigurable(self._recipe.not_configurable)
        return cfg

    # The adapter contract

    def declare(self, cfg: TrainCfg) -> Declaration:
        """Read Unsloth's live objects for exactly this config. Nothing here is hardcoded."""
        cfg = self._framework_cfg(cfg)
        if cfg not in self._declarations:
            probe = Conversation(id="probe", messages=copy.deepcopy(PROBE), tools=None)
            trainer = self._trainer([probe], cfg)
            self._declarations[cfg] = _declaration(trainer, cfg, self._source, self._recipe, self._model_build)
        return self._declarations[cfg]

    def encode(self, conv: Conversation, cfg: TrainCfg) -> Encoded | Dropped:
        cfg = self._framework_cfg(cfg)
        key = (conv.id, conv.content_key, cfg)
        if key not in self._encoded:
            trainer = self._trainer([conv], cfg)
            if len(trainer.train_dataset) == 0:
                self._encoded[key] = Dropped(
                    "the recipe's dataset preparation removed the row (TRL drops fully masked examples; "
                    "train_on_responses_only drops rows with no response)"
                )
            else:
                batch = trainer.data_collator([trainer.train_dataset[0]])
                self._encoded[key] = _row(batch, 0)
        return self._encoded[key]

    def encode_batch(self, convs: Sequence[Conversation], cfg: TrainCfg) -> Batch:
        cfg = self._framework_cfg(cfg)
        key = (tuple(f"{conv.id}:{conv.content_key}" for conv in convs), cfg)
        if key not in self._batches:
            trainer = self._trainer(convs, cfg)
            if len(trainer.train_dataset) != len(convs):
                raise RuntimeError(f"the recipe kept {len(trainer.train_dataset)} of {len(convs)} rows; can't line them up")
            batch = trainer.data_collator([trainer.train_dataset[i] for i in range(len(convs))])
            self._batches[key] = Batch(tuple(_row(batch, i) for i in range(len(convs))))
        return self._batches[key]


def _enter_training_mode(trainer: SFTTrainer) -> None:
    """Put the model in the state Unsloth's own train() puts it in before it draws any batch, so the collator is
    read as training uses it (harness error H-11, found by the skeptic on the GPU run).

    Unsloth's patched SFTTrainer.__init__ ends with `model.for_inference()`, which sets the tokenizer's padding side to
    left. Its train() wrapper (`unsloth/models/rl.py`, `prepare_for_training_mode`) first calls
    `model.for_training(use_gradient_checkpointing=...)`, which sets it back to right. This calls the same public
    method with the same argument the wrapper computes, and nothing else; the CPU route's model has it too.
    """
    model = trainer.model
    if not hasattr(model, "for_training"):
        return
    if hasattr(model, "_unsloth_gradient_checkpointing"):
        use_gc = model._unsloth_gradient_checkpointing
    else:
        use_gc = getattr(trainer.args, "gradient_checkpointing", True)
    model.for_training(use_gradient_checkpointing=use_gc)


def _row(batch: dict[str, Any], i: int) -> Encoded:
    """One row of the collator's output, exactly as the recipe's collator produced it."""
    ids = tuple(int(t) for t in batch["input_ids"][i].tolist())
    labels = tuple(int(t) for t in batch["labels"][i].tolist())
    attention = (
        tuple(int(t) for t in batch["attention_mask"][i].tolist()) if "attention_mask" in batch else (1,) * len(ids)
    )
    positions = tuple(int(t) for t in batch["position_ids"][i].tolist()) if "position_ids" in batch else None
    return Encoded(ids, labels, attention, positions)


def _loss_name(model: Any) -> str:
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    loss = getattr(base, "loss_function", None)
    return getattr(loss, "__name__", repr(loss))


def _declaration(
    trainer: SFTTrainer, cfg: TrainCfg, source: Path, recipe: Recipe, model_build: dict[str, Any]
) -> Declaration:
    mode_setting = recipe.mode_setting
    tokenizer = trainer.processing_class
    template = tokenizer.chat_template
    own = AutoTokenizer.from_pretrained(source).chat_template
    template_source = "model" if template == own else f"unsloth@{VERSION}:loader"

    loss = _loss_name(trainer.model)
    if loss not in LABEL_SHIFTING_LOSSES:
        raise ValueError(f"can't tell the label convention: the model's loss_function is {loss!r}")

    args = trainer.args
    truncation = {"keep_start": "right", "keep_end": "left"}.get(args.truncation_mode)
    if truncation is None:
        raise ValueError(f"truncation_mode={args.truncation_mode!r} has no Parity truncation mode")
    collator = trainer.data_collator
    collator_tokenizer = getattr(collator, "tokenizer", None)
    pad = collator_tokenizer.pad_token_id if collator_tokenizer is not None else getattr(collator, "pad_token_id", None)
    packed = bool(args.packing or args.padding_free)
    return Declaration(
        chat_template=template,
        template_source=template_source,
        label_convention="aligned",
        truncation_mode=truncation,
        batching="packed" if packed else "padded",
        pad_token_id=pad,
        tool_arguments="passed_through",
        serving_mode_setting=mode_setting,
        model_build=model_build,
        sources=(
            f"chat_template: trainer.processing_class.chat_template, the tokenizer FastLanguageModel.from_pretrained "
            f"returned ({DOCS['notebook']})",
            f"objective: {cfg.objective}; train_on_responses_only(instruction_part={recipe.instruction_part!r}, "
            f"response_part={recipe.response_part!r}) for assistant_only ({recipe.parts_source})",
            f"label_convention: the model's loss_function is {loss}, which shifts labels itself",
            f"truncation_mode: trainer.args.truncation_mode={args.truncation_mode!r}, "
            f"max_length={args.max_length} (TRL {trl.__version__})",
            f"batching: trainer.args.packing={args.packing}, padding_free={args.padding_free}",
            f"pad_token_id: {type(collator).__name__}.tokenizer.pad_token_id={pad}, "
            f"pad_to_multiple_of={getattr(collator, 'pad_to_multiple_of', None)} "
            f"(trainer.args.pad_to_multiple_of={getattr(args, 'pad_to_multiple_of', None)}), "
            f"label_pad_token_id={getattr(collator, 'label_pad_token_id', None)}, "
            f"padding_side={getattr(collator_tokenizer, 'padding_side', None)!r}",
            "tool_arguments: no option; the recipe formats with tokenizer.apply_chat_template itself, so a string "
            f"argument is rendered verbatim by the template ({DOCS['tool_notebook']})",
            f"serving_mode_setting: {mode_setting}",
            f"backend: {BACKEND} (unsloth.DEVICE_TYPE={DEVICE_TYPE!r})",
            f"note: stock TRL appends the EOS token to a text row that doesn't end with it ({DOCS['trl_add_eos']}); "
            "Unsloth's CPU backend runs that preparation, its GPU route replaces it (P-0004)",
        ),
    )
