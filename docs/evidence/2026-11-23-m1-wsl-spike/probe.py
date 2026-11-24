"""M1 platform spike probe: import only the data-path modules an adapter needs.

Usage: <env>/bin/python probe.py <framework> <out.json>
Prints nothing on success except what the framework itself prints on import.
"""

import importlib
import importlib.metadata as md
import json
import platform
import sys
import traceback

# (module, [names to fetch from it]). Order matters: Unsloth must be imported first.
DATA_PATH = {
    "trl": [
        ("trl", ["SFTTrainer", "SFTConfig"]),
        ("trl.data_utils", []),
        ("trl.chat_template_utils", ["get_training_chat_template"]),
    ],
    "axolotl": [
        ("axolotl.prompt_strategies", ["load"]),
        ("axolotl.prompt_strategies.chat_template", ["ChatTemplatePrompter", "ChatTemplateStrategy", "StrategyLoader"]),
        ("axolotl.utils.chat_templates", ["get_chat_template"]),
        ("axolotl.utils.collators", ["DataCollatorForSeq2Seq", "V2BatchSamplerDataCollatorForSeq2Seq"]),
        ("axolotl.utils.data.sft", ["prepare_datasets"]),
    ],
    "unsloth": [
        ("unsloth", []),
        ("unsloth.chat_templates", ["get_chat_template", "train_on_responses_only"]),
        ("unsloth_zoo.dataset_utils", []),
    ],
    "verl": [
        ("verl.utils.dataset.multiturn_sft_dataset", ["MultiTurnSFTDataset"]),
        ("verl.utils.dataset.rl_dataset", ["RLHFDataset"]),
    ],
    "skyrl": [
        ("skyrl.train.dataset.sft_dataset", ["SFTDataset"]),
        ("skyrl.train.dataset.dataset", ["PromptDataset"]),
        ("skyrl.train.dataset.collators", ["DefaultCollator"]),
        ("skyrl.train.dataset.preprocess", ["convert_prompts_responses_to_batch_tensors"]),
        ("skyrl.train.generators.utils", ["get_response_ids_and_loss_mask_from_messages"]),
    ],
}

VERSIONED = ["torch", "transformers", "tokenizers", "trl", "triton", "flash-attn",
             "axolotl", "unsloth", "unsloth_zoo", "verl", "skyrl"]


def version(dist: str) -> str:
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return "none"


def main() -> None:
    framework, out_path = sys.argv[1], sys.argv[2]
    imports = []
    for module_name, names in DATA_PATH[framework]:
        record = {"module": module_name, "names": names, "ok": False, "error": None}
        try:
            module = importlib.import_module(module_name)
            for name in names:
                getattr(module, name)
            record["ok"] = True
        except BaseException as exc:  # record SystemExit and friends too
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["traceback_tail"] = traceback.format_exc().splitlines()[-6:]
        imports.append(record)

    cuda = "torch not importable"
    try:
        import torch

        cuda = torch.cuda.is_available()
    except Exception as exc:
        cuda = f"{type(exc).__name__}: {exc}"

    result = {
        "framework": framework,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cuda_available": cuda,
        "versions": {d: version(d) for d in VERSIONED},
        "all_ok": all(r["ok"] for r in imports),
        "imports": imports,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
