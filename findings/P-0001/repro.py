"""P-0001 repro: with Qwen3's own template, Axolotl 0.21.1's default labeling never trains the think block.

Run with axolotl==0.21.1 installed (CPU is enough; only tokenizer and config files are downloaded):
    python findings/P-0001/repro.py
Exit code 1 if the behavior reproduces, 0 if not.
"""

import os

os.environ["AXOLOTL_DO_NOT_TRACK"] = "1"
os.environ["DO_NOT_TRACK"] = "1"

import json  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import yaml  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402

MODEL = "Qwen/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
FILES = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "config.json", "generation_config.json"]
MESSAGES = [
    {"role": "user", "content": "Name three primary colors."},
    {"role": "assistant", "content": "Red, yellow and blue."},
]


def main() -> int:
    from axolotl.cli.args import PreprocessCliArgs
    from axolotl.cli.config import load_cfg
    from axolotl.common.datasets import load_datasets
    from axolotl.utils.trainer import disable_datasets_caching
    from transformers import AutoTokenizer

    local = snapshot_download(MODEL, revision=REVISION, allow_patterns=FILES)
    work = Path(tempfile.mkdtemp())
    (work / "data.jsonl").write_text(json.dumps({"messages": MESSAGES}) + "\n")
    config = {
        "base_model": local,
        "datasets": [{"path": str(work / "data.jsonl"), "type": "chat_template",
                      "roles_to_train": ["assistant"], "train_on_eos": "turn"}],
        "sequence_len": 2048, "dataset_prepared_path": str(work / "prepared"), "output_dir": str(work / "out"),
        "dataset_num_proc": 1, "val_set_size": 0, "micro_batch_size": 1, "gradient_accumulation_steps": 1,
        "num_epochs": 1, "learning_rate": 1e-5, "bf16": False, "fp16": False,
    }  # fmt: skip
    (work / "config.yml").write_text(yaml.safe_dump(config))
    cfg = load_cfg(str(work / "config.yml"))
    with disable_datasets_caching():
        row = load_datasets(cfg=cfg, cli_args=PreprocessCliArgs(debug_num_examples=0)).train_dataset[0]

    tok = AutoTokenizer.from_pretrained(local)
    # Qwen3's own template renders the last assistant turn as
    # "<|im_start|>assistant\n<think>\n\n</think>\n\nRed, yellow and blue.<|im_end|>\n". In thinking mode the
    # generation prompt ends at "<|im_start|>assistant\n", so the model itself writes the think block.
    tokens = [(tok.decode([t]), label != -100) for t, label in zip(row["input_ids"], row["labels"])]
    start = max(i for i, (text, _) in enumerate(tokens) if text == "<think>")
    shown = tokens[start : start + 6]
    print("last assistant turn (token, trained):", shown)
    think_block_trained = [trained for _, trained in tokens[start : start + 4]]
    return 1 if not any(think_block_trained) and shown[4][1] else 0


if __name__ == "__main__":
    sys.exit(main())
