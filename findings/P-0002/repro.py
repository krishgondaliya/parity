"""P-0002 repro: Axolotl 0.21.1 never trains the closing token of an empty assistant turn.

Run with axolotl==0.21.1 installed (CPU is enough; only tokenizer and config files are downloaded):
    python findings/P-0002/repro.py
Exit code 1 if the bug reproduces, 0 if not.
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
    {"role": "user", "content": "Reply with an empty message."},
    {"role": "assistant", "content": ""},
    {"role": "user", "content": "Now reply with the word OK."},
    {"role": "assistant", "content": "OK."},
]


def main() -> int:
    from axolotl.cli.args import PreprocessCliArgs
    from axolotl.cli.config import load_cfg
    from axolotl.common.datasets import load_datasets
    from axolotl.utils.trainer import disable_datasets_caching

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

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(local)
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    assistant = tok.convert_tokens_to_ids("assistant")
    # Each assistant turn renders as "<|im_start|>assistant\n ... <|im_end|>"; find each turn's closing token.
    ids, labels = row["input_ids"], row["labels"]
    closing = []
    for i in range(len(ids) - 1):
        if ids[i] == assistant and i > 0 and ids[i - 1] == tok.convert_tokens_to_ids("<|im_start|>"):
            closing.append(next(j for j in range(i, len(ids)) if ids[j] == im_end))
    trained = [labels[j] != -100 for j in closing]
    print("assistant turn closing tokens at", closing, "trained:", trained)
    return 1 if trained == [False, True] else 0


if __name__ == "__main__":
    sys.exit(main())
