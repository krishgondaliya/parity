"""P-0007: does repro.py's strategy call match Axolotl's full preprocessing path, and does mapping the fields fix it?

Run from the repo root, in the Axolotl environment, as a file:
    envs/axolotl-0.21.1/.venv/bin/python findings/P-0007/path_check.py

repro.py calls the chat_template strategy directly (`load` + `tokenize_prompt`) with a dataset config validated by
`SFTDataset` and dumped with `exclude_none=True`, as Axolotl's config validation does. This checks, on repro.py's own
conversation and the same pinned EXAONE-4.0 tokenizer:
  1. the full path (`load_cfg` + `load_datasets`, what `axolotl preprocess` runs, with the model's config and
     tokenizer files as `base_model`) gives the same input ids and labels as the strategy call;
  2. with the fields listed in `message_property_mappings`, the strategy's row equals the template's render.
Writes path_check.json next to this file.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("AXOLOTL_DO_NOT_TRACK", "1")
os.environ.setdefault("DO_NOT_TRACK", "1")

HERE = Path(__file__).resolve().parent
SPEC = "LGAI-EXAONE/EXAONE-4.0-1.2B@3abf2810673c7c0778df64a73c2d52eab32d91c4"
MAPPED = {"role": "role", "content": "content", "reasoning_content": "reasoning_content", "tool_calls": "tool_calls"}


def main() -> None:
    import yaml
    from axolotl.cli.args import PreprocessCliArgs
    from axolotl.cli.config import load_cfg
    from axolotl.common.datasets import load_datasets
    from axolotl.prompt_strategies.chat_template import load
    from axolotl.utils.dict import DictDefault
    from axolotl.utils.schemas.datasets import SFTDataset
    from axolotl.utils.trainer import disable_datasets_caching

    from parity.release import Release, fetch_release_files

    # repro.py's own objects: run its source with its final sys.exit turned into an assignment
    ns: dict = {}
    source = (HERE / "repro.py").read_text(encoding="utf-8").replace("sys.exit(", "_exit_code = (")
    exec(compile(source, str(HERE / "repro.py"), "exec"), ns)
    tok, messages, tools, kwargs, strategy = ns["tok"], ns["messages"], ns["tools"], ns["kwargs"], ns["strategy"]
    direct = strategy.tokenize_prompt({"messages": messages, "tools": tools})

    work = Path(tempfile.mkdtemp(prefix="p0007-path-"))
    (work / "data.jsonl").write_text(json.dumps({"messages": messages, "tools": tools}) + "\n", encoding="utf-8")
    cfg = {
        "base_model": str(fetch_release_files(Release.parse(SPEC))),
        "chat_template": "exaone4",
        "chat_template_kwargs": kwargs,
        "datasets": [{"path": str(work / "data.jsonl"), "type": "chat_template"}],
        "sequence_len": 4096,
        "dataset_prepared_path": str(work / "prepared"),
        "output_dir": str(work / "out"),
        "val_set_size": 0,
        "dataset_num_proc": 1,
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "num_epochs": 1,
        "learning_rate": 1e-5,
    }
    (work / "config.yml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with disable_datasets_caching():
        row = load_datasets(cfg=load_cfg(str(work / "config.yml")), cli_args=PreprocessCliArgs(debug_num_examples=0)).train_dataset[0]

    ds = SFTDataset(path="data.jsonl", type="chat_template", chat_template="exaone4", message_property_mappings=MAPPED)
    mapped = load(tok, DictDefault({"sequence_len": 4096, "chat_template_kwargs": kwargs}),
                  DictDefault(ds.model_dump(exclude_none=True)))
    mapped_row = tok.decode(mapped.tokenize_prompt({"messages": messages, "tools": tools})["input_ids"])
    result = {
        "model": SPEC,
        "full_path_input_ids_equal": list(row["input_ids"]) == list(direct["input_ids"]),
        "full_path_labels_equal": list(row["labels"]) == list(direct["labels"]),
        "mapped": MAPPED,
        "mapped_row_equals_render": mapped_row == ns["expected"],
    }
    (HERE / "path_check.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
