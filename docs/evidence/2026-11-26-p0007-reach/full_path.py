"""P-0007's reach today: the affected bundled templates through Axolotl 0.21.1's full preprocessing path.

Run from the repo root, in the Axolotl environment, as a file (dataset loading may start worker processes):
    envs/axolotl-0.21.1/.venv/bin/python docs/evidence/2026-11-26-p0007-reach/full_path.py

scan_bundled.py found three bundled templates that read a message field Axolotl drops: exaone4, gemma4 and
gemma4_unified. This runs each one the way a user would, with a model Axolotl 0.21.1 loads (its model type is in
transformers 5.20.1) and that Axolotl pairs with the template: gemma-4-31B with `chat_template: gemma4` and
gemma-4-12B-it with `chat_template: gemma4_unified`, as in Axolotl's v0.21.1 examples, and EXAONE-4.0-1.2B with
`chat_template: exaone4`. Only the tokenizer and config files are downloaded (rule 6, `parity.release`).

Each model also runs with its own template (`chat_template: tokenizer_default`) where it ships one (gemma-4-31B, a
base model, ships none). The conversations are in plain OpenAI format (a jsonl with `messages`, and `tools`):
  - reasoning: one assistant turn with `reasoning_content`;
  - tool_call: a call with reasoning and empty content, the tool's result (`tool_call_id`, `name`), a final answer;
  - tool_call_native (Gemma 4 only): the same call with its result on the assistant message (`tool_responses`, the
    Gemma-native form these templates also read), then the final answer.
Each goes through `load_cfg` + `load_datasets` (what `axolotl preprocess` runs), with `type: chat_template`, the
bundled template, and no other dataset setting (Axolotl's validated defaults), then again with the fields mapped
explicitly in `message_property_mappings` (the workaround). The prepared row is compared with the template's own
render of the same messages and template kwargs; the script also reports whether the reasoning text and the tool
call are in the row and trained, or the exception if preprocessing raised, and what Axolotl's analyzer lists for each
template. The mapped run maps the OpenAI fields only, not Gemma's `tool_responses`. exaone4 renders reasoning only with
`skip_think: false`, so its runs set `chat_template_kwargs: {skip_think: false}`. Writes full_path.json.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("AXOLOTL_DO_NOT_TRACK", "1")
os.environ.setdefault("DO_NOT_TRACK", "1")

HERE = Path(__file__).resolve().parent

CASES = (
    ("exaone4", "LGAI-EXAONE/EXAONE-4.0-1.2B@3abf2810673c7c0778df64a73c2d52eab32d91c4", {"skip_think": False}),
    ("gemma4", "google/gemma-4-31B@5bbc2fb1c1b2c611d06e3d9f23c170ba21659d89", {}),
    ("gemma4_unified", "google/gemma-4-12B-it@43c9d014b231474fa7212dbb48a762566c2b255d", {}),
)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a city.",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        },
    }
]
ROWS = {
    "reasoning": {
        "messages": [
            {"role": "user", "content": "Is 97 prime?"},
            {"role": "assistant", "reasoning_content": "PROBE_REASONING: no divisor up to 9.", "content": "Yes, 97 is prime."},
        ]
    },
    "tool_call_native": {
        "messages": [
            {"role": "user", "content": "What's the weather in Paris?"},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "PROBE_REASONING: I should call the weather tool.",
                "tool_calls": [
                    {"id": "abc123XYZ", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "Paris"}}}
                ],
                "tool_responses": [{"name": "get_weather", "response": "18C and sunny"}],
            },
            {"role": "assistant", "content": "It is 18C and sunny in Paris."},
        ],
        "tools": TOOLS,
    },
    "tool_call": {
        "messages": [
            {"role": "user", "content": "What's the weather in Paris?"},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "PROBE_REASONING: I should call the weather tool.",
                "tool_calls": [
                    {"id": "abc123XYZ", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "Paris"}}}
                ],
            },
            {"role": "tool", "tool_call_id": "abc123XYZ", "name": "get_weather", "content": "18C and sunny"},
            {"role": "assistant", "content": "It is 18C and sunny in Paris."},
        ],
        "tools": TOOLS,
    },
}
MAPPED = {
    "message_property_mappings": {
        "role": "role",
        "content": "content",
        "reasoning_content": "reasoning_content",
        "tool_calls": "tool_calls",
        "tool_call_id": "tool_call_id",
        "name": "name",
    }
}
# What shows the reasoning and the tool call in each template's render
REASONING_MARK = "PROBE_REASONING"
# (EXAONE's tool list in the system prompt also has '"name": "get_weather"', so its mark includes the call tag)
CALL_MARKS = {"exaone4": '<tool_call>{"name": "get_weather"', "gemma4": "call:get_weather", "gemma4_unified": "call:get_weather"}


def first_difference(a: str, b: str) -> dict | None:
    if a == b:
        return None
    i = next((k for k, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    return {"char": i, "row": a[max(0, i - 80) : i + 120], "render": b[max(0, i - 80) : i + 120]}


def run_one(model_dir: Path, template: str, source: str, kwargs: dict, row: dict, extra_ds: dict) -> dict:
    """One conversation through load_cfg + load_datasets. `source` is the config's chat_template: the bundled
    template's name, or tokenizer_default (the model's own); `template` names the family for the marks."""
    import yaml
    from axolotl.cli.args import PreprocessCliArgs
    from axolotl.cli.config import load_cfg
    from axolotl.common.datasets import load_datasets
    from axolotl.utils.chat_templates import get_chat_template
    from axolotl.utils.trainer import disable_datasets_caching
    from transformers import AutoTokenizer

    work = Path(tempfile.mkdtemp(prefix=f"p0007-{template}-"))
    (work / "data.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    cfg = {
        "base_model": str(model_dir),
        "chat_template": source,
        "datasets": [{"path": str(work / "data.jsonl"), "type": "chat_template", **extra_ds}],
        "sequence_len": 4096,
        "dataset_prepared_path": str(work / "prepared"),
        "output_dir": str(work / "out"),
        "val_set_size": 0,
        "dataset_num_proc": 1,
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "num_epochs": 1,
        "learning_rate": 1e-5,
        "shuffle_merged_datasets": False,
    }
    if kwargs:
        cfg["chat_template_kwargs"] = kwargs
    (work / "config.yml").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    tok = AutoTokenizer.from_pretrained(model_dir)
    own = source == "tokenizer_default"
    render = tok.apply_chat_template(
        row["messages"],
        tools=row.get("tools"),
        chat_template=None if own else get_chat_template(source),
        tokenize=False,
        **kwargs,
    )
    out: dict = {"render_has_reasoning": REASONING_MARK in render, "render_has_call": CALL_MARKS[template] in render}
    try:
        c = load_cfg(str(work / "config.yml"))
        out["validated_mappings"] = dict(c.datasets[0].get("message_property_mappings") or {})
        with disable_datasets_caching():
            meta = load_datasets(cfg=c, cli_args=PreprocessCliArgs(debug_num_examples=0))
    except Exception as e:  # the outcome to record, not a harness failure
        out["raised"] = f"{type(e).__name__}: {str(e)[:300]}"
        return out
    if len(meta.train_dataset) == 0:
        out["rows"] = 0
        return out
    ex = meta.train_dataset[0]
    ids, labels = list(ex["input_ids"]), list(ex["labels"])
    text = tok.decode(ids)
    trained = tok.decode([t for t, lab in zip(ids, labels) if lab != -100])
    out.update(
        {
            "rows": len(meta.train_dataset),
            "row_equals_render": text == render,
            "row_has_reasoning": REASONING_MARK in text,
            "row_has_call": CALL_MARKS[template] in text,
            "reasoning_trained": REASONING_MARK in trained,
            "call_trained": CALL_MARKS[template] in trained,
            "first_difference": first_difference(text, render),
        }
    )
    return out


def main() -> None:
    import axolotl
    import transformers

    from parity.release import Release, fetch_release_files

    result = {"axolotl": axolotl.__version__, "transformers": transformers.__version__, "cases": {}}
    for template, spec, kwargs in CASES:
        model_dir = fetch_release_files(Release.parse(spec))
        config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        case = {"model": spec, "model_type": config.get("model_type"), "chat_template_kwargs": kwargs, "runs": {}}
        has_own = (model_dir / "chat_template.jinja").exists() or "chat_template" in json.loads(
            (model_dir / "tokenizer_config.json").read_text(encoding="utf-8")
        )
        sources = (template, "tokenizer_default") if has_own else (template,)
        case["analyzer_fields"] = {}
        for source in sources:
            from axolotl.prompt_strategies.jinja_template_analyzer import JinjaTemplateAnalyzer
            from axolotl.utils.chat_templates import get_chat_template
            from transformers import AutoTokenizer

            text = AutoTokenizer.from_pretrained(model_dir).chat_template if source == "tokenizer_default" else get_chat_template(source)
            label = "own" if source == "tokenizer_default" else "bundled"
            case["analyzer_fields"][label] = sorted(JinjaTemplateAnalyzer(text).get_message_vars("messages"))
        for source in sources:
            for row_name, row in ROWS.items():
                if row_name == "tool_call_native" and not template.startswith("gemma4"):
                    continue
                for variant, extra in (("defaults", {}), ("mapped", MAPPED)):
                    key = f"{'own' if source == 'tokenizer_default' else 'bundled'}/{row_name}/{variant}"
                    case["runs"][key] = run_one(model_dir, template, source, kwargs, row, extra)
        result["cases"][template] = case
    (HERE / "full_path.json").write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"axolotl {result['axolotl']}, transformers {result['transformers']}")
    for template, case in result["cases"].items():
        print(f"{template} ({case['model']}, model_type {case['model_type']}); analyzer: {case['analyzer_fields']}")
        for key, r in case["runs"].items():
            if "raised" in r:
                print(f"  {key}: RAISED {r['raised'][:160]}")
                continue
            print(
                f"  {key}: row == render {r.get('row_equals_render')}; reasoning in render/row/trained "
                f"{r['render_has_reasoning']}/{r.get('row_has_reasoning')}/{r.get('reasoning_trained')}; "
                f"call in render/row/trained {r['render_has_call']}/{r.get('row_has_call')}/{r.get('call_trained')}"
            )


if __name__ == "__main__":
    main()
