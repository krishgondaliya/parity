"""P-0008: tml-renderers on the repro's own conversation (2026-11-27), for the model-repo discussion.

Run from the repo root, in TRL's vlm environment:
    envs/trl-1.18.2/vlm/.venv/bin/python docs/evidence/2026-11-26-p0008-renderer/repro_cases.py <tml-renderers venv python>

compare.py ran the renderer on Parity's `tool_call` fixture. The discussion's repro (findings/P-0008/repro.py) uses
a shorter conversation, so this gives the renderer exactly that conversation, under the same sandbox (compare.sandbox:
the human's conditions, recorded in docs/DECISIONS.md, 2026-11-26), with the tool-call turn's content in four forms:
"", null, missing, and one empty text part. The conversation is read from repro.py itself (its `conversation`
function), so the two can't drift. Effort 0.9, the template's default. Writes repro_results.json next to this file.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
REPRO = REPO / "findings/P-0008/repro.py"
EMPTY_TEXT_MESSAGE = "<|content_text|><|end_message|>"


def repro_namespace() -> dict:
    """repro.py's objects, its source run with its final sys.exit turned into an assignment."""
    ns: dict = {"__name__": "repro"}
    source = REPRO.read_text(encoding="utf-8").replace("sys.exit(", "_exit_code = (")
    exec(compile(source, str(REPRO), "exec"), ns)
    return ns


def main() -> None:
    from trl.data_utils import prepare_multimodal_messages

    sys.path.insert(0, str(HERE))
    import compare

    ns = repro_namespace()
    tok, conversation = ns["tok"], ns["conversation"]
    forms = {'""': "", "null": None, "missing": ..., '[{"type": "text", "text": ""}]': [{"type": "text", "text": ""}]}
    cases = []
    for name, content in forms.items():
        messages = conversation("")
        if content is ...:
            del messages[1]["content"]
        else:
            messages[1]["content"] = content
        cases.append({"name": name, "messages": messages, "tools": None})
    rendered, record = compare.sandbox(Path(sys.argv[1]), {"effort": compare.EFFORT, "cases": cases})
    out = {"tml_renderers": {k: rendered[k] for k in ("version", "module_sha256")}, "effort": compare.EFFORT,
           "sandbox": record, "cases": {}}  # fmt: skip
    for case in cases:
        hf = tok.apply_chat_template(case["messages"], tokenize=False)
        r = rendered["cases"][case["name"]]
        if "error" in r:
            out["cases"][case["name"]] = {"given": case["messages"][1], "renderer_error": r["error"]}
            continue
        [example] = r["examples"]
        out["cases"][case["name"]] = {
            "given": case["messages"][1],
            "renderer_text": example["text"],
            "renderer_empty_text_messages": example["text"].count(EMPTY_TEXT_MESSAGE),
            "hf_template_empty_text_messages": hf.count(EMPTY_TEXT_MESSAGE),
            "renderer_equals_hf_template": {
                "text": example["text"] == hf,
                "ids": example["tokens"] == list(tok(hf, add_special_tokens=False)["input_ids"]),
            },
        }
    wrapped = prepare_multimodal_messages(copy.deepcopy(conversation("")))
    out["trl_vision_path"] = {
        "content_becomes": wrapped[1]["content"],
        "empty_text_messages": tok.apply_chat_template(wrapped, tokenize=False).count(EMPTY_TEXT_MESSAGE),
    }
    texts = {c["renderer_text"] for c in out["cases"].values() if "renderer_text" in c}
    out["renderer_output_identical_across_forms"] = len(texts) == 1
    (HERE / "repro_results.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for name, c in out["cases"].items():
        print(f"{name}: renderer empty text messages {c.get('renderer_empty_text_messages')}, HF template "
              f"{c.get('hf_template_empty_text_messages')}, renderer = HF template {c.get('renderer_equals_hf_template')}"
              f"{' ERROR ' + c['renderer_error'] if 'renderer_error' in c else ''}")  # fmt: skip
    print("renderer output identical across the four forms:", out["renderer_output_identical_across_forms"])
    print("TRL's vision path:", out["trl_vision_path"])
    print("sandbox:", {k: record[k] for k in ("returncode", "probe_outside_written", "temp_dir_left")})
    print("renderer text:", repr(next(iter(texts))))


if __name__ == "__main__":
    main()
