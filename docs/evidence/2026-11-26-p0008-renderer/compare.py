"""P-0008's reference: what Thinking Machines' own renderer does with an empty assistant content on a tool-call turn.

Run from the repo root, in TRL's vlm environment:
    envs/trl-1.18.2/vlm/.venv/bin/python docs/evidence/2026-11-26-p0008-renderer/compare.py <tml-renderers venv python>

tml-renderers 0.5.0 is Thinking Machines' official renderer for Inkling, so it is P-0008's reference
(docs/DECISIONS.md, 2026-11-26, recorded before this ran). It runs in a sandbox under the human's conditions:
  - a throwaway venv outside envs/ (Python 3.12, torch CPU, the wheel pinned by sha256; the module's sha256 is
    recorded from inside the run);
  - `unshare -rnm`: no network, and every mount remounted read-only except an empty temp dir, so a write anywhere else
    fails the run. Before the renderer starts, the sandbox checks that a write outside the temp dir fails, and aborts
    if it doesn't;
  - `env -i`; the venv's Python with `-I -B`, run from the temp dir; cases in on stdin, results out on stdout.

The cases are the `tool_call` fixture with the tool-call turn's content in each form an OpenAI-format dataset can
carry: "" (the fixture as given), null, absent, and one empty text part ([{"type": "text", "text": ""}], what TRL's
vision path turns "" into). For each, the renderer's SFT tokens are compared with Inkling's Hugging Face template
(string and list forms) and with TRL's row text (`prepare_multimodal_messages`, then the template), and the script
counts the empty text message (<|content_text|><|end_message|>) in each. The renderer runs with effort 0.9, the
thinking effort Inkling's Hugging Face template renders by default ("Thinking effort level: 0.9"); with no effort the
renderer adds no effort message, which was the only other difference seen. Writes results.json and summary.md.
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from parity.fixtures import load_fixture
from parity.release import load_release
from parity.targets import PHASE0_TARGETS

HERE = Path(__file__).resolve().parent
WHEEL_SHA256 = "465f2c7e33ab3f098610aa5086b06ce68e7d964053700e7fd261f873e054db14"
INKLING = next(t for t in PHASE0_TARGETS if t.repo_id == "thinkingmachines/Inkling-2-Mini")
EMPTY_TEXT_MESSAGE = "<|content_text|><|end_message|>"
EFFORT = 0.9  # the HF template's default thinking effort
FORMS = {"empty string": "", "null": None, "absent": ..., "one empty text part": [{"type": "text", "text": ""}]}


def cases() -> list[dict]:
    conv = load_fixture("tool_call")
    out = []
    for name, content in FORMS.items():
        messages = copy.deepcopy(list(conv.messages))
        turn = next(m for m in messages if m["role"] == "assistant" and m.get("tool_calls"))
        if content is ...:
            del turn["content"]
        else:
            turn["content"] = content
        out.append({"name": name, "messages": messages, "tools": conv.tools})
    return out


def sandbox(venv_python: Path, payload: dict) -> tuple[dict, dict]:
    work = Path(tempfile.mkdtemp(prefix="parity-tmlr-"))
    probe = HERE / "sandbox-write-probe"  # outside the temp dir: writing it must fail
    script = (
        "for m in $(awk '{print $5}' /proc/self/mountinfo | sort -u); do mount -o remount,bind,ro \"$m\" 2>/dev/null; done\n"
        f'mount --bind "{work}" "{work}" && mount -o remount,bind,rw "{work}" || exit 97\n'
        f'if (echo x > "{probe}") 2>/dev/null; then echo "sandbox: a write outside the temp dir succeeded" >&2; exit 98; fi\n'
        f'cd "{work}" && exec /usr/bin/env -i "{venv_python}" -I -B "{HERE / "driver.py"}"\n'
    )
    proc = subprocess.run(
        ["unshare", "-rnm", "/bin/sh", "-c", script],
        input=json.dumps(payload, ensure_ascii=False),
        capture_output=True,
        text=True,
        timeout=600,
    )
    left = sorted(p.name for p in work.iterdir())
    record = {
        "command": ["unshare", "-rnm", "/bin/sh", "-c", "<remount all read-only but the temp dir; probe a write outside; "
                    "cd temp; exec env -i venv/bin/python -I -B driver.py>"],  # fmt: skip
        "returncode": proc.returncode,
        "probe_outside_written": probe.exists(),
        "temp_dir_left": left,
        "stderr_tail": proc.stderr[-1500:],
    }
    if proc.returncode != 0 or probe.exists():
        raise SystemExit(f"sandbox failed: {json.dumps(record, indent=1)}")
    return json.loads(proc.stdout), record


def main() -> None:
    from trl.data_utils import prepare_multimodal_messages

    venv_python = Path(sys.argv[1])
    release = load_release(INKLING)
    tok = release.tokenizer
    template = tok.chat_template if isinstance(tok.chat_template, str) else tok.chat_template["default"]
    rendered, record = sandbox(venv_python, {"effort": EFFORT, "cases": cases()})
    results = {
        "tml_renderers": {k: rendered[k] for k in ("version", "module_sha256")},
        "effort": EFFORT,
        "sandbox": record,
        "cases": {},
    }
    for case in cases():
        name = case["name"]
        hf = tok.apply_chat_template(case["messages"], tools=case["tools"], tokenize=False)
        trl_row = tok.apply_chat_template(
            prepare_multimodal_messages(copy.deepcopy(case["messages"])), tools=case["tools"], tokenize=False
        )
        r = rendered["cases"][name]
        entry = {
            "hf_template_empty_text_messages": hf.count(EMPTY_TEXT_MESSAGE),
            "trl_vision_row_empty_text_messages": trl_row.count(EMPTY_TEXT_MESSAGE),
        }
        if "error" in r:
            entry["renderer"] = {"error": r["error"]}
        else:
            [example] = r["examples"]
            text = example["text"]
            entry["renderer"] = {
                "empty_text_messages": text.count(EMPTY_TEXT_MESSAGE),
                "equals_hf_template_text": text == hf,
                "equals_hf_template_ids": example["tokens"] == list(tok(hf, add_special_tokens=False)["input_ids"]),
                "equals_trl_vision_row_text": text == trl_row,
                "text": text,
            }
        results["cases"][name] = entry
    (HERE / "results.json").write_text(json.dumps(results, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# P-0008's reference: tml-renderers on empty tool-call content",
        "",
        f"tml-renderers {results['tml_renderers']['version']} (module sha256 {results['tml_renderers']['module_sha256'][:12]}…), "
        f"effort {EFFORT}, sandbox return code {record['returncode']}, temp dir left {record['temp_dir_left']}, "
        f"write probe outside the temp dir created: {record['probe_outside_written']}.",
        "",
        "| Tool-call turn's content | Renderer | Empty text messages: renderer / HF template / TRL vision row | Renderer = HF template (text, ids) | Renderer = TRL vision row |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, e in results["cases"].items():
        rr = e["renderer"]
        if "error" in rr:
            lines.append(f"| {name} | raises `{rr['error'][:120]}` | – / {e['hf_template_empty_text_messages']} / {e['trl_vision_row_empty_text_messages']} | – | – |")
        else:
            lines.append(
                f"| {name} | renders | {rr['empty_text_messages']} / {e['hf_template_empty_text_messages']} / "
                f"{e['trl_vision_row_empty_text_messages']} | {rr['equals_hf_template_text']}, {rr['equals_hf_template_ids']} | "
                f"{rr['equals_trl_vision_row_text']} |"
            )
    (HERE / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
