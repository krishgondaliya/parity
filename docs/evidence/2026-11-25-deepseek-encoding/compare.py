"""DeepSeek V4.1: does chat_template.jinja (a port) render what encoding/encoding.py (the model's own format) does?

Run from the repo root, in the core environment:
    uv run python docs/evidence/2026-11-25-deepseek-encoding/compare.py

Decided first (docs/DECISIONS.md, 2026-11-25): encoding.py is the model's own prompt format and the Jinja is a port,
so any disagreement is a candidate model-template finding against the Jinja.

How encoding.py runs (the human's conditions, 2026-11-25):
- the file is fetched at the pinned revision and must match its pinned sha256 (driver.py checks it again);
- a separate process: `unshare -rn` (no network: a fresh network namespace with only loopback), `env -i` (an empty
  environment, so no tokens and no .env), the base Python 3.12 with `-I -S -B` (no user or site packages, no script
  directory, no bytecode written), an empty temporary working directory;
- encoding.py sits alone in a read-only folder, outside the release's cache snapshot; the run fails if the working
  directory or that folder gains a file, and rule 6's snapshot check runs again afterwards;
- cases go in through stdin as JSON, rendered strings come out through stdout as JSON.

What is compared, for the 8 fixtures in both modes (thinking -> thinking_mode="thinking", non_thinking -> "chat";
reasoning effort and drop_thinking left at both sides' defaults):
- every full conversation (no generation prompt), and every generation-prompt render before each assistant turn
  (messages[:i] with add_generation_prompt=True on the Jinja side), since the Tier 1 reference uses both;
- the strings, exactly; and the token ids: the Jinja's through apply_chat_template, encoding.py's through
  `tokenizer.encode(prompt)`, as the model's inference/generate.py does (L156; `add_special_tokens` left at its
  default, True). Whether add_special_tokens=False gives the same ids is recorded too.

Two ways to hand a fixture's top-level tools to encoding.py (only `tool_call` has tools, and no system message):
- "documented": encoding/README.md defines tools on the system message, so they go on a new empty leading system
  message (what the Jinja itself does with a top-level `tools`, L119-L124). This is the primary comparison;
- "loader": encoding.py's own load_cases()/encode_case(), the path inference/generate.py takes for JSON cases with
  top-level tools. It attaches them to messages[0] whatever its role. Reported separately.

Writes results.json and summary.md next to this file.
"""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import tokenizers
import transformers
from huggingface_hub import hf_hub_download

from parity.fixtures import FIXTURE_IDS, load_fixture
from parity.references.template import model_chat_template, render_text, template_token_ids
from parity.release import fetch_release_files, load_release
from parity.serving import template_kwargs
from parity.targets import PHASE0_TARGETS

HERE = Path(__file__).resolve().parent
DEEPSEEK = next(t for t in PHASE0_TARGETS if t.repo_id == "deepseek-ai/DeepSeek-V4.1-Flash-1017")
ENCODING_FILE = "encoding/encoding.py"
ENCODING_SHA256 = "13231c592e958fb39e2af6001dae69d90e7a36b265702b9b390236fb387599e2"
THINKING_MODE = {"thinking": "thinking", "non_thinking": "chat"}


def fetch_encoding() -> Path:
    """encoding.py in a read-only folder of its own, never in the release's cache snapshot.

    Downloading it into the default cache put it in the same snapshot as the release's tokenizer files, and running
    it there left a .pyc next to it; rule 6's check in fetch_release_files then refused the snapshot (2026-11-26).
    So it is downloaded into a throwaway cache, copied into a fresh folder, and both are made read-only."""
    path = Path(hf_hub_download(DEEPSEEK.repo_id, ENCODING_FILE, revision=DEEPSEEK.revision, cache_dir=tempfile.mkdtemp()))
    data = path.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != ENCODING_SHA256:
        raise SystemExit(f"{ENCODING_FILE} at {DEEPSEEK.revision} has sha256 {actual}, pinned {ENCODING_SHA256}")
    code = Path(tempfile.mkdtemp(prefix="parity-deepseek-encoding-code-"))
    (code / "encoding.py").write_bytes(data)
    (code / "encoding.py").chmod(0o444)
    code.chmod(0o555)
    return code / "encoding.py"


def documented_messages(messages: list[dict], tools: list[dict] | None) -> list[dict]:
    """encoding/README.md: "Tools are defined on the system message via the tools field". A conversation with tools
    and no leading system message gets a new empty one to hold them."""
    messages = copy.deepcopy(messages)
    if tools:
        if messages and messages[0]["role"] == "system":
            messages[0] = {**messages[0], "tools": copy.deepcopy(tools)}
        else:
            messages = [{"role": "system", "content": "", "tools": copy.deepcopy(tools)}, *messages]
    return messages


def sandbox(encoding_path: Path, mapping: str, cases: list[dict]) -> tuple[list[dict], dict]:
    python = Path(sys.executable).resolve()  # the base interpreter behind the venv; with -I -S only its stdlib
    work = Path(tempfile.mkdtemp(prefix="parity-deepseek-encoding-"))
    command = [
        "unshare", "-rn",
        "/usr/bin/env", "-i",
        str(python), "-I", "-S", "-B",
        str(HERE / "driver.py"), str(encoding_path), ENCODING_SHA256, mapping,
    ]  # fmt: skip
    proc = subprocess.run(
        command, cwd=work, input=json.dumps(cases, ensure_ascii=False), capture_output=True, text=True, timeout=300
    )
    if proc.returncode != 0:
        raise SystemExit(f"sandbox failed ({proc.returncode}): {proc.stderr[-2000:]}")
    left = sorted(p.name for p in work.iterdir())
    code_folder = sorted(p.name for p in encoding_path.parent.iterdir())
    record = {
        "mapping": mapping,
        "command": ["unshare", "-rn", "/usr/bin/env", "-i", f"python{sys.version_info.major}.{sys.version_info.minor}",
                    "-I", "-S", "-B", "driver.py", ENCODING_FILE, ENCODING_SHA256, mapping],  # fmt: skip
        "python": sys.version.split()[0],
        "working_dir_left_empty": not left,
        "code_folder_unchanged": code_folder == ["encoding.py"],
        "stderr": proc.stderr[-2000:],
    }
    if left or code_folder != ["encoding.py"]:
        raise SystemExit(f"the sandbox wrote files: working dir {left}, code folder {code_folder}")
    return json.loads(proc.stdout), record


def first_difference(a: str, b: str) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def main() -> None:
    release = load_release(DEEPSEEK)
    tokenizer = release.tokenizer
    template = model_chat_template(tokenizer)
    encoding_path = fetch_encoding()

    plans = []  # (key, fixture, mode, render, messages, tools, add_generation_prompt)
    for fid in FIXTURE_IDS:
        conv = load_fixture(fid)
        for mode in THINKING_MODE:
            plans.append((f"{mode}/{fid}/full", conv, mode, "full", conv.messages, conv.tools, False))
            for i, m in enumerate(conv.messages):
                if m["role"] == "assistant":
                    plans.append((f"{mode}/{fid}/prompt@{i}", conv, mode, f"prompt@{i}", conv.messages[:i], conv.tools, True))

    documented_cases = [
        {"messages": documented_messages(msgs, tools), "thinking_mode": THINKING_MODE[mode]}
        for _, _, mode, _, msgs, tools, _ in plans
    ]
    loader_cases = [
        {"messages": copy.deepcopy(msgs), **({"tools": copy.deepcopy(tools)} if tools else {}), "thinking_mode": THINKING_MODE[mode]}
        for _, _, mode, _, msgs, tools, _ in plans
    ]  # fmt: skip
    outputs = {}
    runs = []
    for mapping, cases in (("documented", documented_cases), ("loader", loader_cases)):
        outputs[mapping], record = sandbox(encoding_path, mapping, cases)
        runs.append(record)

    rows = []
    for n, (key, conv, mode, render, msgs, tools, gp) in enumerate(plans):
        kwargs = template_kwargs(release, mode)
        jinja_text = render_text(tokenizer, msgs, tools, template, add_generation_prompt=gp, **kwargs)
        jinja_ids = template_token_ids(tokenizer, conv.with_messages(msgs), template, add_generation_prompt=gp, **kwargs)
        row = {"key": key, "fixture": conv.id, "mode": mode, "render": render}
        for mapping in ("documented", "loader"):
            out = outputs[mapping][n]
            if "error" in out:
                row[mapping] = {"error": out["error"]}
                continue
            text = out["prompt"]
            ids = tokenizer.encode(text)  # as inference/generate.py L156
            at = first_difference(jinja_text, text)
            entry = {
                "text_equal": at is None,
                "ids_equal": list(jinja_ids) == ids,
                "add_special_tokens_false_same_ids": tokenizer.encode(text, add_special_tokens=False) == ids,
                "sha256": hashlib.sha256(text.encode()).hexdigest()[:16],
            }
            if at is not None:
                entry["first_char_difference"] = at
                entry["jinja_context"] = jinja_text[max(0, at - 80) : at + 120]
                entry["encoding_context"] = text[max(0, at - 80) : at + 120]
                entry["jinja_text"] = jinja_text
                entry["encoding_text"] = text
            row[mapping] = entry
        row["jinja_sha256"] = hashlib.sha256(jinja_text.encode()).hexdigest()[:16]
        rows.append(row)

    results = {
        "target": DEEPSEEK.spec,
        "encoding_file": ENCODING_FILE,
        "encoding_sha256": ENCODING_SHA256,
        "jinja_sha256": hashlib.sha256(template.encode()).hexdigest(),
        "tokenization": "tokenizer.encode(prompt), add_special_tokens left at its default (True), as "
        "inference/generate.py L156; the prompt string already starts with <｜begin▁of▁sentence｜>",
        "versions": {"transformers": transformers.__version__, "tokenizers": tokenizers.__version__},
        "sandbox": runs,
        "rows": rows,
    }
    fetch_release_files(DEEPSEEK)  # rule 6: the release's snapshot still holds tokenizer and config files only
    (HERE / "results.json").write_text(json.dumps(results, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (HERE / "summary.md").write_text(render_summary(results), encoding="utf-8")
    for mapping in ("documented", "loader"):
        bad = [r["key"] for r in rows if not (r[mapping].get("text_equal") and r[mapping].get("ids_equal"))]
        print(f"{mapping}: {len(rows) - len(bad)} of {len(rows)} renders agree; disagree: {bad}")


def render_summary(results: dict) -> str:
    rows = results["rows"]
    out = [
        "# DeepSeek V4.1: chat_template.jinja (a port) against encoding.py (the model's own format)",
        "",
        f"Generated by `compare.py`. Target `{results['target']}`; `{results['encoding_file']}` sha256 "
        f"`{results['encoding_sha256'][:16]}…`; `chat_template.jinja` sha256 `{results['jinja_sha256'][:16]}…`; "
        f"transformers {results['versions']['transformers']}, tokenizers {results['versions']['tokenizers']}.",
        "",
        f"- Tokenization of encoding.py's output: {results['tokenization']}.",
    ]
    for run in results["sandbox"]:
        out.append(
            f"- Sandbox ({run['mapping']}): `{' '.join(run['command'][:6])} …`, Python {run['python']}, working "
            f"directory left empty: {run['working_dir_left_empty']}."
        )
    out += ["", "| render | mode | documented mapping | loader mapping |", "| --- | --- | --- | --- |"]

    def cell(e: dict) -> str:
        if "error" in e:
            return f"ERROR: {e['error'][:80]}"
        if e["text_equal"] and e["ids_equal"]:
            return "same"
        return f"DIFFERS (text equal: {e['text_equal']}, ids equal: {e['ids_equal']})"

    for r in rows:
        out.append(f"| `{r['fixture']}` {r['render']} | {r['mode']} | {cell(r['documented'])} | {cell(r['loader'])} |")
    for mapping in ("documented", "loader"):
        bad = [r for r in rows if not (r[mapping].get("text_equal") and r[mapping].get("ids_equal"))]
        out += ["", f"## {mapping} mapping: {len(rows) - len(bad)} of {len(rows)} renders agree", ""]
        for r in bad:
            e = r[mapping]
            if "error" in e:
                out.append(f"- `{r['key']}`: error {e['error']}")
                continue
            out.append(f"- `{r['key']}`, first difference at character {e.get('first_char_difference')}:")
            out.append(f"  - Jinja: `{e.get('jinja_context', '')!r}`")
            out.append(f"  - encoding.py: `{e.get('encoding_context', '')!r}`")
            out.append(
                f"  - tool schema (`## Tools`) in the Jinja's render: {'## Tools' in e.get('jinja_text', '')}; "
                f"in encoding.py's: {'## Tools' in e.get('encoding_text', '')}"
            )
    special = all(r[m].get("add_special_tokens_false_same_ids", True) for r in rows for m in ("documented", "loader"))
    out += ["", f"- `add_special_tokens=False` gives the same ids as the default on every render: {special}.", ""]
    return "\n".join(out)


if __name__ == "__main__":
    main()
