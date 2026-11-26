"""Runs DeepSeek V4.1's own encoding.py on cases read from stdin and writes the rendered prompts to stdout.

Standard library only. compare.py starts it inside the sandbox the human set (2026-11-25): `env -i` (an empty
environment, so no tokens), `python -I -S -B` (no user or site packages, no script directory on the path, no
bytecode), an empty temporary working directory, and no network (`unshare -rn`). It writes no files; compare.py
checks that afterwards.

    driver.py <path to encoding.py> <its expected sha256> <mapping>

mapping "documented": stdin is a JSON list of {"messages": [...], "thinking_mode": ...} already in encoding.py's
    documented message format; each goes to encode_messages().
mapping "loader": stdin is a JSON list of OpenAI-format cases ({"messages": [...], "tools": [...],
    "thinking_mode": ...}), read by encoding.py's own load_cases() and encoded by its own encode_case(), the path
    inference/generate.py takes for JSON input.

stdout: a JSON list, one {"prompt": str} or {"error": str} per case, in order.
"""

import hashlib
import importlib.util
import json
import sys


def main() -> None:
    sys.dont_write_bytecode = True  # also set by -B; never write a .pyc next to encoding.py
    path, expected, mapping = sys.argv[1], sys.argv[2], sys.argv[3]
    with open(path, "rb") as f:
        actual = hashlib.sha256(f.read()).hexdigest()
    if actual != expected:
        sys.exit(f"encoding.py sha256 mismatch: {actual} != {expected}")
    spec = importlib.util.spec_from_file_location("deepseek_v41_encoding", path)
    encoding = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(encoding)

    if mapping == "documented":
        cases = json.load(sys.stdin)
        run = [lambda c=c: encoding.encode_messages(c["messages"], thinking_mode=c["thinking_mode"]) for c in cases]
    elif mapping == "loader":
        cases = encoding.load_cases("/dev/stdin")  # the model's own loader, reading stdin
        run = [lambda c=c: encoding.encode_case(c, thinking_mode=c["thinking_mode"])[0] for c in cases]
    else:
        sys.exit(f"unknown mapping {mapping!r}")

    out = []
    for fn in run:
        try:
            out.append({"prompt": fn()})
        except Exception as exc:  # noqa: BLE001 - reported per case
            out.append({"error": f"{type(exc).__name__}: {exc}"})
    json.dump(out, sys.stdout, ensure_ascii=False)


if __name__ == "__main__":
    main()
