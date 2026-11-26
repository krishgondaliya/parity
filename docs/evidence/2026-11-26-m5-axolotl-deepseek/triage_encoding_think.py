"""Triage of M5 Axolotl x DeepSeek V4.1: in thinking mode, is </think> something the model generates?

Run from the repo root, in the core environment:
    uv run python docs/evidence/2026-11-26-m5-axolotl-deepseek/triage_encoding_think.py

It asks the model's own format, encoding/encoding.py (docs/DECISIONS.md, 2026-11-25: the model's own prompt format),
run in the same sandbox as the encoding comparison (docs/evidence/2026-11-25-deepseek-encoding/compare.py: unshare
-rn, env -i, python -I -S -B, an empty working dir, stdin/stdout). For each conversation, the prompt is the encoding
of the messages before the last assistant turn (encode_messages adds the generation prompt after a user turn), and the
full text is the encoding with that turn. Whatever the full text adds after the prompt is what the model generates.
Writes triage_encoding_think.json next to this file.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
COMPARE = HERE.parents[0] / "2026-11-25-deepseek-encoding" / "compare.py"

U = {"role": "user", "content": "Name the three primary colors."}
A = {"role": "assistant", "content": "Red, yellow and blue."}
AR = {"role": "assistant", "content": "Red, yellow and blue.", "reasoning_content": "Classic color theory."}
CASES = {
    "one turn, no reasoning": [U, A],
    "one turn, with reasoning": [U, AR],
    "after a history turn": [U, {"role": "assistant", "content": "Sure."}, {"role": "user", "content": "Again."}, A],
}


def main() -> None:
    spec = importlib.util.spec_from_file_location("deepseek_encoding_compare", COMPARE)
    compare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compare)
    encoding_path = compare.fetch_encoding()
    cases = []
    for messages in CASES.values():
        cases.append({"messages": messages[:-1], "thinking_mode": "thinking"})
        cases.append({"messages": messages, "thinking_mode": "thinking"})
    outputs, record = compare.sandbox(encoding_path, "documented", cases)
    out = {"sandbox": record, "cases": {}}
    for n, name in enumerate(CASES):
        prompt, full = outputs[2 * n]["prompt"], outputs[2 * n + 1]["prompt"]
        out["cases"][name] = {
            "prompt_ends": prompt[-40:],
            "full_starts_with_prompt": full.startswith(prompt),
            "generated": full[len(prompt):] if full.startswith(prompt) else None,
        }
    (HERE / "triage_encoding_think.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for name, r in out["cases"].items():
        print(f"{name}: prompt ends {r['prompt_ends'][-20:]!r}; prefix of the full text: {r['full_starts_with_prompt']}; "
              f"the model generates {r['generated']!r}")  # fmt: skip


if __name__ == "__main__":
    main()
