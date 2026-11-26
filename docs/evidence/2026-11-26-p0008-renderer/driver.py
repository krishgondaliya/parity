"""Runs inside the sandbox (see compare.py): Thinking Machines' tml-renderers renders each case for SFT.

Input on stdin: {"effort": float | None, "cases": [{"name", "messages", "tools"}]}. Output on stdout: the installed version, the sha256 of
the compiled module, and per case the rendered token ids, their per-token loss weights and the decoded text, or the
exception the renderer raised. Nothing else is read or written.
"""

import hashlib
import json
import sys
from importlib import metadata
from pathlib import Path

import tml_renderers
from tml_renderers import chat, renderers


def tokens_of(example) -> list[int]:
    ids: list[int] = []
    for token_span in example.input_token_spans:
        span = token_span.span
        if not hasattr(span, "tokens"):
            raise TypeError(f"unexpected span type {type(span).__name__}")
        ids.extend(span.tokens)
    return ids


def main() -> None:
    payload = json.load(sys.stdin)
    cases = payload["cases"]
    module = Path(tml_renderers.__file__).parent / "_tml_renderers.abi3.so"
    out = {
        "version": metadata.version("tml-renderers"),
        "module_sha256": hashlib.sha256(module.read_bytes()).hexdigest(),
        "cases": {},
    }
    renderer = renderers.Inkling2(effort=payload.get("effort"))
    for case in cases:
        try:
            messages = chat.OpenAIMessage.from_openai_chat_dicts(case["messages"], case.get("tools"))
            examples = renderer.render_for_sft(messages)
            rendered = []
            for example in examples:
                ids = tokens_of(example)
                rendered.append(
                    {
                        "tokens": ids,
                        "weights": list(example.per_token_loss_weight_unshifted),
                        "text": renderer.tokenizer.decode(ids),
                    }
                )
            out["cases"][case["name"]] = {"examples": rendered}
        except Exception as e:  # the renderer's own refusal is a result
            out["cases"][case["name"]] = {"error": f"{type(e).__name__}: {e}"}
    json.dump(out, sys.stdout, ensure_ascii=False)


if __name__ == "__main__":
    main()
