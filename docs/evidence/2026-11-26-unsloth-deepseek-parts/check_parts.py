"""Unsloth x DeepSeek V4.1: do the template's own role markers delimit the turns, and what would they train?

Run in the core environment (our reference only; no framework, nothing of Unsloth's runs):
    uv run python docs/evidence/2026-11-26-unsloth-deepseek-parts/check_parts.py

The recipe (the human's decision, 2026-11-26): Unsloth documents no training recipe for DeepSeek V4.1, so its
general SFT docs apply, and `train_on_responses_only` takes the template's own role markers as its parts:
instruction_part = "<｜User｜>", response_part = "<｜Assistant｜>". Sources: encoding/README.md at the pinned revision
(its special-token table: "<｜User｜> | User turn prefix", "<｜Assistant｜> | Assistant turn prefix") and
chat_template.jinja (which emits them). Unsloth's own examples for other templates use each template's user and
assistant headers (lora-hyperparameters-guide).

For each fixture and mode, against our reference render (parity.checks.common.official_reference):
1. each part is one token the tokenizer declares, and appears in no message's content;
2. "<｜Assistant｜>" occurs once per assistant message, and "<｜User｜>" once per user or tool message (DeepSeek
   renders a tool result as a user turn, `<｜User｜><tool_result>...`);
3. the region the parts mark as response (from the token after each "<｜Assistant｜>" up to the next "<｜User｜>", or the
   end), set against our reference's zones: which must-train tokens fall outside it (would go untrained), and which
   must-not-train tokens fall inside it (would be trained). That is what the assistant-mask check will judge.
Writes results.json and summary.md next to this file.
"""

from __future__ import annotations

import json
from pathlib import Path

from parity.checks.common import official_reference, official_template
from parity.fixtures import FIXTURE_IDS, load_fixture
from parity.references.template import TemplateReferenceError, Zone, template_token_ids
from parity.release import load_release
from parity.serving import SERVING_MODES, not_applicable_in_mode, template_kwargs
from parity.targets import PHASE0_TARGETS

HERE = Path(__file__).resolve().parent
INSTRUCTION, RESPONSE = "<｜User｜>", "<｜Assistant｜>"


def strings(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in strings(v)]
    if isinstance(value, list):
        return [s for v in value for s in strings(v)]
    return []


def main() -> None:
    target = next(t for t in PHASE0_TARGETS if t.repo_id == "deepseek-ai/DeepSeek-V4.1-Flash-1017")
    release = load_release(target)
    tok = release.tokenizer
    ins, res = tok.convert_tokens_to_ids(INSTRUCTION), tok.convert_tokens_to_ids(RESPONSE)
    declared = {t.content for t in tok.added_tokens_decoder.values()}
    rows = []
    for mode in SERVING_MODES:
        for fid in FIXTURE_IDS:
            conv = load_fixture(fid)
            row: dict = {"fixture": fid, "mode": mode}
            if not_applicable_in_mode(conv, mode):
                row["status"] = "n/a (mode rule)"
                rows.append(row)
                continue
            content = [s for m in conv.messages for s in strings(m)]
            row["parts_declared_tokens"] = INSTRUCTION in declared and RESPONSE in declared
            row["parts_in_content"] = any(INSTRUCTION in s or RESPONSE in s for s in content)
            ids = template_token_ids(tok, conv, official_template(release), **template_kwargs(release, mode))
            row["assistant_markers"] = ids.count(res)
            row["assistant_messages"] = sum(m["role"] == "assistant" for m in conv.messages)
            row["user_markers"] = ids.count(ins)
            row["user_or_tool_messages"] = sum(m["role"] in ("user", "tool") for m in conv.messages)
            response = [False] * len(ids)
            inside = False
            for i, t in enumerate(ids):
                if t == res:
                    inside = True
                    continue
                if t == ins:
                    inside = False
                response[i] = inside
            try:
                ref = official_reference(release, conv, mode)
            except TemplateReferenceError as exc:
                row["status"] = "reference refuses"
                row["reason"] = str(exc)[:200]
                rows.append(row)
                continue
            assert list(ref.input_ids) == ids
            missed = [i for i, z in enumerate(ref.zones) if z == Zone.MUST_TRAIN and not response[i]]
            extra = [i for i, z in enumerate(ref.zones) if z == Zone.MUST_NOT_TRAIN and response[i]]
            row["status"] = "computed"
            row["must_train_outside_response"] = [(i, tok.convert_ids_to_tokens(ids[i])) for i in missed]
            row["must_not_train_inside_response"] = [(i, tok.convert_ids_to_tokens(ids[i])) for i in extra]
            rows.append(row)
    (HERE / "results.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# Unsloth x DeepSeek V4.1: the template's role markers as train_on_responses_only parts, on our reference",
        "",
        f"Generated by `check_parts.py` (core environment, our reference only). Parts: instruction `{INSTRUCTION}`, "
        f"response `{RESPONSE}`.",
        "",
        "| mode | fixture | markers (assistant / user+tool) | must-train outside the response region | must-not-train inside it |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        if r["status"] != "computed":
            lines.append(f"| {r['mode']} | `{r['fixture']}` | {r['status']} | | |")
            continue
        marks = f"{r['assistant_markers']}/{r['assistant_messages']}, {r['user_markers']}/{r['user_or_tool_messages']}"
        a = ", ".join(f"{t}@{i}" for i, t in r["must_train_outside_response"]) or "none"
        b = ", ".join(f"{t}@{i}" for i, t in r["must_not_train_inside_response"]) or "none"
        lines.append(f"| {r['mode']} | `{r['fixture']}` | {marks} | {a} | {b} |")
    ok = all(r.get("parts_declared_tokens", True) and not r.get("parts_in_content", False) for r in rows)
    lines += ["", f"- Both parts are declared tokens and appear in no message's content, on every fixture: {ok}.", ""]
    (HERE / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[4:]))


if __name__ == "__main__":
    main()
