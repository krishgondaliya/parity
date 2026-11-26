"""Triage of M5 Axolotl x DeepSeek V4.1, t1.assistant_mask: the last turn's </think> is untrained. Why is DeepSeek
different from Qwen3, where Axolotl trains </think> and leaves only the opening <think> untrained (KD-0002/KD-0003)?

Run from the repo root, as a file (Axolotl's dataset loading starts worker processes that re-import it):
    envs/axolotl-0.21.1/.venv/bin/python docs/evidence/2026-11-26-m5-axolotl-deepseek/triage_think_close.py

Through the adapter (the full `axolotl preprocess` path of the M5 run), for single_turn and multi_turn on both
models, it prints each token of the last assistant turn with Axolotl's label (trained or not) and our reference's
zone. Writes triage_think_close.json next to this file.
"""

from __future__ import annotations

import json
from pathlib import Path

from parity.adapters.axolotl import AxolotlAdapter  # imports Axolotl first, as the adapter does
from parity.adapters.base import LABEL_IGNORE, TrainCfg
from parity.checks.common import official_reference
from parity.fixtures import load_fixture
from parity.references.template import Zone
from parity.release import load_release
from parity.targets import CALIBRATION, PHASE0_TARGETS

HERE = Path(__file__).resolve().parent
ZONE = {Zone.MUST_TRAIN: "must-train", Zone.MUST_NOT_TRAIN: "prompt side", Zone.NEUTRAL: "neutral"}


def last_turn(release, adapter, fixture_id: str) -> list[dict]:
    conv = load_fixture(fixture_id)
    row = adapter.encode(conv, TrainCfg())
    ref = official_reference(release, conv)
    assert list(row.input_ids) == list(ref.input_ids), "rows differ from the template; see template parity"
    start = ref.spans[-1].token_message_start or ref.spans[-1].token_start - 3
    tokens = release.tokenizer.convert_ids_to_tokens(list(row.input_ids))
    return [
        {"token": tokens[i], "axolotl_trained": row.labels[i] != LABEL_IGNORE, "reference": ZONE[ref.zones[i]]}
        for i in range(max(0, start), len(tokens))
    ]


def main() -> None:
    deepseek = next(t for t in PHASE0_TARGETS if t.repo_id == "deepseek-ai/DeepSeek-V4.1-Flash-1017")
    out = {}
    for name, pin in (("qwen3", CALIBRATION), ("deepseek", deepseek)):
        release = load_release(pin)
        adapter = AxolotlAdapter(release)
        out[name] = {fid: last_turn(release, adapter, fid) for fid in ("single_turn", "multi_turn")}
    (HERE / "triage_think_close.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for name, fixtures in out.items():
        for fid, tokens in fixtures.items():
            shown = " ".join(f"{t['token']}[{'T' if t['axolotl_trained'] else '.'}/{t['reference'][0]}]" for t in tokens[:14])
            print(f"{name} {fid}: {shown}")


if __name__ == "__main__":
    main()
