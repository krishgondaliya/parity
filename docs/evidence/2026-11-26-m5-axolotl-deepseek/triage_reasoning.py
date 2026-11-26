"""Triage of M5 Axolotl x DeepSeek V4.1: why does the reasoning fixture's row drop the last turn's reasoning, when
the same strategy loaded from a bare dataset config keeps it? And how does Axolotl's prompter differ between
DeepSeek V4.1 (its deepseek_v4 stand-in) and Qwen3?

Run from the repo root, as a file (Axolotl's dataset loading starts worker processes that re-import it):
    envs/axolotl-0.21.1/.venv/bin/python docs/evidence/2026-11-26-m5-axolotl-deepseek/triage_reasoning.py

For DeepSeek V4.1 and Qwen3, through the adapter (the full `axolotl preprocess` path the M5 run took), it prints the
dataset config Axolotl validated, the prompter Axolotl's own loader builds from it (`load_dataset_with_strategy`'s
strategy), the message thread the strategy hands the template, and whether the prepared row equals the template's
render. Writes triage_reasoning.json next to this file.
"""

from __future__ import annotations

import json
from pathlib import Path

from parity.adapters.axolotl import AxolotlAdapter, load_cfg  # imports Axolotl first, as the adapter does
from parity.adapters.base import TrainCfg
from parity.fixtures import load_fixture
from parity.release import load_release
from parity.targets import CALIBRATION, PHASE0_TARGETS

HERE = Path(__file__).resolve().parent


def probe(release_pin) -> dict:
    from axolotl.prompt_strategies.chat_template import load as load_strategy
    from axolotl.loaders import load_tokenizer

    release = load_release(release_pin)
    adapter = AxolotlAdapter(release)
    conv = load_fixture("reasoning")
    cfg = load_cfg(str(adapter._config(adapter._rows([conv]), TrainCfg(), 1)))
    ds_cfg = cfg.datasets[0]
    strategy = load_strategy(load_tokenizer(cfg), cfg, ds_cfg)
    prompter = strategy.prompter
    thread = strategy.get_conversation_thread({"messages": conv.messages})
    row = adapter.encode(conv, TrainCfg())
    rendered = release.tokenizer.apply_chat_template(conv.messages, tokenize=True, return_dict=True)["input_ids"]
    return {
        "release": release_pin.spec,
        "model_config_type": cfg.model_config_type,
        "validated": {k: str(ds_cfg.get(k)) for k in ("message_property_mappings", "field_thinking",
                                                       "template_thinking_key", "split_thinking")},  # fmt: skip
        "prompter": {
            "message_property_mappings": dict(prompter.message_property_mappings),
            "field_thinking": getattr(prompter, "field_thinking", None),
            "template_thinking_key": getattr(prompter, "template_thinking_key", None),
        },
        "last_turn_keys": sorted(thread[-1]),
        "row_equals_template": list(row.input_ids) == list(rendered),
        "row_tokens": len(row.input_ids),
        "template_tokens": len(rendered),
    }


def main() -> None:
    deepseek = next(t for t in PHASE0_TARGETS if t.repo_id == "deepseek-ai/DeepSeek-V4.1-Flash-1017")
    out = {"deepseek": probe(deepseek), "qwen3": probe(CALIBRATION)}
    (HERE / "triage_reasoning.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for name, r in out.items():
        print(f"== {name}: {r['model_config_type']}")
        print(f"   validated: {r['validated']}")
        print(f"   prompter:  {r['prompter']}")
        print(f"   last turn keys: {r['last_turn_keys']}; row == template: {r['row_equals_template']} "
              f"({r['row_tokens']} vs {r['template_tokens']} tokens)")  # fmt: skip


if __name__ == "__main__":
    main()
