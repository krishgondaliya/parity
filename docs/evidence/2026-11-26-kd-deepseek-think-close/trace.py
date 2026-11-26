"""The human's condition for a known-difference entry on the 9 Axolotl x DeepSeek V4.1 `</think>` cells: same Axolotl
code path and same documented test as KD-0002. This traces the code path.

Run from the repo root, in the Axolotl environment:
    envs/axolotl-0.21.1/.venv/bin/python docs/evidence/2026-11-26-kd-deepseek-think-close/trace.py

It runs our Axolotl adapter (the full `load_cfg` + `load_datasets` path, in-process, dataset_num_proc 1, as in M4 and
M5) in thinking mode on Qwen3-0.6B (KD-0002's model) and DeepSeek V4.1, and wraps
`ChatTemplateStrategy.find_turn`, the call whose span `_tokenize_single_prompt` labels
(prompt_strategies/chat_template.py L728-L764 at v0.21.1). For every assistant turn it records:
  - the arguments the labeling code passed (content_only, reasoning_only), and whether the turn has reasoning
    as Axolotl sees it (after its field filtering);
  - the span find_turn returned, and the tokens between the end of the turn's generation prompt (the model's own
    render with add_generation_prompt=True) and the span's start: the template structure the model generates that
    gets no loss;
  - whether the row's labels are exactly that span (plus the end token), so nothing else changed them.
Fixtures: single_turn, multi_turn, system_prompt, long_answer and non_ascii (the 9 cells' fixtures, no reasoning),
and reasoning (P-0007's, for contrast). Writes trace.json next to this file.

Axolotl tokenizes in a spawned worker process, which re-imports this file but not the patch a function would make in
the main process. So the wrapper is installed at import time (in the main process and in every worker), and each call
is appended as a JSON line to the file named by PARITY_TRACE_FILE, which workers inherit.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from parity.adapters.axolotl import AxolotlAdapter
from parity.adapters.base import TrainCfg
from parity.fixtures import load_fixture
from parity.release import load_release
from parity.serving import template_kwargs
from parity.targets import CALIBRATION, PHASE0_TARGETS

HERE = Path(__file__).resolve().parent
FIXTURES = ("single_turn", "multi_turn", "system_prompt", "long_answer", "non_ascii", "reasoning")


def _install_recorder() -> None:
    from axolotl.prompt_strategies.chat_template import ChatTemplateStrategy

    original = ChatTemplateStrategy.find_turn

    def recording_find_turn(self, turns, turn_idx, tools=None, content_only=False, reasoning_only=False, locator=None):
        span = original(self, turns, turn_idx, tools, content_only, reasoning_only, locator)
        turn = turns[turn_idx]
        record = {
            "pid": os.getpid(),
            "turn_idx": turn_idx,
            "role": turn.get("role"),
            "content_only": content_only,
            "reasoning_only": reasoning_only,
            "has_reasoning": turn.get(self.prompter.template_thinking_key) is not None,
            "locator": locator is not None,
            "span": list(span),
        }
        with open(os.environ["PARITY_TRACE_FILE"], "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        return span

    ChatTemplateStrategy.find_turn = recording_find_turn


_install_recorder()


def main() -> None:
    trace_dir = Path(tempfile.mkdtemp(prefix="p0007-kd-trace-"))
    deepseek = next(t for t in PHASE0_TARGETS if t.repo_id.startswith("deepseek-ai/"))
    out: dict = {}
    for pin in (CALIBRATION, deepseek):
        release = load_release(pin)
        tok = release.tokenizer
        adapter = AxolotlAdapter(release)
        kwargs = template_kwargs(release, "thinking")
        for fid in FIXTURES:
            conv = load_fixture(fid)
            trace_file = trace_dir / f"{pin.repo_id.replace('/', '__')}-{fid}.jsonl"
            trace_file.write_text("", encoding="utf-8")
            os.environ["PARITY_TRACE_FILE"] = str(trace_file)
            enc = adapter.encode(conv, TrainCfg(serving_mode="thinking"))
            calls = [json.loads(line) for line in trace_file.read_text(encoding="utf-8").splitlines()]
            ids, labels = list(enc.input_ids), list(enc.labels)
            assistant = [k for k, m in enumerate(conv.messages) if m["role"] == "assistant"]
            turns = []
            labeled_from_spans: set[int] = set()
            for call in (c for c in calls if c["role"] == "assistant"):
                k = assistant[len(turns)]
                prompt = tok.apply_chat_template(
                    conv.messages[:k], tools=conv.tools, add_generation_prompt=True, tokenize=False, **kwargs
                )
                prompt_len = len(tok(prompt, add_special_tokens=False)["input_ids"])
                start, end = call["span"]
                labeled_from_spans.update(range(start, end))
                turns.append(
                    {
                        **call,
                        "untrained_generated_structure": tok.decode(ids[prompt_len:start]) if start >= prompt_len else None,
                        "first_labeled_token": tok.decode([ids[start]]) if start >= 0 else None,
                    }
                )
            labeled = {i for i, lab in enumerate(labels) if lab != -100}
            out[f"{pin.repo_id} / {fid}"] = {
                "template_kwargs": kwargs,
                "worker_pids": sorted({c["pid"] for c in calls} - {os.getpid()}),
                "turns": turns,
                "labels_outside_spans": [tok.decode([ids[i]]) for i in sorted(labeled - labeled_from_spans)],
                "span_tokens_unlabeled": len(labeled_from_spans - labeled),
            }
    (HERE / "trace.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for key, r in out.items():
        for t in r["turns"]:
            print(f"{key} turn {t['turn_idx']}: content_only={t['content_only']} reasoning_only={t['reasoning_only']} "
                  f"has_reasoning={t['has_reasoning']} locator={t['locator']} | untrained generated structure "
                  f"{t['untrained_generated_structure']!r} | first labeled {t['first_labeled_token']!r}")  # fmt: skip
        print(f"   labels outside the spans: {r['labels_outside_spans']}; span tokens unlabeled: {r['span_tokens_unlabeled']}")


if __name__ == "__main__":
    main()
