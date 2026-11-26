# axolotl==0.21.1, transformers==5.20.1, Python 3.12, CPU. Tokenizer files only (no weights, no architecture), no
# Parity code. P-0002's mechanism on the Phase 0 targets (2026-11-26, for Gate A): through Axolotl's chat_template
# strategy with the documented settings (roles_to_train: [assistant], train_on_eos: turn, eot_tokens left at its
# default, the tokenizer's EOS), is an empty assistant turn's end token labeled?
# Three shapes per target: the empty_assistant fixture's (empty middle turn), the repro's (empty last turn), and a
# control with no empty turn. For each assistant turn it prints every closing token after the turn's header, and
# whether Axolotl labeled it.
import os

os.environ["AXOLOTL_DO_NOT_TRACK"] = os.environ["DO_NOT_TRACK"] = os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
from transformers import AutoTokenizer  # noqa: E402
from axolotl.prompt_strategies.chat_template import load  # noqa: E402
from axolotl.utils.dict import DictDefault  # noqa: E402

# repo -> (revision, the assistant header token(s), the tokens that close an assistant turn), read from each
# template's render of these conversations.
TARGETS = {
    "deepseek-ai/DeepSeek-V4.1-Flash-1017": (
        "08c8657f02291e11b6f6b63e0e16014f8e748518", ["<｜Assistant｜>"], ["<｜end▁of▁sentence｜>"]),
    "thinkingmachines/Inkling-2-Mini": (
        "7a67ae2314b44ef05c87fd1d4d311c58aaeb18f6", ["<|message_model|>"],
        ["<|end_message|>", "<|content_model_end_sampling|>"]),
    "tencent/Hy4-flash": (
        "eccd3123eec73f2040f4c9ed0de1883593470510", ["<｜hy_start:opensource｜>", "assistant"], ["<｜hy_end:opensource｜>"]),
}  # fmt: skip
U1, U2 = {"role": "user", "content": "Say hi."}, {"role": "user", "content": "Again."}
A = lambda text: {"role": "assistant", "content": text}  # noqa: E731
SHAPES = {
    "empty middle turn (the empty_assistant fixture)": [U1, A(""), U2, A("OK.")],
    "empty last turn (P-0002's repro)": [U1, A("Hi."), U2, A("")],
    "control, no empty turn": [U1, A("Hi."), U2, A("OK.")],
}
DS_CFG = {"type": "chat_template", "roles_to_train": ["assistant"], "train_on_eos": "turn"}

for repo, (revision, header, closers) in TARGETS.items():
    tok = AutoTokenizer.from_pretrained(repo, revision=revision)
    head = [tok.convert_tokens_to_ids(t) if t.startswith("<") else tok.encode(t, add_special_tokens=False)[0]
            for t in header]  # fmt: skip
    close = {tok.convert_tokens_to_ids(t): t for t in closers}
    strategy = load(tok, DictDefault({"train_on_inputs": False, "sequence_len": 4096}), DictDefault(DS_CFG))
    print(f"== {repo}@{revision[:7]} (EOS {tok.eos_token})")
    for name, messages in SHAPES.items():
        row = strategy.tokenize_prompt({"messages": messages})
        ids, labels = row["input_ids"], row["labels"]
        same = ids == tok.apply_chat_template(messages, tokenize=True, return_dict=True)["input_ids"]
        starts = [i for i in range(len(ids)) if ids[i : i + len(head)] == head]
        turns = []
        for n, s in enumerate(starts):
            stop = starts[n + 1] if n + 1 < len(starts) else len(ids)
            found = [(close[ids[i]], i, labels[i] != -100) for i in range(s, stop) if ids[i] in close]
            content = [m["content"] for m in messages if m["role"] == "assistant"][n]
            turns.append(f"turn {n + 1} ({content!r}): " + ", ".join(f"{t}@{i} labeled={lab}" for t, i, lab in found))
        print(f"  {name}: ids equal the template's render: {same}")
        for line in turns:
            print(f"    {line}")
