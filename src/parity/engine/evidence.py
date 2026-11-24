"""Evidence helpers: where two token sequences first differ, shown as decoded tokens."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from transformers import PreTrainedTokenizerBase

WINDOW = 8  # decoded tokens shown on each side (blueprint: "8 decoded tokens on each side")


def decode_each(tokenizer: PreTrainedTokenizerBase, ids: Sequence[int]) -> list[str]:
    """Decode token by token, keeping special tokens, so evidence shows exact boundaries."""
    return [tokenizer.decode([i], skip_special_tokens=False, clean_up_tokenization_spaces=False) for i in ids]


def first_divergence(expected: Sequence[int], got: Sequence[int]) -> int | None:
    """Index of the first position where the sequences differ, or None if they are equal.

    If one is a strict prefix of the other, that is the length of the shorter one.
    """
    for i, (a, b) in enumerate(zip(expected, got)):
        if a != b:
            return i
    if len(expected) != len(got):
        return min(len(expected), len(got))
    return None


def token_evidence(
    tokenizer: PreTrainedTokenizerBase,
    expected: Sequence[int],
    got: Sequence[int],
    at: int | None = None,
) -> dict[str, Any]:
    """The first divergence plus WINDOW decoded tokens from each side, and the WINDOW tokens before it."""
    i = first_divergence(expected, got) if at is None else at
    if i is None:
        return {"first_divergence": None, "expected_len": len(expected), "got_len": len(got)}
    return {
        "first_divergence": i,
        "context_before": decode_each(tokenizer, expected[max(0, i - WINDOW) : i]),
        "expected": decode_each(tokenizer, expected[i : i + WINDOW]),
        "got": decode_each(tokenizer, got[i : i + WINDOW]),
        "expected_len": len(expected),
        "got_len": len(got),
    }
