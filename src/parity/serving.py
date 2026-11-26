"""Serving modes: the chat-template arguments that put a model in thinking or non-thinking mode.

Why it matters: some tokens change zone with the mode. Qwen3's empty think block at the start of the last
assistant turn is model output in thinking mode (the generation prompt ends at "<|im_start|>assistant\\n"), but a
generation-prompt pre-fill in non-thinking mode, so prompt side. The reference renders every conversation with the
mode's template arguments, so a mode-dependent token gets its zone from the declared mode.

The mode is set in `TrainCfg.serving_mode`, defaults to the model's documented default, and is recorded in every
verdict. Decided 2026-11-24, after the Axolotl results on Qwen3 (P-0001); see docs/DECISIONS.md.

What the framework is told (`framework_mode`): nothing in the model's default mode, so the primary results encode
with the framework's own defaults, as its users do by default. In any other mode the adapter gives the framework its
documented setting for the mode (the model's chat-template arguments below), or raises ModeNotConfigurable when the
framework documents no way to set it. Revised 2026-11-24 after P-0003: a framework's output depends on the mode,
so judging default-mode rows against a non-default reference was a harness error.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from parity.references.template import TemplateReferenceError
from parity.release import LoadedRelease

SERVING_MODES = ("thinking", "non_thinking")


class ServingModeError(TemplateReferenceError):
    """The model documents no way to serve in the requested mode: the reference can't be computed."""


@dataclass(frozen=True)
class ModeSpec:
    """A model's serving modes. `citation_level` says where the default comes from: "model card" when the card at
    the pinned revision states it, "template only" when only the chat template sets it. `sources` are permalinks at
    the pinned revision (M5 prep, 2026-11-25: every Phase 0 target's default mode is cited before any run)."""

    default: str
    kwargs: dict[str, dict[str, Any] | None]  # mode -> chat-template arguments; None if the model has no such mode
    evidence: str
    citation_level: str = ""
    sources: tuple[str, ...] = ()


MODE_SPECS: dict[str, ModeSpec] = {
    "Qwen/Qwen3-0.6B": ModeSpec(
        default="thinking",
        kwargs={"thinking": {"enable_thinking": True}, "non_thinking": {"enable_thinking": False}},
        evidence=(
            "Qwen3's template thinks unless enable_thinking is false; with it false the generation prompt pre-fills "
            "an empty think block. The model card says thinking is enabled by default (read by the skeptic for "
            "P-0001; findings/P-0001/record.md)."
        ),
    ),
    "deepseek-ai/DeepSeek-V4.1-Flash-1017": ModeSpec(
        default="thinking",
        kwargs={"thinking": {"thinking": True}, "non_thinking": {"thinking": False}},
        evidence=(
            "The model card documents no default. It says the release has no Jinja template and points to "
            "encoding/encoding.py as the prompt-format reference, whose encode_messages takes thinking_mode as a "
            "required argument ('thinking' or 'chat'; its quick start uses 'thinking'). The chat_template.jinja at "
            "this revision was added after the release (commit 08c8657, 'Add chat template (#11)', 2026-11-10) and "
            "says it is 'ported from encoding/encoding.py'; it defaults thinking_mode to 'thinking' (L2, L10-L16), "
            "and thinking=False selects 'chat'."
        ),
        citation_level="template only",
        sources=(
            "https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/08c8657f02291e11b6f6b63e0e16014f8e748518/README.md",
            "https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/08c8657f02291e11b6f6b63e0e16014f8e748518/encoding/README.md",
            "https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/08c8657f02291e11b6f6b63e0e16014f8e748518/encoding/encoding.py#L810-L816",
            "https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/08c8657f02291e11b6f6b63e0e16014f8e748518/chat_template.jinja#L1-L20",
        ),
    ),
    "tencent/Hy4-flash": ModeSpec(
        default="thinking",
        kwargs={"thinking": {"reasoning_effort": "high"}, "non_thinking": {"reasoning_effort": "no_think"}},
        evidence=(
            "The model card: 'reasoning_effort supports \"high\" (default, thinking enabled) and \"no_think\" "
            "(direct responses).' The template agrees: reasoning_effort defaults to 'high' and accepts only 'high' "
            "and 'no_think' (L38-L45)."
        ),
        citation_level="model card",
        sources=(
            "https://huggingface.co/tencent/Hy4-flash/blob/eccd3123eec73f2040f4c9ed0de1883593470510/README.md#L242",
            "https://huggingface.co/tencent/Hy4-flash/blob/eccd3123eec73f2040f4c9ed0de1883593470510/chat_template.jinja#L38-L45",
        ),
    ),
    "thinkingmachines/Inkling-2-Mini": ModeSpec(
        default="thinking",
        kwargs={"thinking": {}, "non_thinking": None},
        evidence=(
            "The model card: 'Setting reasoning effort to its minimum does not fully disable reasoning.' So the "
            "model documents no non-thinking mode. The template takes a numeric reasoning_effort in [0, 0.99] "
            "(default 0.9, L4-L16), emitted as a system line."
        ),
        citation_level="model card",
        sources=(
            "https://huggingface.co/thinkingmachines/Inkling-2-Mini/blob/7a67ae2314b44ef05c87fd1d4d311c58aaeb18f6/README.md#L202",
            "https://huggingface.co/thinkingmachines/Inkling-2-Mini/blob/7a67ae2314b44ef05c87fd1d4d311c58aaeb18f6/chat_template.jinja#L4-L16",
        ),
    ),
}
UNKNOWN = ModeSpec(
    default="thinking",
    kwargs={"thinking": {}, "non_thinking": None},
    evidence="No documented mode switch for this model: renders with the template's defaults.",
)


def mode_spec(release: LoadedRelease) -> ModeSpec:
    return MODE_SPECS.get(release.release.repo_id, UNKNOWN)


def resolve_mode(release: LoadedRelease, mode: str | None) -> str:
    """The requested mode, or the model's documented default when none is requested."""
    if mode is None:
        return mode_spec(release).default
    if mode not in SERVING_MODES:
        raise ValueError(f"serving_mode must be one of {SERVING_MODES}, got {mode!r}")
    return mode


def framework_mode(release: LoadedRelease, mode: str | None) -> str | None:
    """The mode a framework must be told about: None in the model's default mode (the framework's own defaults
    render it), otherwise the resolved mode."""
    resolved = resolve_mode(release, mode)
    return None if resolved == mode_spec(release).default else resolved


def template_kwargs(release: LoadedRelease, mode: str | None) -> dict[str, Any]:
    """The chat-template arguments for this mode on this model."""
    resolved = resolve_mode(release, mode)
    kwargs = mode_spec(release).kwargs.get(resolved)
    if kwargs is None:
        raise ServingModeError(f"{release.release.spec} documents no {resolved} mode")
    return dict(kwargs)


def mode_not_documented(release: LoadedRelease, mode: str) -> str | None:
    """Why a model has nothing to test in this serving mode, or None. Decided from the model alone, before any
    framework output (D3, the human's decision, 2026-11-25): a model whose mode spec documents no such mode (its
    chat-template arguments are None, e.g. Inkling-2-Mini's non-thinking mode) gets not_applicable for that mode's
    whole table, unless its model card at the pinned revision documents the mode."""
    if mode_spec(release).kwargs.get(mode) is None:
        return (
            f"{mode} mode: the model documents no such mode ({mode_spec(release).evidence}) Decided from the model "
            "alone, before any framework output (D3)"
        )
    return None


REASONING_NOT_NON_THINKING = (
    "non-thinking mode: an assistant turn carries reasoning, and with thinking off the model generates no think "
    "content, so this conversation is not non-thinking data. Decided from the fixture and the serving mode, before "
    "any framework output; reasoning is judged in thinking mode"
)


def not_applicable_in_mode(conversation: Any, mode: str) -> str | None:
    """Why a conversation has nothing to test in this serving mode, or None. Decided before any framework output.

    One rule (human decision, 2026-11-25, M4 calibration): a conversation whose assistant turn carries reasoning
    (`reasoning_content`, the fixtures' one way to carry it) is not applicable in non-thinking mode. Qwen3's model
    card says that with `enable_thinking=False` the model "will not generate think content", so such a turn is not
    something a non-thinking model writes. Reasoning stays fully judged in thinking mode, the primary table.
    The rule takes no framework's reading of reasoning history, so no framework's verdicts are judged against
    themselves.
    """
    if mode == "non_thinking" and any(
        m["role"] == "assistant" and m.get("reasoning_content") for m in conversation.messages
    ):
        return REASONING_NOT_NON_THINKING
    return None
