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
    default: str
    kwargs: dict[str, dict[str, Any] | None]  # mode -> chat-template arguments; None if the model has no such mode
    evidence: str


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
        evidence="The template's thinking_mode defaults to 'thinking'; thinking=False selects 'chat'.",
    ),
    "tencent/Hy4-flash": ModeSpec(
        default="thinking",
        kwargs={"thinking": {"reasoning_effort": "high"}, "non_thinking": {"reasoning_effort": "no_think"}},
        evidence="The template's reasoning_effort defaults to 'high' and accepts only 'high' and 'no_think'.",
    ),
    "thinkingmachines/Inkling-2-Mini": ModeSpec(
        default="thinking",
        kwargs={"thinking": {}, "non_thinking": None},
        evidence=(
            "The template takes a numeric reasoning_effort in [0, 0.99] (default 0.9), emitted as a system line; it "
            "documents no value that turns thinking off, so no non-thinking mode is assumed."
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
