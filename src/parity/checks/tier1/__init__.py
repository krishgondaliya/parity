"""Tier 1: data-path checks on the release's real tokenizer, no model weights."""

from parity.checks.tier1.assistant_mask import AssistantMask
from parity.checks.tier1.end_of_turn import EndOfTurn
from parity.checks.tier1.multiturn_prefix import MultiturnPrefix
from parity.checks.tier1.special_tokens import SpecialTokens
from parity.checks.tier1.template_parity import TemplateParity
from parity.checks.tier1.truncation import Truncation



def tier1_checks(serving_mode: str | None = None) -> tuple:
    """The six checks, judging against the reference rendered in this serving mode (None: the model's default)."""
    return tuple(
        cls(serving_mode) for cls in (TemplateParity, AssistantMask, EndOfTurn, SpecialTokens, Truncation, MultiturnPrefix)
    )


TIER1_CHECKS = tier1_checks()

__all__ = [
    "TIER1_CHECKS",
    "AssistantMask",
    "EndOfTurn",
    "MultiturnPrefix",
    "SpecialTokens",
    "TemplateParity",
    "Truncation",
    "tier1_checks",
]
