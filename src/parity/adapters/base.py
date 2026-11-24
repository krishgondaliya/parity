"""What every framework adapter reports, and the settings a check asks for.

An adapter calls a framework's own public data path and reports exactly what
that framework would train on (hard rule 2: it never re-implements the
framework's tokenization, masking, collation or truncation). This module
holds only the types. Real adapters arrive in Phase 0 M3; until then the
checks are tested against stand-ins that live in `tests/`.

Two kinds of information come back from an adapter, and checks keep them apart:

- `declare(cfg)` returns what the framework's own configuration and defaults
  say it will do (which chat template, how it truncates, whether it pads or
  packs, how its labels are laid out). It is read before any encoding, so a
  check can decide what to expect and which planted faults apply without
  looking at the framework's output. A declaration is a claim: checks verify
  it against behavior wherever behavior can show it, and every verdict
  records it.
- `encode` and `encode_batch` return what the framework actually produced.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol

from parity.fixtures import Conversation
from parity.release import LoadedRelease

LABEL_IGNORE = -100  # the label value that means "no loss on this token"

OBJECTIVES = ("assistant_only", "completion_only", "all")
LABEL_CONVENTIONS = ("aligned", "pre_shifted")
TRUNCATION_MODES = ("right", "left", "drop")
BATCHING = ("padded", "packed")


@dataclass(frozen=True)
class TrainCfg:
    """The settings a check asks for. Each adapter maps them onto the framework's own option names.

    objective: which tokens get loss (for TRL: `assistant_only_loss`, `completion_only_loss`).
    max_len: the length limit. None means "no limit": the adapter turns the
        framework's truncation off, or sets its limit above any fixture.
    packing: ask the framework to pack rows (Tier 1 never does).
    grad_accum: used from Tier 2 on.
    pad_to: `encode_batch` only. Pad every row to exactly this many tokens
        through the framework's own collator. Checks set it longer than every row.
    """

    objective: str = "assistant_only"
    max_len: int | None = None
    packing: bool = False
    grad_accum: int = 1
    pad_to: int | None = None

    def __post_init__(self) -> None:
        if self.objective not in OBJECTIVES:
            raise ValueError(f"objective must be one of {OBJECTIVES}, got {self.objective!r}")
        if self.max_len is not None and self.max_len < 1:
            raise ValueError(f"max_len must be positive, got {self.max_len}")
        if self.pad_to is not None and self.pad_to < 1:
            raise ValueError(f"pad_to must be positive, got {self.pad_to}")


@dataclass(frozen=True)
class Encoded:
    """One training row, exactly as the framework would train on it.

    `labels` are always aligned with `input_ids` (label i is the target for
    token i, and the model shifts internally), with LABEL_IGNORE where there is
    no loss. An adapter whose framework pre-shifts labels converts them back
    and says so in `Declaration.label_convention`.
    """

    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    attention_mask: tuple[int, ...]
    position_ids: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        n = len(self.input_ids)
        if len(self.labels) != n or len(self.attention_mask) != n:
            raise ValueError(
                f"input_ids, labels and attention_mask must have the same length "
                f"({n}, {len(self.labels)}, {len(self.attention_mask)})"
            )
        if self.position_ids is not None and len(self.position_ids) != n:
            raise ValueError("position_ids must match input_ids in length")

    def supervised(self, i: int) -> bool:
        return self.labels[i] != LABEL_IGNORE


@dataclass(frozen=True)
class Dropped:
    """The framework skipped the conversation instead of producing a row."""

    reason: str  # the framework's own message or setting, as the adapter captured it


@dataclass(frozen=True)
class Batch:
    """Rows after the framework's own collator, one per input conversation, in order."""

    rows: tuple[Encoded, ...]


@dataclass(frozen=True)
class Declaration:
    """What the framework says it will do under a TrainCfg, read from its own config and defaults.

    chat_template: the template the framework will render with, read from its
        own objects after setup (a framework may swap in its own training template).
    template_source: "model" for the model's own template, otherwise a label
        such as "trl@1.18.2:qwen3_training.jinja".
    label_convention: how the framework lays out labels natively. `encode`
        always returns aligned labels; this records whether the adapter shifted them.
    truncation_mode: "right" (keep the start), "left" (keep the end) or "drop".
    batching: "padded" or "packed" under this config.
    pad_token_id: the id the framework pads with.
    sources: where each value was read from (config field or default), for the verdict.
    """

    chat_template: str
    template_source: str
    label_convention: str
    truncation_mode: str
    batching: str
    pad_token_id: int | None
    sources: tuple[str, ...] = ()

    FIELDS: ClassVar[tuple[str, ...]] = (
        "chat_template",
        "label_convention",
        "truncation_mode",
        "batching",
        "pad_token_id",
    )

    def record(self, verified: dict[str, str] | None = None) -> dict[str, Any]:
        """The declaration as it goes into a verdict. Fields a check didn't verify stay "unverified"."""
        status = {name: "unverified" for name in self.FIELDS}
        status.update(verified or {})
        return {
            "template_source": self.template_source,
            "template_sha256": hashlib.sha256(self.chat_template.encode("utf-8")).hexdigest(),
            "label_convention": self.label_convention,
            "truncation_mode": self.truncation_mode,
            "batching": self.batching,
            "pad_token_id": self.pad_token_id,
            "sources": list(self.sources),
            "verified": status,
        }

    def __post_init__(self) -> None:
        if self.label_convention not in LABEL_CONVENTIONS:
            raise ValueError(f"label_convention must be one of {LABEL_CONVENTIONS}, got {self.label_convention!r}")
        if self.truncation_mode not in TRUNCATION_MODES:
            raise ValueError(f"truncation_mode must be one of {TRUNCATION_MODES}, got {self.truncation_mode!r}")
        if self.batching not in BATCHING:
            raise ValueError(f"batching must be one of {BATCHING}, got {self.batching!r}")


class Adapter(Protocol):
    """A framework at a pinned version, bound to one model release. Tier 2 methods arrive with Tier 2."""

    name: str  # "trl", "unsloth", ...
    version: str  # pinned in envs/
    release: LoadedRelease
    env_lock: str | None  # path of the lockfile the framework environment was built from

    def declare(self, cfg: TrainCfg) -> Declaration: ...

    def encode(self, conv: Conversation, cfg: TrainCfg) -> Encoded | Dropped: ...

    def encode_batch(self, convs: Sequence[Conversation], cfg: TrainCfg) -> Batch: ...
