"""Planted faults: the same adapter with one known bug injected.

A check's planted faults are its self-test (hard rule 3). Each `Plant` is the
blueprint's `plant(adapter) -> Adapter` for one fault: `plant.apply(adapter)`
returns a `PlantedAdapter` that calls the real adapter and then injects the
fault, either into the conversation it is given (`before_encode`) or into
what comes back (`after_encode`, `after_batch`, `declare`).

Which faults apply to a fixture is decided by the check before any framework
output exists, from the fixture, the model's own tokenizer and template, and
the framework's declaration. Never from what the adapter returns.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any

from parity.adapters.base import LABEL_IGNORE, Batch, Declaration, Dropped, Encoded, TrainCfg
from parity.fixtures import Conversation

BeforeEncode = Callable[[Conversation], Conversation]
AfterEncode = Callable[[Encoded | Dropped, Conversation, TrainCfg], Encoded | Dropped]
AfterBatch = Callable[[Batch, Sequence[Conversation], TrainCfg], Batch]
OnDeclare = Callable[[Declaration], Declaration]


class PlantedAdapter:
    """Wraps a real adapter and injects one fault. Every hook defaults to pass-through."""

    def __init__(
        self,
        inner: Any,
        plant_name: str,
        *,
        before_encode: BeforeEncode | None = None,
        after_encode: AfterEncode | None = None,
        after_batch: AfterBatch | None = None,
        on_declare: OnDeclare | None = None,
    ):
        self.inner = inner
        self.plant_name = plant_name
        self.name = inner.name
        self.version = inner.version
        self.release = inner.release
        self.env_lock = inner.env_lock
        self.env_fingerprint = getattr(inner, "env_fingerprint", None)
        self._before_encode = before_encode
        self._after_encode = after_encode
        self._after_batch = after_batch
        self._on_declare = on_declare

    def declare(self, cfg: TrainCfg) -> Declaration:
        declaration = self.inner.declare(cfg)
        return self._on_declare(declaration) if self._on_declare else declaration

    def encode(self, conv: Conversation, cfg: TrainCfg) -> Encoded | Dropped:
        if self._before_encode:
            conv = self._before_encode(conv)
        out = self.inner.encode(conv, cfg)
        return self._after_encode(out, conv, cfg) if self._after_encode else out

    def encode_batch(self, convs: Sequence[Conversation], cfg: TrainCfg) -> Batch:
        if self._before_encode:
            convs = [self._before_encode(c) for c in convs]
        out = self.inner.encode_batch(convs, cfg)
        return self._after_batch(out, convs, cfg) if self._after_batch else out


ORIGINS = ("real-world", "synthetic")


@dataclass(frozen=True)
class Plant:
    """One planted fault.

    origin: "real-world" only when it reproduces a bug someone actually hit,
        with a public URL in `source` and the same mechanism; "synthetic"
        otherwise, even if it shares a symptom with a known bug (say so in
        `source`). Catching a synthetic fault says less about real bugs, and
        the verdict shows which kind it was.
    """

    name: str
    description: str
    apply: Callable[[Any], Any]  # adapter -> planted adapter
    origin: str
    source: str | None = None

    def __post_init__(self) -> None:
        if self.origin not in ORIGINS:
            raise ValueError(f"plant {self.name}: origin must be one of {ORIGINS}, got {self.origin!r}")
        if self.origin == "real-world" and (not self.source or "https://" not in self.source):
            raise ValueError(f"plant {self.name}: a real-world plant needs a public URL in its source")


def encoded_plant(
    name: str,
    description: str,
    mutate: Callable[[Encoded], Encoded],
    *,
    origin: str,
    source: str | None = None,
) -> Plant:
    """A fault that rewrites every encoded row (dropped conversations pass through unchanged)."""

    def after(out: Encoded | Dropped, conv: Conversation, cfg: TrainCfg) -> Encoded | Dropped:
        return mutate(out) if isinstance(out, Encoded) else out

    return Plant(
        name,
        description,
        lambda adapter: PlantedAdapter(adapter, name, after_encode=after),
        origin=origin,
        source=source,
    )


def with_labels(row: Encoded, labels: Sequence[int]) -> Encoded:
    return replace(row, labels=tuple(labels))


def supervised_runs(row: Encoded) -> list[tuple[int, int]]:
    """Maximal [start, end) runs of supervised positions."""
    runs: list[tuple[int, int]] = []
    start = None
    for i, label in enumerate(row.labels):
        if label != LABEL_IGNORE and start is None:
            start = i
        elif label == LABEL_IGNORE and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(row.labels)))
    return runs
