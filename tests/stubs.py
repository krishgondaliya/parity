"""Test stand-ins for framework adapters. These are not adapters.

Real adapters arrive in M3 and call each framework's own public code path
(hard rule 2). Until then, the checks need a known-good encoder to run
against, so these stand-ins render a chat template through transformers and
build labels in one of two ways:

- "reference": from our own reference zones. Circular on purpose: it is a
  correct encoder by construction, so a check that fails on it has a bug.
- "markers": from the template's `{% generation %}` markers, read by
  transformers. With TRL's vendored Qwen3 training template this is how TRL
  itself builds its assistant mask, so it is independent of our reference.

They live in tests/ and never produce published verdicts. Passing on them
proves the checks' logic, not anything about a framework; real calibration is M4.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from parity.adapters.base import LABEL_IGNORE, Batch, Declaration, Dropped, Encoded, TrainCfg
from parity.checks.common import official_template, reference_on
from parity.engine.plants import Plant
from parity.engine.runner import evaluate
from parity.fixtures import FIXTURE_IDS, Conversation, load_fixture
from parity.references.template import Zone, assistant_mask_from_markers, template_token_ids
from parity.release import LoadedRelease

TRL_SOURCE = "trl@1.18.2:qwen3_training.jinja"
TRL_QWEN3_TRAINING = Path(__file__).parent / "data" / "trl" / "qwen3_training.jinja"
TRL_QWEN3_TRAINING_SHA256 = "1907918feb7afcf18fac8db1a419985b4773ac434e353cfae91b1949dcbf592b"


def read_trl_qwen3_template() -> str:
    raw = TRL_QWEN3_TRAINING.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == TRL_QWEN3_TRAINING_SHA256, (
        "tests/data/trl/qwen3_training.jinja no longer matches TRL v1.18.2 byte for byte "
        "(line endings changed?). See tests/data/trl/SOURCE.md."
    )
    return raw.decode("utf-8")


def markers_moved_onto_the_header(trl_template: str) -> str:
    """TRL's template with {% generation %} opened before the assistant header instead of after it.

    Its markers then claim the header "<|im_start|>assistant\\n" is trained,
    which the prefix diff says must not be: a template whose markers are wrong.
    """
    header = "{{- '<|im_start|>' + message.role + '\\n' }}"
    opening = "        {%- generation %}\n"
    assert trl_template.count(header) == 1 and trl_template.count(opening) == 1
    return trl_template.replace(opening, "").replace(header, "{%- generation %}" + header)


class TemplateStub:
    """A known-good stand-in encoder. Every behavior a check reads is a constructor argument."""

    def __init__(
        self,
        release: LoadedRelease,
        *,
        chat_template: str | None = None,
        template_source: str = "model",
        name: str = "stub-official",
        version: str = "0",
        labels_from: str = "reference",
        neutral_trained: bool = False,
        truncation_mode: str = "right",
        batching: str = "padded",
        padding_side: str = "right",
        label_convention: str = "aligned",
    ):
        assert labels_from in ("reference", "markers")
        assert padding_side in ("right", "left")
        self.release = release
        self.chat_template = chat_template
        self.template_source = template_source
        self.name = name
        self.version = version
        self.env_lock = None
        self.labels_from = labels_from
        self.neutral_trained = neutral_trained
        self.truncation_mode = truncation_mode
        self.batching = batching
        self.padding_side = padding_side
        self.label_convention = label_convention
        self.pad_token_id = release.tokenizer.pad_token_id

    @property
    def template(self) -> str:
        return self.chat_template if self.chat_template is not None else official_template(self.release)

    def declare(self, cfg: TrainCfg) -> Declaration:
        return Declaration(
            chat_template=self.template,
            template_source=self.template_source,
            label_convention=self.label_convention,
            truncation_mode=self.truncation_mode,
            batching=self.batching,
            pad_token_id=self.pad_token_id,
            sources=("test stand-in: constructor arguments",),
        )

    def encode(self, conv: Conversation, cfg: TrainCfg) -> Encoded | Dropped:
        tokenizer = self.release.tokenizer
        ids = template_token_ids(tokenizer, conv, self.template)
        if self.labels_from == "markers":
            marker_ids, mask = assistant_mask_from_markers(tokenizer, conv, self.template)
            assert marker_ids == ids
            labels = [t if m else LABEL_IGNORE for t, m in zip(ids, mask)]
        else:
            zones = reference_on(self.release, conv, self.template, self.template_source).zones
            train = {Zone.MUST_TRAIN} | ({Zone.NEUTRAL} if self.neutral_trained else set())
            labels = [t if z in train else LABEL_IGNORE for t, z in zip(ids, zones)]

        if cfg.max_len is not None and len(ids) > cfg.max_len:
            m = cfg.max_len
            if self.truncation_mode == "drop":
                return Dropped(f"stand-in drops rows longer than {m}")
            if self.truncation_mode == "right":
                ids, labels = ids[:m], labels[:m]
            else:
                ids, labels = ids[-m:], labels[-m:]
        return Encoded(tuple(ids), tuple(labels), (1,) * len(ids))

    def encode_batch(self, convs: Sequence[Conversation], cfg: TrainCfg) -> Batch:
        if self.batching != "padded":
            raise NotImplementedError("this stand-in packs; a check must not ask it for a padded batch")
        rows = [self.encode(c, replace(cfg, pad_to=None)) for c in convs]
        if any(isinstance(r, Dropped) for r in rows):
            raise ValueError("a row was dropped inside a batch")
        target = cfg.pad_to if cfg.pad_to is not None else max(len(r.input_ids) for r in rows)
        out = []
        for r in rows:
            n = target - len(r.input_ids)
            if n < 0:
                raise ValueError(f"row of {len(r.input_ids)} tokens is longer than pad_to={target}")
            pad = ((self.pad_token_id,) * n, (LABEL_IGNORE,) * n, (0,) * n)
            if self.padding_side == "right":
                out.append(Encoded(r.input_ids + pad[0], r.labels + pad[1], r.attention_mask + pad[2]))
            else:
                out.append(Encoded(pad[0] + r.input_ids, pad[1] + r.labels, pad[2] + r.attention_mask))
        return Batch(tuple(out))


def official_stub(release: LoadedRelease, **kwargs) -> TemplateStub:
    """Encodes with the model's own template; labels from our reference."""
    return TemplateStub(release, **kwargs)


def trl_template_stub(release: LoadedRelease, trl_template: str, **kwargs) -> TemplateStub:
    """Encodes with TRL's vendored Qwen3 training template; labels from its generation markers."""
    kwargs.setdefault("name", "stub-trl-template")
    return TemplateStub(
        release,
        chat_template=trl_template,
        template_source=TRL_SOURCE,
        labels_from="markers",
        **kwargs,
    )


def run_on_all_fixtures(check, adapter) -> dict:
    return {fid: evaluate(check, adapter, load_fixture(fid)) for fid in FIXTURE_IDS}


def with_a_plant_that_does_nothing(check):
    """The same check whose only planted fault changes nothing, so its self-test must miss."""

    class NoOpPlant(type(check)):
        def plants(self, release, fixture, declaration):
            return [Plant("noop", "changes nothing", lambda adapter: adapter)]

    return NoOpPlant()


def plant_named(check, release, fixture, declaration, name: str) -> Plant:
    matches = [p for p in check.plants(release, fixture, declaration) if p.name == name]
    assert len(matches) == 1, f"{check.id}: expected one plant named {name!r} for {fixture.id}"
    return matches[0]
