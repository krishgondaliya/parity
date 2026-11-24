"""t1.special_tokens: are BOS, stop and pad tokens handled the way the template defines?

Bug class: a framework adds a BOS the template already has (double BOS) or one
the model never uses, appends extra EOS tokens, or pads with a token that is
also a real stop token and then masks every "pad", which silently takes the
loss off the model's real stop tokens.

Reference: the template the framework declares it uses.

Pass rule, two parts, both exact:

1. Special tokens, on the encoded row:
   - the run of special tokens at the start of the row equals the reference's;
   - each special token occurs exactly as often as in the reference;
   - every stop token in a must-train position keeps its loss (label equal to
     the token). This last rule needs the row to line up with the reference;
     if it doesn't, it is unchecked and the result can be at best inconclusive.
2. Padding, on a padded batch built by the framework's own collator. The
   fixture is batched with the other fixture whose length differs most from
   it, and every row is padded to `pad_to` = the longer row + 8 tokens:
   - padding sits at one end of the row and leaves the row's tokens and labels
     exactly as the unpadded row has them;
   - pad positions hold the declared pad token and have no loss;
   - every stop token in a must-train position keeps its loss in the batch too.
   If the framework packs instead of padding under our config, part 2 is
   not applicable for that config and the verdict records it. A row that isn't
   padded to `pad_to` makes part 2 inconclusive (check the adapter's mapping).

Planted faults (both apply to every fixture):
- extra_leading_special: prepend one more special token: a second BOS when
  the model has one, otherwise its pad token (Qwen3 has no BOS).
- pad_is_closing_token_masked: declare the token that closes assistant turns
  as the pad token, pad with it, and take the loss off every occurrence.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from parity.adapters.base import LABEL_IGNORE, Batch, Declaration, Dropped, Encoded
from parity.checks.common import (
    Tier1Check,
    closing_token_id,
    declared_reference,
    reference_info,
    special_ids,
    token_ids_on,
)
from parity.engine.evidence import decode_each
from parity.engine.plants import Plant, PlantedAdapter
from parity.engine.verdict import Verdict
from parity.fixtures import FIXTURE_IDS, Conversation, Fixture, load_fixture
from parity.references.template import TemplateReference, Zone
from parity.release import LoadedRelease

PAD_MARGIN = 8


class SpecialTokens(Tier1Check):
    id = "t1.special_tokens"

    def run(self, adapter: Any, release: LoadedRelease, fixture: Fixture) -> Verdict:
        tokenizer = release.tokenizer
        declaration = adapter.declare(self.base_cfg)
        ref = declared_reference(release, fixture, declaration)
        info = reference_info(release, declaration, ref)
        pad = declaration.pad_token_id
        conventions: dict[str, Any] = {
            "batching": declaration.batching,
            "pad_token_id": pad,
            "pad_token": None if pad is None else decode_each(tokenizer, [pad])[0],
            "pad_is_a_stop_token": pad in release.stop_token_ids,
        }
        out = adapter.encode(fixture, self.base_cfg)
        if isinstance(out, Dropped):
            return self.verdict(
                adapter,
                fixture,
                "fail",
                kind="bug",
                reference=info,
                conventions=conventions,
                evidence={"dropped": out.reason, "note": "the conversation was dropped with no length limit"},
            )

        problems: list[dict[str, Any]] = []
        open_questions: list[str] = []
        parts: dict[str, str] = {}

        specials = special_ids(tokenizer)
        problems += _special_token_problems(tokenizer, specials, ref.input_ids, out.input_ids)
        if out.input_ids == ref.input_ids:
            problems += _untrained_stop_tokens(tokenizer, release, ref, out, where="row")
        else:
            open_questions.append(
                "the row's tokens differ from the declared template's render, so stop-token loss is unchecked "
                "(see t1.template_parity)"
            )
        parts["special_tokens"] = "fail" if problems else ("unchecked" if open_questions else "pass")

        if declaration.batching == "packed":
            parts["padding"] = "not_applicable"
        else:
            n_before = len(problems)
            try:
                side, pad_problems, pad_question = self._padding(adapter, release, fixture, declaration, ref, out)
            except Exception as exc:  # noqa: BLE001 - a padding crash leaves this part open, never passed
                side, pad_problems, pad_question = None, [], f"padded batch failed: {type(exc).__name__}: {exc}"
            problems += pad_problems
            if pad_question:
                open_questions.append(pad_question)
            conventions["padding_side"] = side
            parts["padding"] = "fail" if len(problems) > n_before else ("inconclusive" if pad_question else "pass")

        evidence = {"parts": parts, "problems": problems}
        if problems:
            return self.verdict(
                adapter, fixture, "fail", kind="bug", reference=info, conventions=conventions, evidence=evidence
            )
        if open_questions:
            return self.verdict(
                adapter,
                fixture,
                "inconclusive",
                reason="; ".join(open_questions),
                reference=info,
                conventions=conventions,
                evidence=evidence,
            )
        return self.verdict(adapter, fixture, "pass", reference=info, conventions=conventions, evidence=evidence)

    def _padding(
        self,
        adapter: Any,
        release: LoadedRelease,
        fixture: Fixture,
        declaration: Declaration,
        ref: TemplateReference,
        single: Encoded,
    ) -> tuple[str | None, list[dict[str, Any]], str | None]:
        """Returns (padding side, problems, open question)."""
        partner, pad_to = batch_plan(release, fixture, declaration)
        cfg = replace(self.base_cfg, pad_to=pad_to)
        row = adapter.encode_batch([fixture, partner], cfg).rows[0]
        if len(row.input_ids) != pad_to:
            question = (
                f"the padded row has {len(row.input_ids)} tokens, not pad_to={pad_to}; "
                "check the adapter's mapping of pad_to"
            )
            return None, [], question

        content = [i for i, a in enumerate(row.attention_mask) if a == 1]
        if not content or content != list(range(content[0], content[-1] + 1)):
            problem = {"rule": "padding_at_one_end", "detail": "attended positions are not one contiguous block"}
            return None, [problem], None
        start, end = content[0], content[-1] + 1
        if start > 0 and end < len(row.input_ids):
            return None, [{"rule": "padding_at_one_end", "detail": "padding on both sides of the row"}], None
        side = "left" if start > 0 else "right"

        problems = []
        body = Encoded(row.input_ids[start:end], row.labels[start:end], row.attention_mask[start:end])
        if (body.input_ids, body.labels) != (single.input_ids, single.labels):
            pairs = zip(zip(body.input_ids, body.labels), zip(single.input_ids, single.labels))
            problems.append(
                {
                    "rule": "padding_keeps_the_row",
                    "detail": "the padded row's tokens or labels differ from the unpadded row",
                    "changed_positions": [i for i, (a, b) in enumerate(pairs) if a != b][:16],
                    "lengths": [len(body.input_ids), len(single.input_ids)],
                }
            )
        pads = [i for i in range(len(row.input_ids)) if not start <= i < end]
        wrong_id = [i for i in pads if row.input_ids[i] != declaration.pad_token_id]
        trained = [i for i in pads if row.labels[i] != LABEL_IGNORE]
        if wrong_id:
            problems.append(
                {"rule": "pad_token", "detail": "pad positions hold a different token", "positions": wrong_id[:16]}
            )
        if trained:
            problems.append({"rule": "pads_untrained", "detail": "pad positions have loss", "positions": trained[:16]})
        if body.input_ids == ref.input_ids:
            problems += _untrained_stop_tokens(release.tokenizer, release, ref, body, where="padded batch")
        return side, problems, None

    def plants(self, release: LoadedRelease, fixture: Fixture, declaration: Declaration) -> list[Plant]:
        tokenizer = release.tokenizer
        ref = declared_reference(release, fixture, declaration)
        out = []

        extra = next(
            (t for t in (tokenizer.bos_token_id, tokenizer.pad_token_id, tokenizer.eos_token_id) if t is not None),
            None,
        )
        if extra is not None:
            name = decode_each(tokenizer, [extra])[0]
            out.append(
                Plant(
                    "extra_leading_special",
                    f"prepend {name} to every row",
                    lambda adapter: PlantedAdapter(
                        adapter,
                        "extra_leading_special",
                        after_encode=lambda row, conv, cfg: _prepend(row, extra),
                        after_batch=lambda batch, convs, cfg: Batch(
                            tuple(_prepend_in_padded(r, extra) for r in batch.rows)
                        ),
                    ),
                )
            )

        if ref.spans:
            closing = closing_token_id(ref)
            name = decode_each(tokenizer, [closing])[0]
            out.append(
                Plant(
                    "pad_is_closing_token_masked",
                    f"pad with {name} and take the loss off every {name}",
                    lambda adapter: PlantedAdapter(
                        adapter,
                        "pad_is_closing_token_masked",
                        on_declare=lambda d: replace(d, pad_token_id=closing),
                        after_encode=lambda row, conv, cfg: _mask_token(row, closing),
                        after_batch=lambda batch, convs, cfg: Batch(
                            tuple(_mask_token(_repad_with(r, closing), closing) for r in batch.rows)
                        ),
                    ),
                )
            )
        return out


def batch_plan(release: LoadedRelease, fixture: Fixture, declaration: Declaration) -> tuple[Conversation, int]:
    """The partner fixture (the one whose length differs most) and pad_to. Decided from references, not output."""
    template = declaration.chat_template
    own = len(token_ids_on(release, fixture, template))
    lengths = {fid: len(token_ids_on(release, load_fixture(fid), template)) for fid in FIXTURE_IDS if fid != fixture.id}
    partner_id = max(lengths, key=lambda fid: abs(lengths[fid] - own))  # ties: first in FIXTURE_IDS order
    return load_fixture(partner_id), max(own, lengths[partner_id]) + PAD_MARGIN


def _leading_specials(ids: Sequence[int], specials: frozenset[int]) -> list[int]:
    run = []
    for t in ids:
        if t not in specials:
            break
        run.append(t)
    return run


def _special_token_problems(
    tokenizer: Any, specials: frozenset[int], expected: Sequence[int], got: Sequence[int]
) -> list[dict[str, Any]]:
    problems = []
    lead_expected, lead_got = _leading_specials(expected, specials), _leading_specials(got, specials)
    if lead_expected != lead_got:
        problems.append(
            {
                "rule": "leading_special_tokens",
                "expected": decode_each(tokenizer, lead_expected),
                "got": decode_each(tokenizer, lead_got),
            }
        )
    count_expected = Counter(t for t in expected if t in specials)
    count_got = Counter(t for t in got if t in specials)
    if count_expected != count_got:
        problems.append(
            {
                "rule": "special_token_counts",
                "expected": {decode_each(tokenizer, [t])[0]: n for t, n in sorted(count_expected.items())},
                "got": {decode_each(tokenizer, [t])[0]: n for t, n in sorted(count_got.items())},
            }
        )
    return problems


def _untrained_stop_tokens(
    tokenizer: Any, release: LoadedRelease, ref: TemplateReference, row: Encoded, where: str
) -> list[dict[str, Any]]:
    lost = [
        i
        for i, (t, z) in enumerate(zip(ref.input_ids, ref.zones))
        if z == Zone.MUST_TRAIN and t in release.stop_token_ids and row.labels[i] != t
    ]
    if not lost:
        return []
    return [
        {
            "rule": "stop_tokens_trained",
            "where": where,
            "positions": lost,
            "tokens": decode_each(tokenizer, [ref.input_ids[i] for i in lost]),
        }
    ]


def _prepend(row: Encoded | Dropped, token: int) -> Encoded | Dropped:
    if isinstance(row, Dropped):
        return row
    return Encoded((token,) + row.input_ids, (LABEL_IGNORE,) + row.labels, (1,) + row.attention_mask)


def _prepend_in_padded(row: Encoded, token: int) -> Encoded:
    """Prepend to the attended block, then drop one pad so the row keeps its length."""
    attn = row.attention_mask
    start = attn.index(1)
    end = len(attn) - attn[::-1].index(1)
    ids, labels, mask = list(row.input_ids), list(row.labels), list(attn)
    ids.insert(start, token)
    labels.insert(start, LABEL_IGNORE)
    mask.insert(start, 1)
    drop = -1 if end < len(attn) else (0 if start > 0 else None)
    if drop is not None:
        del ids[drop], labels[drop], mask[drop]
    return Encoded(tuple(ids), tuple(labels), tuple(mask))


def _repad_with(row: Encoded, token: int) -> Encoded:
    ids = tuple(t if a == 1 else token for t, a in zip(row.input_ids, row.attention_mask))
    return replace(row, input_ids=ids)


def _mask_token(row: Encoded | Dropped, token: int) -> Encoded | Dropped:
    if isinstance(row, Dropped):
        return row
    labels = tuple(LABEL_IGNORE if t == token else label for t, label in zip(row.input_ids, row.labels))
    return replace(row, labels=labels)
