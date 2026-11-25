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
   it, asking for every row to be padded to `pad_to` = the longer row + 8
   tokens. Two rules, judged and reported separately (split 2026-11-25, so a
   dropped width, P-0005, no longer hides how pads are handled):
   2a. Requested width honored (`padding_width`): every row is exactly
       `pad_to` tokens. Any other width fails. Per rule 4, /triage checks the
       adapter's mapping of pad_to before it blames the framework.
   2b. Pad tokens handled correctly at whatever width the collator produced
       (`pad_handling`), in both rows: padding sits at one end and leaves the
       row's tokens and labels exactly as its unpadded row has them; pad
       positions have no loss and hold one token, the declared pad token. In
       the fixture's own row, every stop token in a must-train position also
       keeps its loss (the partner's content is judged in its own cells).
   If the framework declares packing for the batch's config, both are
   inconclusive: Phase 0 configs never ask for packing, and packing isolation
   is a Phase 2 check. The row and the batch are each judged against the
   declaration for their own config.

Declarations are verified here: rows that show packing when padding was
declared (a different number of rows, or position ids restarting inside a
row), or pads that are all one token other than the declared pad token,
contradict the declaration: a declared/actual mismatch, routed to /triage,
so the verdict is inconclusive. Pads holding several different tokens fail.

Planted faults (each part of the check has one of its own):
- extra_leading_special (every fixture): prepend one more special token: a
  second BOS when the model has one, otherwise its pad token (Qwen3 has no BOS).
- pad_is_closing_token_masked (when the framework pads): the collator pads
  with the token that closes assistant turns and takes the loss off every
  occurrence while collating, as transformers' DataCollatorForLanguageModeling
  does with pad = EOS. Only the padded batch shows it, so it guards rule 2b.
- pad_to_ignored (when the framework pads): the batch comes back padded only
  to its longest row, the requested width dropped, as Unsloth's
  train_on_responses_only does when it swaps the collator (P-0005). It guards
  rule 2a.
- closing_token_masked_per_row (every fixture): the same masking done in each
  row before batching. It guards part 1's stop-token rule, and is the only pad
  fault for a framework that packs.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from parity.adapters.base import LABEL_IGNORE, Batch, Declaration, Dropped, Encoded, TrainCfg
from parity.checks.common import (
    CONSISTENT,
    CONTRADICTED,
    MISMATCH,
    Declare,
    Tier1Check,
    adapter_error,
    closing_token_id,
    declared_reference,
    is_contradicted,
    reference_info,
    special_ids,
    template_status,
    token_ids_on,
    unaligned_verdict,
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
        # Two encode calls, each with the declaration for its own config: the row, then the padded batch.
        declaration = adapter.declare(self.base_cfg)
        ref = declared_reference(release, fixture, declaration, self.mode(release))
        info = reference_info(release, ref)
        _, pad_to = batch_plan(release, fixture, declaration, self.mode(release))
        batch_cfg = replace(self.base_cfg, pad_to=pad_to)
        batch_declaration = adapter.declare(batch_cfg)
        pad = batch_declaration.pad_token_id
        conventions: dict[str, Any] = {
            "batching": batch_declaration.batching,
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
                declared=declaration,
                evidence={"dropped": out.reason, "note": "the conversation was dropped with no length limit"},
            )
        template = template_status(release, fixture, declaration, out.input_ids, self.mode(release))
        if is_contradicted(template):
            return unaligned_verdict(
                self,
                adapter,
                release,
                fixture,
                declaration,
                out.input_ids,
                ref.input_ids,
                reference=info,
                conventions=conventions,
            )

        problems: list[dict[str, Any]] = []
        open_questions: list[str] = []
        parts: dict[str, str] = {}
        verified = {"chat_template": template}

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

        batch_verified: dict[str, str] = {}
        if batch_declaration.batching == "packed":
            # Phase 0 configs never ask for packing. Until the Phase 2 packing-isolation check
            # exists, a framework that packs anyway leaves both padding rules open, never passed.
            parts["padding_width"] = parts["pad_handling"] = "inconclusive"
            batch_verified["batching"] = "unverified: no Phase 0 check can confirm packing"
            open_questions.append(
                "the framework declares packing for a config that doesn't ask for it; the padding rules can't "
                "be judged until the Phase 2 packing-isolation check exists"
            )
        else:
            try:
                pad = self._padding(adapter, release, fixture, batch_declaration, batch_cfg, ref, out)
            except Exception as exc:  # noqa: BLE001 - a padding crash leaves both rules open, never passed
                pad = _PaddingResult(question=f"padded batch failed: {type(exc).__name__}: {exc}", judged=False)
            problems += pad.width_problems + pad.problems
            if pad.question:
                open_questions.append(pad.question)
            batch_verified.update(pad.verified)
            conventions["padding_side"] = pad.side
            parts["padding_width"] = "fail" if pad.width_problems else ("pass" if pad.judged else "inconclusive")
            parts["pad_handling"] = "fail" if pad.problems else ("inconclusive" if pad.question else "pass")
        batch_call = batch_declaration.record(batch_verified, batch_cfg)

        evidence = {"parts": parts, "problems": problems}

        def verdict(result: str, **fields: Any) -> Verdict:
            return self.verdict(
                adapter,
                fixture,
                result,
                reference=info,
                conventions=conventions,
                declared=declaration,
                verified=verified,
                other_calls=[batch_call],
                evidence=evidence,
                **fields,
            )

        if problems:
            return verdict("fail", kind="bug")
        if open_questions:
            return verdict("inconclusive", reason="; ".join(open_questions))
        return verdict("pass")

    def _padding(
        self,
        adapter: Any,
        release: LoadedRelease,
        fixture: Fixture,
        declaration: Declaration,
        cfg: TrainCfg,
        ref: TemplateReference,
        single: Encoded,
    ) -> _PaddingResult:
        """Judge the padded batch. `declaration` is the adapter's declaration for `cfg`, the batch's config."""
        partner, pad_to = batch_plan(release, fixture, declaration, self.mode(release))
        assert cfg.pad_to == pad_to
        batch = adapter.encode_batch([fixture, partner], cfg)

        # Declared padding, but the rows show packing: a declared/actual mismatch for /triage.
        packed = None
        if len(batch.rows) != 2:
            packed = f"returned {len(batch.rows)} rows for 2 conversations"
        elif _positions_restart(batch.rows[0]):
            packed = "position ids restart inside the row"
        if packed:
            status = CONTRADICTED + packed + ", so the framework packed"
            return _PaddingResult(question=adapter_error("batching", status), verified={"batching": status}, judged=False)

        # Rule 2a: the requested width.
        widths = sorted({len(r.input_ids) for r in batch.rows})
        width_problems = []
        if widths != [pad_to]:
            width_problems.append(
                {
                    "rule": "pad_to_honored",
                    "detail": f"asked to pad every row to {pad_to} tokens, the batch is {widths} wide",
                    "pad_to": pad_to,
                    "widths": widths,
                    "note": "rule 4: check the adapter's mapping of pad_to before blaming the framework",
                }
            )

        # Rule 2b: pad handling at whatever width the collator produced, in both rows.
        out = self._pad_handling(release, batch.rows[0], single, ref, declaration, own_row=True)
        partner_single = adapter.encode(partner, replace(cfg, pad_to=None))
        if isinstance(partner_single, Dropped):
            question = f"the batch partner {partner.id} was dropped unpadded: {partner_single.reason}"
            return _PaddingResult(width_problems=width_problems, question=question)
        other = self._pad_handling(release, batch.rows[1], partner_single, None, declaration, own_row=False)
        sides = {s for s in (out.side, other.side) if s is not None}
        verified = {"batching": CONSISTENT}
        pad_tokens = sorted(set(out.pad_tokens) | set(other.pad_tokens))
        question = None
        problems = [{**p, "row": fixture.id} for p in out.problems] + [{**p, "row": partner.id} for p in other.problems]
        if len(sides) > 1:
            problems.append({"rule": "padding_at_one_end", "detail": "the two rows are padded on different sides"})
        if not pad_tokens:
            verified["pad_token_id"] = "unverified: no row has pad positions at this width"
        elif pad_tokens == [declaration.pad_token_id]:
            verified["pad_token_id"] = CONSISTENT
        elif len(pad_tokens) == 1:
            shown = decode_each(release.tokenizer, pad_tokens)[0]
            verified["pad_token_id"] = CONTRADICTED + f"the framework pads with {shown!r} (id {pad_tokens[0]})"
            question = adapter_error("pad_token_id", verified["pad_token_id"])
        else:
            verified["pad_token_id"] = MISMATCH + "pad positions hold several different tokens"
            problems.append({"rule": "pad_token", "detail": "pad positions hold several different tokens"})
        side = next(iter(sides)) if len(sides) == 1 else None
        return _PaddingResult(side, width_problems, problems, question, verified)

    def _pad_handling(
        self,
        release: LoadedRelease,
        row: Encoded,
        single: Encoded,
        ref: TemplateReference | None,
        declaration: Declaration,
        own_row: bool,
    ) -> _RowPadding:
        """Rule 2b on one row of the batch. Stop-token loss is judged only on the fixture's own row (ref given)."""
        content = [i for i, a in enumerate(row.attention_mask) if a == 1]
        if not content or content != list(range(content[0], content[-1] + 1)):
            return _RowPadding(problems=[{"rule": "padding_at_one_end", "detail": "attended positions are not one contiguous block"}])
        start, end = content[0], content[-1] + 1
        if start > 0 and end < len(row.input_ids):
            return _RowPadding(problems=[{"rule": "padding_at_one_end", "detail": "padding on both sides of the row"}])
        side = "left" if start > 0 else ("right" if end < len(row.input_ids) else None)  # None: no pads in this row
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
        trained = [i for i in pads if row.labels[i] != LABEL_IGNORE]
        if trained:
            problems.append({"rule": "pads_untrained", "detail": "pad positions have loss", "positions": trained[:16]})
        if own_row and ref is not None and body.input_ids == ref.input_ids:
            problems += _untrained_stop_tokens(release.tokenizer, release, ref, body, where="padded batch")
        return _RowPadding(side=side, problems=problems, pad_tokens=sorted({row.input_ids[i] for i in pads}))

    def plants(self, release: LoadedRelease, fixture: Fixture, declare: Declare) -> list[Plant]:
        tokenizer = release.tokenizer
        declaration = declare(self.base_cfg)
        ref = declared_reference(release, fixture, declaration, self.mode(release))
        _, pad_to = batch_plan(release, fixture, declaration, self.mode(release))
        batch_declaration = declare(replace(self.base_cfg, pad_to=pad_to))  # the config the batch runs under
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
                    origin="synthetic",
                )
            )

        if ref.spans:
            closing = closing_token_id(ref)
            name = decode_each(tokenizer, [closing])[0]

            def masked_batch(batch: Batch, convs: Any, cfg: Any) -> Batch:
                return Batch(tuple(_mask_token(_repad_with(r, closing), closing) for r in batch.rows))

            if batch_declaration.batching == "padded":
                # The real mechanism masks pad ids while collating, so only the padded batch shows it:
                # this fault guards rule 2b, pad handling.
                out.append(
                    Plant(
                        "pad_is_closing_token_masked",
                        f"collator pads with {name} and takes the loss off every {name}",
                        lambda adapter: PlantedAdapter(
                            adapter,
                            "pad_is_closing_token_masked",
                            on_declare=lambda d: replace(d, pad_token_id=closing),
                            after_batch=masked_batch,
                        ),
                        origin="real-world",
                        source=(
                            "pad_token set to the EOS token, with id-based pad masking at collation: "
                            "https://github.com/huggingface/transformers/blob/v5.24.0/src/transformers/data/"
                            "data_collator.py#L798-L800 (DataCollatorForLanguageModeling, mlm=False: "
                            "labels[labels == pad_token_id] = -100 also takes the loss off every real EOS)"
                        ),
                    )
                )
            # The same masking done per row, before any batching: this fault guards the row-level rule,
            # and is the only pad fault for a framework that packs.
            out.append(
                Plant(
                    "closing_token_masked_per_row",
                    f"pad id set to {name} and every {name} masked in each row before batching",
                    lambda adapter: PlantedAdapter(
                        adapter,
                        "closing_token_masked_per_row",
                        on_declare=lambda d: replace(d, pad_token_id=closing),
                        after_encode=lambda row, conv, cfg: _mask_token(row, closing),
                        after_batch=masked_batch,
                    ),
                    origin="synthetic",
                )
            )
        if batch_declaration.batching == "padded":
            out.append(
                Plant(
                    "pad_to_ignored",
                    "the batch comes back padded only to its longest row: the requested width is dropped",
                    lambda adapter: PlantedAdapter(
                        adapter,
                        "pad_to_ignored",
                        after_batch=lambda batch, convs, cfg: _trim_to_longest(batch),
                    ),
                    origin="real-world",
                    source=(
                        "Unsloth's train_on_responses_only replaces TRL's collator with DataCollatorForSeq2Seq and "
                        "keeps none of its settings, so SFTConfig.pad_to_multiple_of is dropped (P-0005): "
                        "https://github.com/unslothai/unsloth-zoo/blob/cad81ef2dc557b1060896d3b294ef6b9a7f9c1ea/"
                        "unsloth_zoo/dataset_utils.py#L2547-L2583"
                    ),
                )
            )
        return out


@dataclass
class _PaddingResult:
    side: str | None = None
    width_problems: list[dict[str, Any]] = field(default_factory=list)  # rule 2a
    problems: list[dict[str, Any]] = field(default_factory=list)  # rule 2b
    question: str | None = None  # why a padding rule can't be judged (makes the verdict inconclusive)
    verified: dict[str, str] = field(default_factory=dict)
    judged: bool = True  # False when the batch couldn't be read at all (packing, a crash): rule 2a is open too


@dataclass
class _RowPadding:
    side: str | None = None
    problems: list[dict[str, Any]] = field(default_factory=list)
    pad_tokens: list[int] = field(default_factory=list)


def _trim_to_longest(batch: Batch) -> Batch:
    """Every row cut back to the batch's longest attended length, keeping its padding side: the requested width
    dropped, as P-0005's swapped collator does."""
    width = max(sum(r.attention_mask) for r in batch.rows)

    def trim(row: Encoded) -> Encoded:
        n = len(row.input_ids)
        keep = range(n - width, n) if row.attention_mask[0] == 0 else range(width)  # left-padded rows keep their tail
        cut = lambda seq: tuple(seq[i] for i in keep)  # noqa: E731
        positions = cut(row.position_ids) if row.position_ids is not None else None
        return Encoded(cut(row.input_ids), cut(row.labels), cut(row.attention_mask), positions)

    return Batch(tuple(trim(r) for r in batch.rows))


def _positions_restart(row: Encoded) -> bool:
    """True if position ids go back down inside the attended part of the row: several sequences packed together."""
    if row.position_ids is None:
        return False
    attended = [p for p, a in zip(row.position_ids, row.attention_mask) if a == 1]
    return any(b <= a for a, b in zip(attended, attended[1:]))


def batch_plan(
    release: LoadedRelease, fixture: Fixture, declaration: Declaration, mode: str | None = None
) -> tuple[Conversation, int]:
    """The partner fixture (the one whose length differs most) and pad_to. Decided from references, not output."""
    template = declaration.chat_template
    own = len(token_ids_on(release, fixture, template, mode))
    lengths = {fid: len(token_ids_on(release, load_fixture(fid), template, mode)) for fid in FIXTURE_IDS if fid != fixture.id}
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
