"""t1.template_parity: does the framework feed the model the tokens the model's own chat template defines?

Bug class: a framework renders the conversation differently from the model's
official chat template (a dropped or extra newline, a wrong or swapped special
token, a missing think block, a different tool-call format), so the model
trains on a format it will never see at inference.

Reference: the model's own template through transformers' apply_chat_template
with default options, on the full fixture, with no length limit.

Pass rule: exact. The framework's input ids must equal the reference ids
token for token. Any divergence is a fail. It is a fail of kind `difference`
only when a known-differences entry matches it exactly (framework, version,
both template hashes, and the full token sequence); otherwise it is kind
`bug`. Dropping the conversation with no length limit is a fail. If the
framework declares a template of its own but its tokens are exactly the
model's own template's render, the declaration is contradicted: an adapter
error, so the result is inconclusive.

Planted faults (each must be caught on every fixture it applies to):
- drop_newline: drop the first newline token. Applies when the declared
  template's render contains a newline token.
- swap_stop_token: replace the first stop token with a different special
  token (Qwen3: <|im_end|> becomes <|endoftext|>). Applies when the declared
  template's render contains a stop token.
- rstrip_render: strip trailing whitespace from the render, which drops only
  the final separator (a neutral token every other check ignores). Applies
  when the declared template's render ends in whitespace.
All three are synthetic.
"""

from __future__ import annotations

from typing import Any

from parity.adapters.base import Dropped, Encoded
from parity.checks.common import (
    Declare,
    Tier1Check,
    adapter_error,
    declared_reference,
    is_contradicted,
    official_reference,
    official_template,
    reference_info,
    replacement_special,
    template_status,
)
from parity.checks.known_differences import match_known_difference
from parity.engine.evidence import decode_each, token_evidence
from parity.engine.plants import Plant, encoded_plant
from parity.engine.verdict import Verdict
from parity.fixtures import Fixture
from parity.release import LoadedRelease


class TemplateParity(Tier1Check):
    id = "t1.template_parity"

    def run(self, adapter: Any, release: LoadedRelease, fixture: Fixture) -> Verdict:
        declaration = adapter.declare(self.base_cfg)
        ref = official_reference(release, fixture)
        info = reference_info(release, ref)
        out = adapter.encode(fixture, self.base_cfg)
        if isinstance(out, Dropped):
            return self.verdict(
                adapter,
                fixture,
                "fail",
                kind="bug",
                reference=info,
                declared=declaration,
                evidence={"dropped": out.reason, "note": "the conversation was dropped with no length limit"},
            )

        status = template_status(release, fixture, declaration, out.input_ids)
        verified = {"chat_template": status}

        def verdict(result: str, **fields: Any) -> Verdict:
            return self.verdict(
                adapter, fixture, result, reference=info, declared=declaration, verified=verified, **fields
            )

        if is_contradicted(status):
            return verdict(
                "inconclusive",
                reason=adapter_error("chat_template", status),
                evidence=token_evidence(release.tokenizer, ref.input_ids, out.input_ids),
            )
        if out.input_ids == ref.input_ids:
            return verdict("pass", evidence={"tokens": len(ref.input_ids)})

        evidence = token_evidence(release.tokenizer, ref.input_ids, out.input_ids)
        entry = match_known_difference(
            check_id=self.id,
            framework=adapter.name,
            version=adapter.version,
            release=release,
            conv=fixture,
            official_template=official_template(release),
            declared_template=declaration.chat_template,
            got_ids=out.input_ids,
        )
        if entry is not None:
            evidence["known_difference"] = {"id": entry.id, "signature": entry.signature, "docs": list(entry.docs)}
            return verdict("fail", kind="difference", known_difference=entry.id, evidence=evidence)
        return verdict("fail", kind="bug", evidence=evidence)

    def reference_properties(self, release: LoadedRelease, fixture: Fixture, declaration: Any) -> list[str]:
        return []  # judged on token ids alone, never on spans

    def plants(self, release: LoadedRelease, fixture: Fixture, declare: Declare) -> list[Plant]:
        tokenizer = release.tokenizer
        ids = declared_reference(release, fixture, declare(self.base_cfg)).input_ids
        out = []

        newline_ids = {i for i in set(ids) if decode_each(tokenizer, [i])[0] == "\n"}
        if newline_ids:

            def drop_newline(row: Encoded) -> Encoded:
                for i, t in enumerate(row.input_ids):
                    if t in newline_ids:
                        return _without(row, i)
                return row

            out.append(
                encoded_plant("drop_newline", "drop the first newline token", drop_newline, origin="synthetic")
            )

        stops = [t for t in ids if t in release.stop_token_ids]
        if stops:
            old = stops[0]
            new = replacement_special(release, old)

            def swap_stop_token(row: Encoded) -> Encoded:
                for i, t in enumerate(row.input_ids):
                    if t == old:
                        return _with_token(row, i, new)
                return row

            names = decode_each(tokenizer, [old, new])
            out.append(
                encoded_plant(
                    "swap_stop_token",
                    f"replace the first {names[0]} with {names[1]}",
                    swap_stop_token,
                    origin="synthetic",
                )
            )

        trailing = _trailing_whitespace(tokenizer, ids)
        if trailing:

            def rstrip_render(row: Encoded) -> Encoded:
                n = _trailing_whitespace(tokenizer, row.input_ids)
                end = len(row.input_ids) - n
                return Encoded(row.input_ids[:end], row.labels[:end], row.attention_mask[:end])

            out.append(
                encoded_plant(
                    "rstrip_render",
                    "strip trailing whitespace from the rendered text (drops the final separator)",
                    rstrip_render,
                    origin="synthetic",
                )
            )
        return out


def _trailing_whitespace(tokenizer: Any, ids: tuple[int, ...]) -> int:
    """How many tokens at the end decode to whitespace only."""
    n = 0
    for token in reversed(ids):
        text = decode_each(tokenizer, [token])[0]
        if not text or not text.isspace():
            break
        n += 1
    return n


def _without(row: Encoded, i: int) -> Encoded:
    def cut(seq):
        return seq[:i] + seq[i + 1 :]

    return Encoded(cut(row.input_ids), cut(row.labels), cut(row.attention_mask))


def _with_token(row: Encoded, i: int, token: int) -> Encoded:
    ids = list(row.input_ids)
    labels = list(row.labels)
    if labels[i] == ids[i]:
        labels[i] = token  # a supervised position keeps its label aligned with its token
    ids[i] = token
    return Encoded(tuple(ids), tuple(labels), row.attention_mask)
