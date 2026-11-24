"""Shared check helpers: the reference cache must never hand one conversation another's reference."""

from parity.checks.common import official_template, token_ids_on
from parity.fixtures import load_fixture
from parity.references.template import template_token_ids


def _with_argument_order(order: list[str]):
    fx = load_fixture("tool_call")
    call = fx.messages[1]["tool_calls"][0]["function"]
    call["arguments"] = {key: call["arguments"][key] for key in order}
    return fx


def test_key_order_in_tool_arguments_is_part_of_the_cache_key(qwen3):
    """transformers' tojson keeps key order, so two orders render differently and must not share a cache entry."""
    a = _with_argument_order(["city", "celsius"])
    b = _with_argument_order(["celsius", "city"])
    assert a.content_key != b.content_key
    template = official_template(qwen3)
    rendered_a = template_token_ids(qwen3.tokenizer, a, template)
    rendered_b = template_token_ids(qwen3.tokenizer, b, template)
    assert rendered_a != rendered_b
    assert token_ids_on(qwen3, a, template) == tuple(rendered_a)
    assert token_ids_on(qwen3, b, template) == tuple(rendered_b)
