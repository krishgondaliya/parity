"""Unsloth 2026.11.10 adapter, on the calibration model. Runs inside envs/unsloth-2026.11.10; skipped elsewhere."""

import pytest

unsloth = pytest.importorskip("unsloth")
if unsloth.__version__ != "2026.11.10":
    pytest.skip("needs unsloth==2026.11.10 (envs/unsloth-2026.11.10)", allow_module_level=True)

from parity.adapters.base import TrainCfg  # noqa: E402
from parity.adapters.unsloth import BACKEND, UnslothAdapter  # noqa: E402
from parity.fixtures import FIXTURE_IDS, load_fixture  # noqa: E402
from parity.references.template import template_token_ids  # noqa: E402


CPU_ONLY = pytest.mark.skipif(BACKEND != "cpu", reason="pins Unsloth's CPU-backend behavior (P-0004)")


@pytest.fixture(scope="module")
def adapter(qwen3):
    return UnslothAdapter(qwen3)


def test_the_backend_is_read_from_unsloth_and_recorded(adapter):
    assert adapter.backend == BACKEND == ("cpu" if unsloth.DEVICE_TYPE == "cpu" else "gpu")
    assert f"backend: {BACKEND}" in " ".join(adapter.declare(TrainCfg()).sources)


def test_the_lockfile_can_be_the_cuda_12_6_variant_but_nothing_else(qwen3):
    variant = UnslothAdapter(qwen3, env_lock="envs/unsloth-2026.11.10/cu126/uv.lock")
    assert variant.env_lock.endswith("cu126/uv.lock") and variant.env_fingerprint != UnslothAdapter(qwen3).env_fingerprint
    with pytest.raises(ValueError, match="env_lock"):
        UnslothAdapter(qwen3, env_lock="envs/trl-1.18.2/uv.lock")


def test_declare_reads_the_recipes_live_objects(adapter, qwen3):
    d = adapter.declare(TrainCfg())
    assert d.template_source == "model"  # the loader leaves Qwen3's template unchanged
    assert (d.label_convention, d.truncation_mode, d.batching) == ("aligned", "right", "padded")
    assert d.pad_token_id == qwen3.tokenizer.convert_tokens_to_ids("<|endoftext|>")
    assert d.tool_arguments == "passed_through"
    assert d.serving_mode_setting.startswith("framework defaults")
    assert any("train_on_responses_only" in s for s in d.sources)


def test_non_thinking_mode_is_not_configurable(adapter):
    """P-0006 (2026-11-25): Unsloth documents only the /no_think soft switch, no hard-switch training."""
    from parity.adapters.base import ModeNotConfigurable
    from parity.adapters.unsloth_docs import NOT_CONFIGURABLE

    for call in (lambda c: adapter.declare(c), lambda c: adapter.encode(load_fixture("single_turn"), c),
                 lambda c: adapter.encode_batch([load_fixture("single_turn")], c)):  # fmt: skip
        with pytest.raises(ModeNotConfigurable, match="soft switch"):
            call(TrainCfg(serving_mode="non_thinking"))
    assert str(NOT_CONFIGURABLE).startswith("Unsloth documents non-thinking training via the /no_think soft switch")
    assert adapter.declare(TrainCfg(serving_mode="thinking")).serving_mode_setting.startswith("framework defaults")


def test_saved_non_thinking_cells_rederive_to_exactly_what_the_runner_now_gives(adapter):
    from dataclasses import replace

    from parity.adapters.unsloth_docs import DECISION, NOT_CONFIGURABLE
    from parity.checks.tier1 import tier1_checks
    from parity.engine.runner import as_not_configurable, evaluate_suite
    from parity.fixtures import load_all

    fields = ("check", "fixture", "result", "kind", "self_test", "reason", "declaration", "serving_mode", "plants")
    for real in evaluate_suite(tier1_checks("non_thinking"), adapter, load_all()):
        saved = replace(real, result="pass", reason=None, self_test="caught", plants=[{"plant": "p", "caught": True}],
                        declaration={}) if real.result == "not_configurable" else real  # fmt: skip
        derived = as_not_configurable(saved, NOT_CONFIGURABLE, DECISION)
        assert real.result in ("not_configurable", "not_applicable")
        assert {f: getattr(derived, f) for f in fields} == {f: getattr(real, f) for f in fields}, real.check


@pytest.mark.parametrize("fid", FIXTURE_IDS)
def test_encode_runs_on_every_fixture(adapter, fid):
    row = adapter.encode(load_fixture(fid), TrainCfg())
    assert len(row.input_ids) == len(row.labels) == len(row.attention_mask)
    assert any(label != -100 for label in row.labels)


def test_encode_batch_runs(adapter):
    batch = adapter.encode_batch([load_fixture("single_turn"), load_fixture("multi_turn")], TrainCfg())
    assert len({len(r.input_ids) for r in batch.rows}) == 1


@CPU_ONLY
@pytest.mark.parametrize("fid", FIXTURE_IDS)
def test_rows_are_the_declared_render_plus_one_appended_eos(adapter, qwen3, fid):
    """What the recipe actually produces, pinned exactly: TRL 1.15 appends the EOS token to a "text" row that doesn't
    end with it, and Qwen3's render ends with a newline. Under /triage (M3 Unsloth); if this changes, re-triage."""
    fx = load_fixture(fid)
    declared = adapter.declare(TrainCfg()).chat_template
    eos = qwen3.tokenizer.convert_tokens_to_ids("<|im_end|>")
    row = adapter.encode(fx, TrainCfg())
    assert list(row.input_ids) == template_token_ids(qwen3.tokenizer, fx, declared) + [eos]


@CPU_ONLY
@pytest.mark.xfail(strict=True, reason="rows carry one appended EOS (see the test above); under /triage, M3 Unsloth")
def test_rows_match_the_declared_template_exactly(adapter, qwen3):
    """The adapter contract (the new-adapter skill): rows equal apply_chat_template with the declared template."""
    fx = load_fixture("single_turn")
    declared = adapter.declare(TrainCfg()).chat_template
    assert list(adapter.encode(fx, TrainCfg()).input_ids) == template_token_ids(qwen3.tokenizer, fx, declared)


def test_collators_are_read_in_training_mode_the_way_unsloths_train_enters_it():
    """H-11: on the GPU route SFTTrainer.__init__ ends in model.for_inference() (padding left); train() first calls
    model.for_training(use_gradient_checkpointing=...) (padding right). The adapter makes that same call."""
    from types import SimpleNamespace

    from parity.adapters.unsloth import _enter_training_mode

    calls = []
    model = SimpleNamespace(_unsloth_gradient_checkpointing="unsloth", for_training=lambda **kw: calls.append(kw))
    _enter_training_mode(SimpleNamespace(model=model, args=SimpleNamespace(gradient_checkpointing=False)))
    assert calls == [{"use_gradient_checkpointing": "unsloth"}]  # the recorded mode wins, as in Unsloth's wrapper

    calls.clear()
    bare = SimpleNamespace(for_training=lambda **kw: calls.append(kw))
    _enter_training_mode(SimpleNamespace(model=bare, args=SimpleNamespace(gradient_checkpointing=True)))
    assert calls == [{"use_gradient_checkpointing": True}]  # else the training args


def test_padded_batches_come_out_padded_on_the_right_as_training_pads_them(adapter):
    d = adapter.declare(TrainCfg(pad_to=545))
    assert "padding_side='right'" in next(s for s in d.sources if s.startswith("pad_token_id:"))


def test_the_tiny_deepseek_v41_checkpoint_has_engram_tables_unsloth_can_bind(tmp_path):
    """H-13 (2026-11-27): the tiny checkpoint lacked the engram table rows, and from_pretrained refused to load it.
    The factory now writes them where the release has them, and Unsloth's own bind step finds every table."""
    from transformers import AutoConfig, AutoModelForCausalLM
    from unsloth_zoo.frozen_table_store import bind_frozen_tables, read_safetensors_header

    from parity.release import Release, fetch_release_files
    from parity.targets import PHASE0_TARGETS
    from parity.tiny import build, save

    pin = next(t for t in PHASE0_TARGETS if t.repo_id.startswith("deepseek-ai/"))
    local = fetch_release_files(Release.parse(pin.spec))
    tiny = build(local, "text")
    folder = save(tiny, local, tmp_path / "model")
    names = {}
    for path in folder.glob("*.safetensors"):
        header, _ = read_safetensors_header(str(path))
        names.update({k: v["dtype"] for k, v in header.items() if ".engram.embed." in k})
    assert names == {"model.layers.1.engram.embed.weight": "F8_E4M3", "model.layers.1.engram.embed.scale": "F8_E8M0"}
    model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(folder))
    assert sorted(bind_frozen_tables(model, str(folder), mode="disk")) == ["model.engram_tables.1"]
