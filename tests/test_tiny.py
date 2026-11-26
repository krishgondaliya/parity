"""The tiny model factory (parity.tiny): the release's own architecture where the environment builds it, otherwise
the D1 stand-in, recorded as a deviation. Needs torch, so it runs in the framework environments only.

Which targets build natively depends on the environment (docs/evidence/2026-11-25-m5-env-support/), so each test
asks the environment first and checks the factory did the matching thing.
"""

import json

import pytest

torch = pytest.importorskip("torch")

from transformers import AutoConfig, AutoModelForCausalLM  # noqa: E402

from parity.release import fetch_release_files  # noqa: E402
from parity.targets import CALIBRATION, PHASE0_TARGETS  # noqa: E402
from parity.tiny import QWEN3, STAND_INS, NoStandIn, build, save  # noqa: E402


def _builds_natively(local) -> bool:
    try:
        AutoConfig.from_pretrained(local)
        return True
    except ValueError:
        return False


def test_qwen3_is_exactly_the_model_the_adapters_built_before_the_factory():
    """Calibration must not move: the old per-adapter procedure, inlined, gives the same config and weights."""
    local = fetch_release_files(CALIBRATION)
    config = AutoConfig.from_pretrained(local)
    for key, value in QWEN3.items():
        if hasattr(config, key):
            setattr(config, key, value)
    if getattr(config, "layer_types", None):
        config.layer_types = config.layer_types[: config.num_hidden_layers]
    torch.manual_seed(0)
    before = AutoModelForCausalLM.from_config(config)

    tiny = build(local, "text")
    assert tiny.model.config.to_json_string(use_diff=False) == before.config.to_json_string(use_diff=False)
    after, old = tiny.model.state_dict(), before.state_dict()
    assert after.keys() == old.keys() and all(torch.equal(after[k], old[k]) for k in old)
    assert tiny.record["stand_in"] is None and tiny.record["deviation"] is None


@pytest.mark.parametrize("target", PHASE0_TARGETS, ids=lambda t: t.repo_id)
def test_a_target_builds_as_itself_or_as_its_recorded_stand_in(target):
    local = fetch_release_files(target)
    raw = json.loads((local / "config.json").read_text(encoding="utf-8"))
    tiny = build(local, "text")
    record = tiny.record
    assert record["release_model_type"] == raw["model_type"] and record["path"] == "text"
    if _builds_natively(local):
        assert record["stand_in"] is None and record["deviation"] is None
    else:
        assert record["stand_in"] == STAND_INS[raw["model_type"]]["text"]
        assert record["deviation"].startswith(f"stand-in: the model is {record['class']} ({record['stand_in']})")
        assert "does not recognize this architecture" in record["deviation"]
    text = tiny.model.config.get_text_config()
    assert text.vocab_size == (raw.get("text_config") or raw)["vocab_size"]  # every token id fits the embedding
    assert sum(p.numel() for p in tiny.model.parameters()) < 50_000_000


def test_the_vision_path_builds_an_image_text_to_text_model_or_says_why_not():
    for target in PHASE0_TARGETS:
        local = fetch_release_files(target)
        try:
            tiny = build(local, "vision")
        except NoStandIn as exc:
            assert target.repo_id.startswith("deepseek-ai/") or "not registered" in str(exc), (target, exc)
            continue
        except ModuleNotFoundError:
            continue  # recorded in the support evidence (a stand-in needing a package the lock doesn't have)
        assert tiny.record["path"] == "vision" and tiny.record["class"].endswith("ForConditionalGeneration")


def test_save_puts_the_release_files_next_to_the_tiny_weights(tmp_path):
    local = fetch_release_files(CALIBRATION)
    folder = save(build(local, "text"), local, tmp_path / "model")
    names = {p.name for p in folder.iterdir()}
    assert {"config.json", "tokenizer.json", "tokenizer_config.json"} <= names
    assert json.loads((folder / "config.json").read_text())["num_hidden_layers"] == QWEN3["num_hidden_layers"]


def test_a_path_that_isnt_text_or_vision_is_refused():
    with pytest.raises(ValueError, match="path"):
        build(fetch_release_files(CALIBRATION), "audio")
