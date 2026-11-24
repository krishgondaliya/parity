from pathlib import Path

import pytest

from parity.release import ALLOWED_FILES, WEIGHT_SUFFIXES, Release, fetch_release_files
from parity.targets import CALIBRATION, PHASE0_TARGETS

SHA = "c1899de289a04d12100db370d81485cdf75e47ca"


def test_parse_round_trips():
    release = Release.parse(f"Qwen/Qwen3-0.6B@{SHA}")
    assert release == Release("Qwen/Qwen3-0.6B", SHA)
    assert release.spec == f"Qwen/Qwen3-0.6B@{SHA}"


@pytest.mark.parametrize(
    "spec",
    [
        "Qwen/Qwen3-0.6B",  # no revision
        "Qwen/Qwen3-0.6B@main",  # branch, not a commit
        "Qwen/Qwen3-0.6B@c1899de",  # short SHA
        f"Qwen/Qwen3-0.6B@{SHA.upper()}",  # the Hub uses lowercase SHAs
        f"Qwen3-0.6B@{SHA}",  # no org
        f"a/b/c@{SHA}",  # too many parts
    ],
)
def test_parse_rejects_unpinned_or_malformed_specs(spec):
    with pytest.raises(ValueError):
        Release.parse(spec)


def test_allow_list_is_exact_names_and_never_weights():
    for name in ALLOWED_FILES:
        assert not any(ch in name for ch in "*?[]"), f"{name} is a pattern, not a file name"
        assert not name.endswith(WEIGHT_SUFFIXES)


def test_calibration_download_contains_only_allowed_files():
    local = fetch_release_files(CALIBRATION)
    names = {p.name for p in local.rglob("*") if p.is_file()}
    assert names
    assert names <= set(ALLOWED_FILES)
    assert not any(n.endswith(WEIGHT_SUFFIXES) for n in names)


def test_calibration_stop_tokens_are_im_end_and_endoftext(qwen3):
    tok = qwen3.tokenizer
    assert qwen3.stop_token_ids == {
        tok.convert_tokens_to_ids("<|im_end|>"),
        tok.convert_tokens_to_ids("<|endoftext|>"),
    }


def test_pinned_revisions_match_the_decision_log():
    log = (Path(__file__).parents[1] / "docs" / "DECISIONS.md").read_text(encoding="utf-8")
    for release in (CALIBRATION, *PHASE0_TARGETS):
        assert f"`{release.repo_id}`" in log
        assert f"`{release.revision}`" in log
