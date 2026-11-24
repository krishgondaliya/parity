"""Shared test fixtures.

The calibration tokenizer is downloaded once (tokenizer files only, never
weights) and then served from the huggingface_hub cache.
"""

import pytest
from stubs import official_stub, read_trl_qwen3_template, trl_template_stub

from parity.release import LoadedRelease, load_release
from parity.targets import CALIBRATION


@pytest.fixture(scope="session")
def qwen3() -> LoadedRelease:
    return load_release(CALIBRATION)


@pytest.fixture(scope="session")
def trl_qwen3_template() -> str:
    return read_trl_qwen3_template()


@pytest.fixture(scope="session")
def official(qwen3):
    """Stand-in encoder on Qwen3's own template (labels from our reference)."""
    return official_stub(qwen3)


@pytest.fixture(scope="session")
def trl_markers(qwen3, trl_qwen3_template):
    """Stand-in encoder on TRL's Qwen3 training template (labels from its generation markers)."""
    return trl_template_stub(qwen3, trl_qwen3_template)
