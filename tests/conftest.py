"""Shared test fixtures.

The calibration tokenizer is downloaded once (tokenizer files only, never
weights) and then served from the huggingface_hub cache.
"""

import pytest

from parity.release import LoadedRelease, load_release
from parity.targets import CALIBRATION


@pytest.fixture(scope="session")
def qwen3() -> LoadedRelease:
    return load_release(CALIBRATION)
