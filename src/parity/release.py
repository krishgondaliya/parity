"""Pinned model releases, and a downloader that can only fetch tokenizer files.

Hard rule 6: Tiers 1 and 2 never download model weights. Every download in
Parity goes through `fetch_release_files`, which asks the Hub only for the
file names in `ALLOWED_FILES` and then checks that nothing else arrived.

Hard rule 7: a release is always pinned to a full 40-character commit SHA,
never a branch name like "main", so every result can be re-run byte for byte.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, PreTrainedTokenizerBase

# Exact file names only (no wildcards), so a weight file can never match.
ALLOWED_FILES: tuple[str, ...] = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
    "chat_template.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.model",
    "config.json",
    "generation_config.json",
)

WEIGHT_SUFFIXES: tuple[str, ...] = (".safetensors", ".bin", ".pt", ".pth", ".gguf", ".ckpt", ".h5", ".msgpack", ".onnx")

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


class WeightDownloadError(RuntimeError):
    """Raised if anything that looks like model weights ends up in a download."""


@dataclass(frozen=True)
class Release:
    """One model release pinned to an exact Hugging Face commit."""

    repo_id: str
    revision: str

    def __post_init__(self) -> None:
        if "/" not in self.repo_id or self.repo_id.count("/") != 1:
            raise ValueError(f"repo_id must look like 'org/name', got {self.repo_id!r}")
        if not _SHA_RE.match(self.revision):
            raise ValueError(
                f"revision must be a full 40-character commit SHA, got {self.revision!r}. "
                "Branch names like 'main' are not allowed (hard rule 7)."
            )

    @classmethod
    def parse(cls, spec: str) -> Release:
        """Parse 'org/name@<40-char sha>'."""
        repo_id, sep, revision = spec.partition("@")
        if not sep:
            raise ValueError(f"missing '@<revision>' in {spec!r}")
        return cls(repo_id=repo_id, revision=revision)

    @property
    def spec(self) -> str:
        return f"{self.repo_id}@{self.revision}"


def fetch_release_files(release: Release, cache_dir: Path | None = None) -> Path:
    """Download only the allowed tokenizer/template/config files for a release.

    Returns the local snapshot directory. Raises WeightDownloadError if any file
    outside the allow list is present afterwards.
    """
    local = Path(
        snapshot_download(
            repo_id=release.repo_id,
            revision=release.revision,
            allow_patterns=list(ALLOWED_FILES),
            cache_dir=cache_dir,
        )
    )
    unexpected = [p.name for p in local.rglob("*") if p.is_file() and p.name not in ALLOWED_FILES]
    weights = [name for name in unexpected if name.endswith(WEIGHT_SUFFIXES)]
    if weights or unexpected:
        raise WeightDownloadError(f"unexpected files in {release.spec}: {sorted(unexpected)}")
    return local


@dataclass(frozen=True)
class LoadedRelease:
    """A release's own tokenizer plus the token ids that end generation."""

    release: Release
    tokenizer: PreTrainedTokenizerBase
    stop_token_ids: frozenset[int]


def load_release(release: Release, cache_dir: Path | None = None) -> LoadedRelease:
    """Load a release's own tokenizer and stop tokens. Never runs code from the model repo.

    Stop tokens are the tokenizer's EOS plus every `eos_token_id` in the
    release's generation_config.json: the tokens at which the model stops.
    """
    local = fetch_release_files(release, cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(local, trust_remote_code=False)

    stops: set[int] = set()
    if tokenizer.eos_token_id is not None:
        stops.add(tokenizer.eos_token_id)
    generation_config = local / "generation_config.json"
    if generation_config.exists():
        eos = json.loads(generation_config.read_text(encoding="utf-8")).get("eos_token_id")
        if isinstance(eos, int):
            stops.add(eos)
        elif isinstance(eos, list):
            stops.update(int(i) for i in eos)
    if not stops:
        raise ValueError(f"{release.spec} declares no EOS / stop token")
    return LoadedRelease(release=release, tokenizer=tokenizer, stop_token_ids=frozenset(stops))
