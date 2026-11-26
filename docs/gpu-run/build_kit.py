"""Build the GPU run kit from the committed code: `git archive` of HEAD, never a copy of the working folder.

Run from anywhere in the repo, with a clean working tree:
    uv run python docs/gpu-run/build_kit.py [--models org/name@<sha>,...]
`--models` (default: the calibration model) is written into the kit's copy of the notebook (cell 2). Every target in
it needs its Unsloth predictions committed in HEAD (parity.engine.predictions), or the build refuses.

Writes dist/gpu-kit/<commit12>/ (gitignored):
    parity-src-<commit12>.zip   `git archive --format=zip HEAD`. Git stores the commit id as the zip's comment, and
                                the notebook reads it back from there.
    parity_gpu_run.ipynb        the notebook as committed at HEAD, with cell 2's MODELS set to --models
    KIT.txt                     the commit, the archive's sha256, and what to do next
Refuses to build if tracked files have uncommitted changes, if `.env` is tracked, or if any committed file
looks like it holds a token. Files are named in the error; their contents are never printed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

# Patterns of credentials that must never ship (Hugging Face, GitHub, OpenAI-style, Slack, AWS).
TOKEN_PATTERNS = (
    r"hf_[A-Za-z0-9]{30,}",
    r"ghp_[A-Za-z0-9]{36}",
    r"github_pat_[A-Za-z0-9_]{50,}",
    r"sk-[A-Za-z0-9_-]{32,}",
    r"xox[abprs]-[A-Za-z0-9-]{10,}",
    r"AKIA[0-9A-Z]{16}",
)
REQUIRED = (
    "docs/gpu-run/colab_kit.py",
    "docs/gpu-run/run_checks.py",
    "envs/unsloth-2026.11.10/uv.lock",
    "envs/unsloth-2026.11.10/cu126/uv.lock",
    "src/parity/adapters/unsloth.py",
)


def git(*args: str, root: Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=check)


def refuse(message: str) -> None:
    print(f"REFUSED: {message}", file=sys.stderr)
    sys.exit(2)


UNSLOTH_VERSION = "2026.11.10"
CALIBRATION = "Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca"


def notebook_with_models(notebook: str, models: list[str]) -> str:
    """The notebook with cell 2's MODELS list replaced by `models` (nothing else changes)."""
    nb = json.loads(notebook)
    cell = next(c for c in nb["cells"] if c["cell_type"] == "code" and "MODELS = [" in "".join(c["source"]))
    source = "".join(cell["source"])
    start = source.index("MODELS = [")
    end = source.index("]", start) + 1
    source = source[:start] + "MODELS = [\n" + "".join(f'    "{m}",\n' for m in models) + "]" + source[end:]
    parts = source.split("\n")
    cell["source"] = [p + "\n" for p in parts[:-1]] + ([parts[-1]] if parts[-1] else [])
    return json.dumps(nb, indent=1, ensure_ascii=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", default=CALIBRATION, help="comma-separated org/name@<40-char revision>")
    models = [m.strip() for m in parser.parse_args().models.split(",") if m.strip()]
    root = Path(git("rev-parse", "--show-toplevel").stdout.strip())
    dirty = git("status", "--porcelain", "--untracked-files=no", root=root).stdout.strip()
    if dirty:
        refuse(f"commit or stash these first; the kit is built from HEAD only:\n{dirty}")
    commit = git("rev-parse", "HEAD", root=root).stdout.strip()

    from parity.engine.predictions import PredictionsMissing, committed_predictions, needs_predictions
    from parity.release import Release

    for spec in models:
        Release.parse(spec)  # org/name@<40-char sha> only (rule 7)
        if needs_predictions(spec):
            try:
                committed_predictions(root, "Unsloth", UNSLOTH_VERSION, spec)
            except PredictionsMissing as exc:
                refuse(f"predictions first: {exc}")

    if git("ls-files", "--error-unmatch", ".env", root=root, check=False).returncode == 0:
        refuse(".env is tracked by git; it must never ship")
    pattern_args = [arg for p in TOKEN_PATTERNS for arg in ("-e", p)]
    found = git("grep", "-I", "-l", "-E", *pattern_args, "HEAD", "--", ".", root=root, check=False)
    if found.stdout.strip():
        refuse(f"these committed files look like they hold a token (contents not shown):\n{found.stdout.strip()}")

    out = root / "dist" / "gpu-kit" / commit[:12]
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    archive = out / f"parity-src-{commit[:12]}.zip"
    git("archive", "--format=zip", "-o", str(archive), "HEAD", root=root)

    with zipfile.ZipFile(archive) as zf:
        names = set(zf.namelist())
        if zf.comment.decode("ascii", "replace") != commit:
            refuse("git archive didn't record the commit id as the zip comment")
    missing = [n for n in REQUIRED if n not in names]
    if missing:
        refuse(f"the archive lacks {missing}; commit them first")
    if any(n == ".env" or n.endswith("/.env") for n in names):
        refuse("the archive holds a .env file")

    notebook = git("show", "HEAD:docs/gpu-run/parity_gpu_run.ipynb", root=root).stdout
    (out / "parity_gpu_run.ipynb").write_text(notebook_with_models(notebook, models), encoding="utf-8")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    built = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (out / "KIT.txt").write_text(
        f"Parity GPU run kit\n"
        f"commit          {commit}\n"
        f"archive         {archive.name}\n"
        f"archive sha256  {digest}\n"
        f"archive size    {archive.stat().st_size / 1e6:.1f} MB, {len(names)} files\n"
        f"built (UTC)     {built}\n"
        f"models          {', '.join(models)}\n\n"
        f"Next: follow docs/gpu-run/GPU-RUN.md. Upload parity_gpu_run.ipynb to Colab, then upload {archive.name}\n"
        f"when cell 4 asks for it.\n",
        encoding="utf-8",
    )

    windows = subprocess.run(["wslpath", "-w", str(out)], capture_output=True, text=True, check=False)
    print(f"Kit built from commit {commit}")
    print(f"  {archive.name}  sha256 {digest}  ({archive.stat().st_size / 1e6:.1f} MB)")
    print(f"  models: {', '.join(models)}")
    print(f"  folder (Linux):   {out}")
    if windows.returncode == 0 and windows.stdout.strip():
        print(f"  folder (Windows): {windows.stdout.strip()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
