"""Build the GPU run kit from the committed code: `git archive` of HEAD, never a copy of the working folder.

Run from anywhere in the repo, with a clean working tree:
    uv run python docs/gpu-run/build_kit.py

Writes dist/gpu-kit/<commit12>/ (gitignored):
    parity-src-<commit12>.zip   `git archive --format=zip HEAD`. Git stores the commit id as the zip's comment, and
                                the notebook reads it back from there.
    parity_gpu_run.ipynb        the notebook, as committed at HEAD
    KIT.txt                     the commit, the archive's sha256, and what to do next
Refuses to build if tracked files have uncommitted changes, if `.env` is tracked, or if any committed file
looks like it holds a token. Files are named in the error; their contents are never printed.
"""

from __future__ import annotations

import hashlib
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


def main() -> int:
    root = Path(git("rev-parse", "--show-toplevel").stdout.strip())
    dirty = git("status", "--porcelain", "--untracked-files=no", root=root).stdout.strip()
    if dirty:
        refuse(f"commit or stash these first; the kit is built from HEAD only:\n{dirty}")
    commit = git("rev-parse", "HEAD", root=root).stdout.strip()

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
    (out / "parity_gpu_run.ipynb").write_text(notebook, encoding="utf-8")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    built = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (out / "KIT.txt").write_text(
        f"Parity GPU run kit\n"
        f"commit          {commit}\n"
        f"archive         {archive.name}\n"
        f"archive sha256  {digest}\n"
        f"archive size    {archive.stat().st_size / 1e6:.1f} MB, {len(names)} files\n"
        f"built (UTC)     {built}\n\n"
        f"Next: follow docs/gpu-run/GPU-RUN.md. Upload parity_gpu_run.ipynb to Colab, then upload {archive.name}\n"
        f"when cell 4 asks for it.\n",
        encoding="utf-8",
    )

    windows = subprocess.run(["wslpath", "-w", str(out)], capture_output=True, text=True, check=False)
    print(f"Kit built from commit {commit}")
    print(f"  {archive.name}  sha256 {digest}  ({archive.stat().st_size / 1e6:.1f} MB)")
    print(f"  folder (Linux):   {out}")
    if windows.returncode == 0 and windows.stdout.strip():
        print(f"  folder (Windows): {windows.stdout.strip()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
