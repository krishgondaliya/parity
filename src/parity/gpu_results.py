"""Verify a results zip from the GPU run kit (docs/gpu-run/) before anything in it is kept.

The zip is untrusted data. It is unpacked into a folder the caller makes fresh, and entry names that are absolute,
climb out with "..", use backslashes, or are links are refused. Then `verify_zip` checks:
- every file against the zip's MANIFEST.json, and P-0005/MANIFEST.json when the P-0005 runs added that folder;
- the commit exists in this repo, and `git archive --format=zip <commit>` rebuilt here has the sha256 Colab recorded
  for the uploaded code, so the code that ran is exactly that commit;
- the run's mode, that every step passed, that Unsloth took its CUDA route (the smoke step), that the lockfile's
  sha256 is the committed one at that commit, and that every installed version was as locked.
Callers add their own checks on what the run produced (verdicts, P-0005 runs).
"""

from __future__ import annotations

import hashlib
import json
import posixpath
import stat
import subprocess
import zipfile
from pathlib import Path


class Refused(RuntimeError):
    """The zip failed a check; nothing from it may be kept."""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git(repo: Path, *args: str) -> bytes:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=True).stdout


def safe_extract(zip_path: Path, dest: Path) -> Path:
    """Unpack into dest, refusing unsafe entry names. Returns the zip's one top folder."""
    with zipfile.ZipFile(zip_path) as zf:
        tops = set()
        for info in zf.infolist():
            name = info.filename
            if name.startswith("/") or "\\" in name or posixpath.normpath(name).startswith(".."):
                raise Refused(f"unsafe entry name {name!r}")
            if stat.S_ISLNK(info.external_attr >> 16):
                raise Refused(f"link entry {name!r}")
            tops.add(name.split("/", 1)[0])
        if len(tops) != 1:
            raise Refused(f"expected one top folder, got {sorted(tops)}")
        zf.extractall(dest)
    return dest / tops.pop()


def verify_zip(zip_path: Path, root: Path, repo: Path, mode: str) -> dict:
    """The checks in the module docstring. Returns what was verified; raises Refused on the first failure."""
    facts: dict = {"zip": zip_path.name, "zip_sha256": sha256(zip_path.read_bytes())}
    manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
    listed = dict(manifest["files"])
    if (root / "P-0005" / "MANIFEST.json").is_file():
        extra = json.loads((root / "P-0005" / "MANIFEST.json").read_text(encoding="utf-8"))["files"]
        listed.update({f"P-0005/{k}": v for k, v in extra.items()})
    present = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()} - {"MANIFEST.json", "P-0005/MANIFEST.json"}
    if present != set(listed):
        raise Refused(f"files and manifests differ: {sorted(present ^ set(listed))}")
    bad = [name for name, digest in listed.items() if sha256((root / name).read_bytes()) != digest]
    if bad:
        raise Refused(f"sha256 mismatch: {bad}")
    facts["files_verified"] = len(listed)

    commit = manifest["commit"]
    environment = json.loads((root / "environment.json").read_text(encoding="utf-8"))
    if environment["archive"]["commit"] != commit:
        raise Refused("MANIFEST.json and environment.json name different commits")
    try:
        git(repo, "cat-file", "-e", f"{commit}^{{commit}}")
    except subprocess.CalledProcessError as exc:
        raise Refused(f"commit {commit[:12]} is not in this repo") from exc
    if not sha256(git(repo, "archive", "--format=zip", commit)) == manifest["archive_sha256"] == environment["archive"]["sha256"]:
        raise Refused("git archive of the commit doesn't match the sha256 recorded for the uploaded code")
    facts.update(commit=commit, archive_sha256=manifest["archive_sha256"], archive_rebuilt_here=True)

    status = json.loads((root / "status.json").read_text(encoding="utf-8"))
    run_mode = status.get("mode", "full")  # kits before the P-0005-only mode wrote no mode
    if run_mode != mode:
        raise Refused(f"expected a {mode!r} run, got {run_mode!r}")
    if not status["ok"]:
        raise Refused("the run didn't finish (status.json ok is false)")
    smoke = environment["platform"].get("torch", {})
    if environment["backend"] != "gpu" or smoke.get("unsloth_device_type") != "cuda":
        raise Refused("the smoke step didn't confirm Unsloth's CUDA route")
    lock = environment["lockfile"]
    if sha256(git(repo, "show", f"{commit}:{lock}")) != environment["lockfile_sha256"]:
        raise Refused(f"{lock} at {commit[:12]} isn't the lockfile the run recorded")
    if environment["lock_mismatches"]:
        raise Refused(f"installed versions differ from the lockfile: {environment['lock_mismatches']}")
    facts.update(mode=run_mode, lockfile=lock, lockfile_sha256=environment["lockfile_sha256"],
                 gpu=environment["platform"]["gpu"], driver=environment["platform"]["driver"], torch=smoke["torch"],
                 unsloth_device_type="cuda")  # fmt: skip
    return facts
