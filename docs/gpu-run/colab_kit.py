"""Parity's GPU run kit: install Unsloth's pinned environment on a GPU machine, run the checks, zip the results.

Built for a free Google Colab GPU (Unsloth's own notebooks target Colab), started by `parity_gpu_run.ipynb` from the
root of the unpacked `git archive`. Standard library only: it runs in Colab's own Python, before anything is installed.
Steps, each logged to <out>/logs/<step>.log; any failure stops the run but still writes the zip:

1. platform  The GPU (nvidia-smi), its driver and the CUDA version the driver supports, Python, OS, disk.
2. choose    The lockfile the driver allows. Unsloth's environment (envs/unsloth-2026.11.10) locks PyPI's default
             torch 2.14.1 wheel, built for CUDA 13.0, which needs a driver that supports CUDA 13. On an older
             driver (CUDA 12.x) the kit uses the CUDA 12.6 variant (envs/unsloth-2026.11.10/cu126), which differs
             only in torch, torchvision and their CUDA libraries. Either way it is a committed lockfile, and the
             choice is recorded as a deviation when it isn't the main one.
3. uv        Install uv (pinned) into Colab's Python.
4. sync      `uv sync --locked` from that lockfile into its own virtual environment (Colab's packages aren't used).
5. freeze    Every installed version, compared with the lockfile.
6. smoke     In that environment: torch sees the GPU and runs on it, and Unsloth takes its GPU route
             (UNSLOTH_DISABLE_CPU_FALLBACK=1, so it raises instead of falling back to CPU).
7. checks    docs/gpu-run/run_checks.py: the six Tier 1 checks, both serving modes, for each model.
Then environment.json (what ran, and every deviation from Unsloth's documented install), MANIFEST.json (sha256 of
every file) and one zip.

P-0005 (P0005_RUNS: the repro padded and with padding_free left to Unsloth, then the adapter's pad_to check) runs in
the same environment, two ways:
- `--extra p0005` (notebook cell 6, optional) after a full run;
- `--only p0005` (RUN = "p0005" in the notebook): every step but the checks, then P-0005. Minutes, not about 20.
It writes only into its own folder, P-0005/, with its own MANIFEST.json, and never touches the main results: the zip
is rebuilt with the existing entries copied byte for byte plus that folder, and swapped in only once it is complete
and reads back clean. A failed run is recorded in P-0005/run.json; if anything else fails, the zip is left exactly as
it was.

No tokens or logins. Subprocesses get a scrubbed environment: no variable whose name looks like a secret, no
PYTHONPATH, and Hugging Face's implicit token lookup turned off. Models must be ungated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

FRAMEWORK = "unsloth"
VERSION = "2026.11.10"
MAIN_ENV = f"envs/{FRAMEWORK}-{VERSION}"
CU126_ENV = f"{MAIN_ENV}/cu126"
UV_VERSION = "0.11.3"  # the uv that wrote both lockfiles
STEPS = ("platform", "choose", "uv", "sync", "freeze", "smoke", "checks")  # in order; each is Kit.<name>_step
MODES = ("full", "p0005-only")  # p0005-only: every step but the checks, then the P-0005 runs (minutes, not ~20)


def steps_for(mode: str) -> tuple[str, ...]:
    return STEPS if mode == "full" else tuple(s for s in STEPS if s != "checks")
CALIBRATION = "Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca"
MODEL_SPEC = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+@[0-9a-f]{40}$")
SECRET_NAME = re.compile(r"TOKEN|SECRET|PASSW|CREDENTIAL|API_?KEY|PRIVATE_?KEY|ACCESS_?KEY", re.IGNORECASE)
UNSAFE_PYTHON_VARS = ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONUSERBASE", "VIRTUAL_ENV", "CONDA_PREFIX")

# Unsloth's documented install for Colab, from the recommended Qwen3 notebook's first cell
# (unslothai/notebooks@ebe88117, nb/Qwen3_(14B)-Reasoning-Conversational.ipynb).
DOCUMENTED_COLAB_INSTALL = (
    'pip install sentencepiece protobuf "datasets==4.3.0" "huggingface_hub>=0.34.0" hf_transfer; '
    "pip install --no-deps unsloth_zoo bitsandbytes accelerate peft trl triton unsloth; unsloth install-kernels; "
    'pip install --no-deps --upgrade "torchao>=0.16.0"; pip install transformers==4.57.6; '
    "pip install --no-deps trl==0.22.2 (on top of Colab's preinstalled torch)"
)
DOCUMENTED_COLAB_SOURCE = (
    "https://github.com/unslothai/notebooks/blob/ebe88117bb560d296910f6e733693294d630f024/"
    "nb/Qwen3_(14B)-Reasoning-Conversational.ipynb (install cell)"
)
DOCUMENTED_GENERIC_INSTALL = "uv venv unsloth_env --python 3.13; uv pip install unsloth --torch-backend=auto"

# Deviations that hold for every GPU run of this kit, whatever the machine.
FIXED_DEVIATIONS = [
    "Installed from Parity's lockfile into its own uv virtual environment, not with Unsloth's documented Colab cell "
    f"({DOCUMENTED_COLAB_INSTALL}). So every version is pinned (rule 7), and Colab's preinstalled torch and packages "
    "are not used.",
    "`unsloth install-kernels` not run: it installs optional compute-kernel wheels (flash-attn, xformers, "
    "causal-conv1d, mamba-ssm; unsloth/studio/backend/utils/kernel_install.py). Parity runs no forward pass or "
    "training step, so they are off the data path being checked.",
    "TRL 1.15.0 and transformers 5.17.0, not the notebook's trl==0.22.2 and transformers==4.57.6: this environment "
    "holds the 2026-11-23 spike's resolution of unsloth==2026.11.10 (as in the CPU run; "
    "envs/unsloth-2026.11.10/environment.json).",
    "Python 3.12, not the documented 3.13: Parity's core requires 3.12, and the lock resolves Unsloth on 3.12.",
    "The model is a tiny from_config Qwen3-shaped model (1 layer, random weights) saved next to the release's real "
    "tokenizer, template and config files, loaded through FastLanguageModel.from_pretrained with the notebook's "
    "load_in_4bit=True. No weights are downloaded (rule 6).",
]


@dataclass
class Step:
    name: str
    ok: bool
    seconds: float
    detail: str = ""


class Kit:
    def __init__(self, root: Path, out: Path, models: list[str], commit: str, archive_sha256: str, mode: str = "full"):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.root, self.out, self.models, self.mode = root, out, models, mode
        self.commit, self.archive_sha256 = commit, archive_sha256
        self.logs = out / "logs"
        self.steps: list[Step] = []
        self.env, self.scrubbed = scrubbed_environment(os.environ, cache_root=out.parent)
        self.facts: dict = {"platform": {}, "deviations": list(FIXED_DEVIATIONS)}

    # Running things

    def run(self, step: str, command: list[str], cwd: Path | None = None) -> tuple[int, str]:
        """Run a command, streaming its output to the notebook and to logs/<step>.log."""
        return stream(command, cwd or self.root, self.env, self.logs / f"{step}.log")

    def step(self, name: str, fn) -> bool:
        print(f"\n===== {name} =====", flush=True)
        started = time.time()
        try:
            detail = fn() or ""
            ok = True
        except KitError as exc:
            detail, ok = str(exc), False
        except Exception as exc:  # noqa: BLE001 - recorded in status.json and the zip, never hidden
            detail, ok = f"{type(exc).__name__}: {exc}", False
        self.steps.append(Step(name, ok, round(time.time() - started, 1), detail))
        print(f"----- {name}: {'ok' if ok else 'FAILED'} {detail}", flush=True)
        return ok

    # The steps

    def platform_step(self) -> str:
        no_gpu = "no GPU. In Colab: Runtime > Change runtime type > T4 GPU, then run every cell again."
        if shutil.which("nvidia-smi") is None:
            raise KitError(f"nvidia-smi not found: {no_gpu}")
        code, text = self.run("platform", ["nvidia-smi"])
        if code != 0:
            raise KitError(f"nvidia-smi failed: {no_gpu}")
        query = "name,driver_version,memory.total,compute_cap"
        _, csv = self.run("platform", ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"])
        gpu = [part.strip() for part in csv.strip().splitlines()[0].split(",")]
        cuda = driver_cuda_version(text)
        self.facts["platform"] = {
            "gpu": gpu[0], "driver": gpu[1], "gpu_memory": gpu[2], "compute_capability": gpu[3],
            "driver_cuda": ".".join(map(str, cuda)), "os": platform.platform(), "machine": platform.machine(),
            "colab": any(k.startswith("COLAB_") for k in os.environ),
            "kit_python": sys.version.split()[0], "disk_free_gb": round(shutil.disk_usage(self.root).free / 1e9, 1),
        }  # fmt: skip
        return f"{gpu[0]}, driver {gpu[1]} (CUDA {self.facts['platform']['driver_cuda']})"

    def choose_step(self) -> str:
        cuda = tuple(int(x) for x in self.facts["platform"]["driver_cuda"].split("."))
        env_dir, reason, deviation = choose_environment(cuda)  # type: ignore[arg-type]
        self.facts.update(env_dir=env_dir, env_lock=f"{env_dir}/uv.lock", lock_choice=reason)
        self.facts["env_lock_sha256"] = sha256_file(self.root / env_dir / "uv.lock")
        if deviation:
            self.facts["deviations"].append(deviation)
        return f"{env_dir}/uv.lock: {reason}"

    def uv_step(self) -> str:
        code, _ = self.run("uv", [sys.executable, "-m", "pip", "install", "--quiet", f"uv=={UV_VERSION}"])
        if code != 0:
            raise KitError(f"pip install uv=={UV_VERSION} failed")
        return f"uv {UV_VERSION}"

    def sync_step(self) -> str:
        uv = [sys.executable, "-m", "uv"]
        code, _ = self.run("sync", [*uv, "sync", "--locked", "--python", "3.12"], cwd=self.root / self.facts["env_dir"])
        if code != 0:
            raise KitError("uv sync --locked failed (see logs/sync.log)")
        python = self.venv_python()
        _, where = self.run("sync", [str(python), "-c", "import sys; print(sys.version.split()[0]); print(sys.base_prefix)"])
        version, base = where.strip().splitlines()[-2:]
        managed = "/uv/python/" in base or "/.local/share/uv/" in base
        self.facts["platform"]["env_python"] = {"version": version, "base_prefix": base, "uv_managed": managed}
        if managed:
            self.facts["deviations"].append(
                f"Python {version} is a uv-managed CPython ({base}), not the machine's own: the machine has no 3.12."
            )
        return f"Python {version} ({'uv-managed' if managed else 'system'})"

    def smoke_step(self) -> str:
        code, text = self.run("smoke", [str(self.venv_python()), "-c", SMOKE_TEST])
        if code != 0:
            raise KitError("the environment can't use the GPU, or Unsloth didn't take its GPU route (logs/smoke.log)")
        facts = marked_json(text)
        self.facts["platform"]["torch"] = facts
        return f"torch {facts['torch']} (CUDA {facts['torch_cuda']}) on {facts['device']}; Unsloth {facts['unsloth_device_type']}"

    def checks_step(self) -> str:
        command = [
            str(self.venv_python()), "docs/gpu-run/run_checks.py", "--models", ",".join(self.models),
            "--out", str(self.out / "results"), "--env-lock", self.facts["env_lock"],
        ]  # fmt: skip
        code, _ = self.run("checks", command)
        if code != 0:
            raise KitError(f"run_checks.py exited {code} (logs/checks.log and results/*/run.json)")
        return f"{len(self.models)} model(s)"

    def freeze_step(self) -> str:
        code, text = self.run("freeze", [str(self.venv_python()), "-c", FREEZE, self.facts["env_lock"]])
        if code != 0:
            raise KitError("couldn't list installed versions")
        report = marked_json(text)
        self.facts["resolved"] = report["installed"]
        self.facts["lock_mismatches"] = report["mismatches"]
        if report["mismatches"]:
            raise KitError(f"installed versions differ from the lockfile: {report['mismatches']}")
        return f"{len(report['installed'])} packages, all as locked"

    def venv_python(self) -> Path:
        return self.root / self.facts["env_dir"] / ".venv" / "bin" / "python"

    # Writing the results

    def environment(self) -> dict:
        variant = self.facts.get("env_dir") == CU126_ENV
        on_gpu = self.facts["platform"].get("torch", {}).get("unsloth_device_type") == "cuda"
        return {
            "framework": FRAMEWORK,
            "version": VERSION,
            "backend": "gpu" if on_gpu else None,  # set only once the smoke step saw Unsloth's CUDA route
            "lockfile": self.facts.get("env_lock"),
            "lockfile_sha256": self.facts.get("env_lock_sha256"),
            "lock_choice": self.facts.get("lock_choice"),
            "install": {
                "documented_colab": DOCUMENTED_COLAB_INSTALL,
                "documented_colab_source": DOCUMENTED_COLAB_SOURCE,
                "documented_generic": DOCUMENTED_GENERIC_INSTALL,
                "doc": "https://unsloth.ai/docs/get-started/install/pip-install",
                "ours": f"uv {UV_VERSION}: `uv sync --locked --python 3.12` in {self.facts.get('env_dir')} "
                + ("(the CUDA 12.6 variant)" if variant else "(Unsloth's environment, as on CPU)"),
            },
            "deviations": self.facts["deviations"],
            "runtime_settings": [
                "UNSLOTH_DISABLE_CPU_FALLBACK=1: Unsloth raises instead of silently falling back to CPU.",
                "UNSLOTH_DISABLE_STATISTICS=1 (the adapter also sets it): Unsloth's documented statistics opt-out.",
                "HF_HUB_DISABLE_IMPLICIT_TOKEN=1 and no token variables: nothing is downloaded with a login.",
                "PYTHONPATH and other interpreter-path variables removed, so Colab's packages can't leak in.",
            ],
            "scrubbed_variables": sorted(self.scrubbed),  # names only, never values
            "platform": self.facts["platform"],
            "resolved": self.facts.get("resolved", {}),
            "lock_mismatches": self.facts.get("lock_mismatches"),
            "archive": {"commit": self.commit, "sha256": self.archive_sha256},
            "models": self.models,
            "mode": self.mode,
        }

    def finish(self) -> Path:
        """Write environment.json, status.json and MANIFEST.json, then zip the whole output folder."""
        self.out.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        archive = self.out.parent / f"parity-gpu-results-{self.commit[:12]}-{stamp}.zip"
        (self.out / "environment.json").write_text(json.dumps(self.environment(), indent=1) + "\n", encoding="utf-8")
        status = {
            "mode": self.mode,
            "ok": bool(self.steps) and all(s.ok for s in self.steps) and len(self.steps) == len(steps_for(self.mode)),
            "steps": [s.__dict__ for s in self.steps],
            "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        (self.out / "status.json").write_text(json.dumps(status, indent=1) + "\n", encoding="utf-8")
        manifest = {
            "schema": "parity.gpu_kit/v0",
            "commit": self.commit,
            "archive_sha256": self.archive_sha256,
            "files": manifest_of(self.out),
        }
        (self.out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(self.out.rglob("*")):
                if path.is_file():
                    zf.write(path, Path(archive.stem) / path.relative_to(self.out))
        return archive


class KitError(RuntimeError):
    """A step failed in a way the kit understands; the message says what to do."""


# Helpers (tested in tests/test_gpu_kit.py)


def stream(command: list[str], cwd: Path, env: dict[str, str], log_path: Path) -> tuple[int, str]:
    """Run a command, printing its output as it comes and appending it to log_path. Returns (exit code, output)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"$ {' '.join(command)}\n")
        proc = subprocess.Popen(
            command, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
        )  # fmt: skip
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
            lines.append(line)
        code = proc.wait()
        log.write(f"[exit {code}]\n")
    return code, "".join(lines)


def driver_cuda_version(nvidia_smi: str) -> tuple[int, int]:
    """The CUDA version the driver supports, from nvidia-smi's header ("CUDA Version: 12.4")."""
    match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", nvidia_smi)
    if not match:
        raise KitError("nvidia-smi didn't report a CUDA version")
    return int(match.group(1)), int(match.group(2))


def choose_environment(driver_cuda: tuple[int, int]) -> tuple[str, str, str | None]:
    """(environment folder, why, deviation or None) for the CUDA version the driver supports."""
    shown = f"{driver_cuda[0]}.{driver_cuda[1]}"
    if driver_cuda >= (13, 0):
        return MAIN_ENV, f"the driver supports CUDA {shown}, enough for the main lock's CUDA 13.0 torch", None
    if driver_cuda >= (12, 0):
        return (
            CU126_ENV,
            f"the driver supports CUDA {shown}, too old for the main lock's CUDA 13.0 torch",
            f"CUDA 12.6 variant lock ({CU126_ENV}/uv.lock): the driver supports CUDA {shown}, and PyPI's torch 2.14.1 "
            "wheel needs CUDA 13. torch 2.14.1+cu126 and torchvision 0.29.1+cu126 (PyTorch's CUDA 12.6 index has "
            "no 0.29.2) with their nvidia-*-cu12 libraries; every other package is as in the main lock. Unsloth's "
            "documented `--torch-backend=auto` would also pick a CUDA build that matches the driver.",
        )
    raise KitError(f"the driver supports only CUDA {shown}; torch 2.14.1 needs CUDA 12 or newer")


def scrubbed_environment(environ: dict[str, str] | os._Environ, cache_root: Path) -> tuple[dict[str, str], list[str]]:
    """A copy of the environment with secrets and interpreter-path variables removed (names returned, not values)."""
    removed = sorted(k for k in environ if SECRET_NAME.search(k) or k in UNSAFE_PYTHON_VARS)
    env = {k: v for k, v in environ.items() if k not in removed}
    env.update(
        HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        HF_HOME=str(cache_root / "parity-hf-cache"),
        UV_CACHE_DIR=str(cache_root / "parity-uv-cache"),
        UNSLOTH_DISABLE_CPU_FALLBACK="1",
        UNSLOTH_DISABLE_STATISTICS="1",
        PYTHONUNBUFFERED="1",
    )
    return env, removed


def marked_json(output: str) -> dict:
    """The JSON a helper printed after "PARITY_JSON=" (libraries may print anything around it)."""
    for line in reversed(output.splitlines()):
        if line.startswith("PARITY_JSON="):
            return json.loads(line.removeprefix("PARITY_JSON="))
    raise KitError("the helper printed no PARITY_JSON= line")


def manifest_of(folder: Path) -> dict[str, str]:
    return {
        str(p.relative_to(folder)): sha256_file(p)
        for p in sorted(folder.rglob("*"))
        if p.is_file() and p.name != "MANIFEST.json"
    }


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_models(text: str) -> list[str]:
    models = [m.strip() for m in text.split(",") if m.strip()]
    bad = [m for m in models if not MODEL_SPEC.match(m)]
    if not models or bad:
        raise KitError(f"models must be org/name@<40-character commit>, got {bad or 'nothing'}")
    return models


# Run inside the synced environment

SMOKE_TEST = r"""
import json, torch
assert torch.cuda.is_available(), "torch can't see the GPU (wrong CUDA build for this driver?)"
x = torch.arange(6.0, device="cuda").reshape(2, 3)
assert float((x @ x.T).sum().item()) == 83.0, "a small CUDA matmul gave the wrong answer"
name = torch.cuda.get_device_name(0)
capability = ".".join(map(str, torch.cuda.get_device_capability(0)))
import unsloth  # UNSLOTH_DISABLE_CPU_FALLBACK=1: raises rather than falling back to CPU
assert str(unsloth.DEVICE_TYPE) == "cuda", f"Unsloth took {unsloth.DEVICE_TYPE!r}, not its CUDA route"
assert not getattr(unsloth, "_IS_CPU_BACKEND", False), "Unsloth selected its CPU backend"
print("PARITY_JSON=" + json.dumps({"torch": torch.__version__, "torch_cuda": torch.version.cuda, "device": name,
                  "capability": capability, "arch_list": torch.cuda.get_arch_list(),
                  "unsloth": unsloth.__version__, "unsloth_device_type": str(unsloth.DEVICE_TYPE)}))
"""

FREEZE = r"""
import json, sys, tomllib
from importlib.metadata import distributions
from packaging.utils import canonicalize_name
installed = {canonicalize_name(d.metadata["Name"]): d.version for d in distributions()}
lock = tomllib.load(open(sys.argv[1], "rb"))
locked = {canonicalize_name(p["name"]): p["version"] for p in lock["package"] if "registry" in p.get("source", {})}
mismatches = {n: {"locked": v, "installed": installed.get(n)} for n, v in locked.items()
              if n in installed and installed[n] != v}
print("PARITY_JSON=" + json.dumps({"installed": dict(sorted(installed.items())), "mismatches": mismatches}))
"""


# P-0005 on the GPU route: its own folder in the results zip, never part of the main run's results

P0005_FOLDER = "P-0005"
# name -> (script, arguments, what it shows, the line that proves it ran to the end)
P0005_RUNS = {
    "padded": (
        "findings/P-0005/repro.py", [],
        "The repro with packing=False, padding_free=False, as the recommended Qwen3 notebook sets. Its rule: exit 1 if "
        "pad_to_multiple_of was dropped, 0 if kept.",
        "after  train_on_responses_only:",
    ),
    "padding_free_unset": (
        "findings/P-0005/repro.py", ["--padding-free-unset"],
        "The repro with packing and padding_free left to Unsloth's defaults, max_steps=60 (the Qwen3-4B-Thinking-2507 "
        "notebook). The 'batching:' line shows what Unsloth chose; under packing or padding-free the batch is one "
        "flattened row, so read the printed lines, not the exit code.",
        "after  train_on_responses_only:",
    ),
    "adapter_check": (
        "findings/P-0005/adapter_check.py", [],
        "Rule 4: Parity's own adapter at the pad_to t1.special_tokens asks, with objective all (no "
        "train_on_responses_only) and assistant_only. Exit 0 if the mapping reaches the trainer (objective all pads "
        "to pad_to), 1 if it doesn't (the adapter is the suspect).",
        "mapping:",
    ),
}  # fmt: skip
P0005_LINES = (
    "SFTConfig(", "batching:", "before train_on_responses_only:", "after  train_on_responses_only:", "padding:",
    "backend ", "objective ", "mapping:",
)  # fmt: skip


def p0005_outcome(code: int, output: str, result_line: str) -> dict:
    """What one run showed. It completed only if it printed its result line: a crash also exits 1."""
    lines = [line for line in output.splitlines() if line.startswith(P0005_LINES)]
    return {"completed": any(line.startswith(result_line) for line in lines), "exit_code": code, "lines": lines}


def main_run_environment(out: Path) -> str:
    """The environment folder the main run synced, once its smoke step confirmed Unsloth's GPU route."""
    try:
        status = json.loads((out / "status.json").read_text(encoding="utf-8"))
        environment = json.loads((out / "environment.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise KitError(f"no finished main run in {out} ({exc}); run cell 5 first") from exc
    if not any(step["name"] == "smoke" and step["ok"] for step in status["steps"]):
        raise KitError("the main run never confirmed Unsloth's GPU route (its smoke step didn't pass), so the repro "
                       "would not be a GPU run")  # fmt: skip
    return str(Path(environment["lockfile"]).parent)


def add_folder_to_zip(zip_path: Path, folder: Path, name: str) -> None:
    """Rebuild zip_path with every existing entry copied byte for byte, except an older <top>/<name>/, plus folder's
    files under <top>/<name>/. The new zip replaces the old one only after it reads back clean."""
    tmp = zip_path.with_name(zip_path.name + ".tmp")
    try:
        with zipfile.ZipFile(zip_path) as src:
            tops = {entry.split("/", 1)[0] for entry in src.namelist()}
            if len(tops) != 1:
                raise KitError(f"{zip_path.name} doesn't have one top folder: {sorted(tops)}")
            top = tops.pop()
            with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as dst:
                for info in src.infolist():
                    if not info.filename.startswith(f"{top}/{name}/"):
                        dst.writestr(info, src.read(info))
                for path in sorted(folder.rglob("*")):
                    if path.is_file():
                        dst.write(path, f"{top}/{name}/{path.relative_to(folder).as_posix()}")
        with zipfile.ZipFile(tmp) as check:
            if check.testzip() is not None:
                raise KitError("the rebuilt zip doesn't read back clean")
        os.replace(tmp, zip_path)
    finally:
        tmp.unlink(missing_ok=True)


def run_p0005(root: Path, out: Path, zip_path: Path, python: Path | None = None, runs: dict | None = None) -> int:
    """Run P0005_RUNS in the main run's environment, record them in <out>/P-0005/, and add that folder to the results
    zip. `python` and `runs` exist for tests. Returns 0 only if every run completed."""
    if not zip_path.is_file():
        print(f"REFUSED: no results zip at {zip_path}; run cell 5 first.")
        return 2
    folder = out / P0005_FOLDER
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    record: dict = {
        "finding": "findings/P-0005/record.md",
        "note": "Separate from the verdicts; it never changes the main run's results or status.",
        "runs": {},
    }
    try:
        env_dir = main_run_environment(out)
        record["environment"] = env_dir
        python = python or root / env_dir / ".venv" / "bin" / "python"
        env, _ = scrubbed_environment(os.environ, cache_root=out.parent)
        for name, (script, extra_args, meaning, result_line) in (runs or P0005_RUNS).items():
            print(f"\n===== P-0005: {name} =====", flush=True)
            started = time.time()
            code, text = stream([str(python), str(root / script), *extra_args], root, env, folder / f"{name}.log")
            record["runs"][name] = {
                "script": str(script), "arguments": extra_args, "meaning": meaning,
                **p0005_outcome(code, text, result_line), "seconds": round(time.time() - started, 1),
            }  # fmt: skip
    except Exception as exc:  # noqa: BLE001 - recorded in P-0005/run.json; the main results are never touched
        record["error"] = f"{type(exc).__name__}: {exc}"
        print(f"P-0005: {record['error']}", flush=True)
    (folder / "run.json").write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    (folder / "MANIFEST.json").write_text(json.dumps({"files": manifest_of(folder)}, indent=1) + "\n", encoding="utf-8")
    try:
        add_folder_to_zip(zip_path, folder, P0005_FOLDER)
    except Exception as exc:  # noqa: BLE001 - the main zip stays exactly as it was
        print(f"P-0005: couldn't add its folder to the zip ({type(exc).__name__}: {exc}). "
              f"The results zip is unchanged: {zip_path}")  # fmt: skip
        return 1
    completed = not record.get("error") and all(r["completed"] for r in record["runs"].values())
    print(f"\nP-0005 {'done' if completed else 'did not complete (see P-0005/run.json)'}; added to {zip_path}")
    print(f"RESULT_ZIP={zip_path}")
    return 0 if completed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", default=CALIBRATION, help="comma-separated org/name@<40-char revision>")
    parser.add_argument("--out", type=Path, default=Path("/content/parity-gpu-out"))
    parser.add_argument("--commit", help="the commit the archive was made from (its zip comment)")
    parser.add_argument("--archive-sha256", help="sha256 of the uploaded archive")
    parser.add_argument("--only", choices=["p0005"], help="skip the checks: install, smoke, then P-0005 (minutes)")
    parser.add_argument("--extra", choices=["p0005"], help="optional, after a full run: P-0005 into its zip")
    parser.add_argument("--zip", type=Path, help="with --extra: the results zip to add the P-0005 folder to")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    if args.extra:
        if args.zip is None:
            print("REFUSED: --extra needs --zip, the main run's results zip.")
            return 2
        return run_p0005(root, args.out, args.zip)
    if not args.commit or not re.fullmatch(r"[0-9a-f]{40}", args.commit) or not args.archive_sha256:
        print(f"REFUSED: {args.commit!r} is not a commit id (or no --archive-sha256); upload the archive build_kit.py made.")
        return 2
    try:
        models = parse_models(args.models)
    except KitError as exc:
        print(f"REFUSED: {exc}")
        return 2
    if args.out.exists():
        shutil.rmtree(args.out)
    mode = "p0005-only" if args.only == "p0005" else "full"
    steps = steps_for(mode)
    kit = Kit(root, args.out, models, args.commit, args.archive_sha256, mode=mode)
    for name in steps:
        if not kit.step(name, getattr(kit, f"{name}_step")):
            break  # a failed step stops the run; the zip below still records how far it got
    archive = kit.finish()
    ok = all(s.ok for s in kit.steps) and len(kit.steps) == len(steps)
    print(f"\n{'DONE' if ok else 'STOPPED EARLY'}: {archive}")
    print(f"RESULT_ZIP={archive}")
    if mode == "p0005-only":
        if not ok:
            print("P-0005 not run: the install or the smoke step didn't pass (see the zip's logs).")
            return 1
        return run_p0005(root, args.out, archive)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
