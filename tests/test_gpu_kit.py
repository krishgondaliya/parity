"""The GPU run kit (docs/gpu-run/): lockfile choice, secret scrubbing, the results zip, the notebook, the variant lock."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
import tomllib
import zipfile
from pathlib import Path

import pytest

from parity.targets import CALIBRATION

ROOT = Path(__file__).resolve().parents[1]
KIT_DIR = ROOT / "docs" / "gpu-run"


def load(name: str):
    spec = importlib.util.spec_from_file_location(f"gpu_kit_{name}", KIT_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses need the module registered
    spec.loader.exec_module(module)
    return module


kit = load("colab_kit")
build = load("build_kit")

T4_HEADER = "| NVIDIA-SMI 550.54.15    Driver Version: 550.54.15    CUDA Version: 12.4     |"


# Which lockfile the driver allows


def test_the_driver_cuda_version_is_read_from_nvidia_smi():
    assert kit.driver_cuda_version(T4_HEADER) == (12, 4)
    assert kit.driver_cuda_version("Driver Version: 580.65.06   CUDA Version: 13.0") == (13, 0)
    with pytest.raises(kit.KitError):
        kit.driver_cuda_version("no header here")


def test_a_cuda_13_driver_uses_the_main_lock_with_no_deviation():
    env_dir, reason, deviation = kit.choose_environment((13, 0))
    assert (env_dir, deviation) == ("envs/unsloth-2026.11.10", None) and "13.0" in reason


def test_a_cuda_12_driver_uses_the_cuda_12_6_variant_and_records_it():
    env_dir, _, deviation = kit.choose_environment((12, 4))
    assert env_dir == "envs/unsloth-2026.11.10/cu126"
    assert "torch 2.14.1+cu126" in deviation and "torchvision 0.29.1" in deviation


def test_an_older_driver_stops_the_run():
    with pytest.raises(kit.KitError, match="CUDA 12 or newer"):
        kit.choose_environment((11, 8))


def test_both_lockfiles_are_committed_where_the_kit_looks():
    for env_dir in (kit.MAIN_ENV, kit.CU126_ENV):
        assert (ROOT / env_dir / "uv.lock").is_file() and (ROOT / env_dir / "pyproject.toml").is_file()


def test_the_variant_differs_from_the_main_lock_only_in_torch_torchvision_and_cuda_libraries():
    """Everything on the data path (Unsloth, TRL, transformers, tokenizers, datasets, PEFT) is held identical."""

    def linux_registry_packages(lock_path: Path) -> dict[str, str]:
        lock = tomllib.loads(lock_path.read_text(encoding="utf-8"))
        return {p["name"]: p["version"] for p in lock["package"] if "registry" in p.get("source", {})}

    main = linux_registry_packages(ROOT / kit.MAIN_ENV / "uv.lock")
    variant = linux_registry_packages(ROOT / kit.CU126_ENV / "uv.lock")
    cuda = re.compile(r"^(nvidia-|cuda-)")
    differing = {n for n in variant if main.get(n) != variant[n] and not cuda.match(n)}
    assert differing == {"torch", "torchvision"}
    assert (variant["torch"], variant["torchvision"]) == ("2.14.1+cu126", "0.29.1+cu126")
    for name in ("unsloth", "unsloth-zoo", "trl", "transformers", "tokenizers", "datasets", "peft", "bitsandbytes",
                 "triton", "xformers", "accelerate"):  # fmt: skip
        assert variant[name] == main[name], name


# No secrets, ever


def test_secrets_and_interpreter_paths_never_reach_the_subprocesses(tmp_path):
    fake = "hf_" + "Q" * 34  # built at run time, so this file never matches the kit's token scan
    environ = {
        "PATH": "/usr/bin", "LD_LIBRARY_PATH": "/usr/lib64-nvidia", "HF_TOKEN": fake, "GITHUB_TOKEN": "x",
        "AWS_SECRET_ACCESS_KEY": "y", "OPENAI_API_KEY": "z", "PYTHONPATH": "/env/python", "COLAB_GPU": "1",
    }  # fmt: skip
    env, removed = kit.scrubbed_environment(environ, cache_root=tmp_path)
    assert removed == ["AWS_SECRET_ACCESS_KEY", "GITHUB_TOKEN", "HF_TOKEN", "OPENAI_API_KEY", "PYTHONPATH"]
    assert fake not in json.dumps(env)
    assert (env["PATH"], env["LD_LIBRARY_PATH"], env["COLAB_GPU"]) == ("/usr/bin", "/usr/lib64-nvidia", "1")
    assert env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == env["UNSLOTH_DISABLE_CPU_FALLBACK"] == "1"


def test_the_results_zip_has_a_manifest_and_no_secret_values(tmp_path, monkeypatch):
    fake = "hf_" + "Z" * 34
    monkeypatch.setenv("HF_TOKEN", fake)
    out = tmp_path / "parity-gpu-out"
    run = kit.Kit(ROOT, out, [CALIBRATION.spec], "a" * 40, "b" * 64)
    run.steps.append(kit.Step("platform", False, 0.1, "nvidia-smi not found"))
    (out / "results" / "m").mkdir(parents=True)
    (out / "results" / "m" / "run.json").write_text("{}", encoding="utf-8")
    archive = run.finish()

    assert archive.name.startswith("parity-gpu-results-aaaaaaaaaaaa-") and archive.suffix == ".zip"
    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
        top = archive.stem
        manifest = json.loads(zf.read(f"{top}/MANIFEST.json"))
        environment = json.loads(zf.read(f"{top}/environment.json"))
        status = json.loads(zf.read(f"{top}/status.json"))
        assert all(fake not in zf.read(n).decode("utf-8") for n in names)
    assert set(manifest["files"]) == {n.removeprefix(f"{top}/") for n in names} - {"MANIFEST.json"}
    assert manifest["commit"] == "a" * 40 and environment["archive"]["sha256"] == "b" * 64
    assert "HF_TOKEN" in environment["scrubbed_variables"]  # the name is recorded, never the value
    assert status["ok"] is False and status["steps"][0]["detail"] == "nvidia-smi not found"
    assert environment["backend"] is None  # the smoke step never confirmed Unsloth's GPU route
    assert environment["models"] == [CALIBRATION.spec]


def test_helper_output_is_read_from_its_marker_line_only():
    printed = 'Unsloth banner\nPARITY_JSON={"torch": "2.14.1"}\nsome later warning'
    assert kit.marked_json(printed) == {"torch": "2.14.1"}
    with pytest.raises(kit.KitError):
        kit.marked_json("no marker")


@pytest.mark.parametrize(
    "token",
    ["hf_" + "a" * 34, "ghp_" + "b" * 36, "github_pat_" + "c" * 60, "sk-" + "d" * 40, "AKIA" + "E" * 16],
)
def test_the_build_refuses_files_that_look_like_they_hold_a_token(token):
    assert any(re.search(p, f"key = {token}") for p in build.TOKEN_PATTERNS)


# Models and the notebook


def test_models_must_be_pinned_releases():
    assert kit.parse_models(f" {CALIBRATION.spec} ,") == [CALIBRATION.spec]
    for bad in ("Qwen/Qwen3-0.6B@main", "Qwen3-0.6B", ""):
        with pytest.raises(kit.KitError):
            kit.parse_models(bad)
    assert kit.CALIBRATION == CALIBRATION.spec


def test_the_notebook_defaults_to_the_calibration_model_and_runs_the_kit():
    nb = json.loads((KIT_DIR / "parity_gpu_run.ipynb").read_text(encoding="utf-8"))
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    for source in code:
        compile(source, "<cell>", "exec")
    namespace: dict = {}
    exec(code[0], namespace)  # the parameters cell only assigns MODELS
    assert namespace["MODELS"] == [CALIBRATION.spec]
    assert any("colab_kit.py" in source for source in code)
    assert nb["metadata"]["accelerator"] == "GPU"


def test_the_kit_runs_every_step_in_order():
    assert kit.STEPS == ("platform", "choose", "uv", "sync", "freeze", "smoke", "checks")
    assert all(callable(getattr(kit.Kit, f"{name}_step")) for name in kit.STEPS)


def test_on_a_cuda_12_4_t4_the_kit_records_the_gpu_and_picks_the_variant(tmp_path, monkeypatch):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    smi = fake_bin / "nvidia-smi"
    smi.write_text(
        "#!/bin/sh\n"
        'case "$1" in --query-gpu=*) echo "Tesla T4, 550.54.15, 15360 MiB, 7.5" ;; *) echo "' + T4_HEADER + '" ;; esac\n',
        encoding="utf-8",
    )
    smi.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake_bin}:/usr/bin:/bin")
    run = kit.Kit(ROOT, tmp_path / "out", [CALIBRATION.spec], "a" * 40, "b" * 64)
    assert run.step("platform", run.platform_step) and run.step("choose", run.choose_step)
    platform_facts = run.facts["platform"]
    assert (platform_facts["gpu"], platform_facts["driver_cuda"], platform_facts["compute_capability"]) == ("Tesla T4", "12.4", "7.5")
    assert run.facts["env_lock"] == "envs/unsloth-2026.11.10/cu126/uv.lock"
    assert any("CUDA 12.6 variant" in d for d in run.facts["deviations"])
    assert (tmp_path / "out" / "logs" / "platform.log").read_text(encoding="utf-8").count("$ nvidia-smi") == 2
