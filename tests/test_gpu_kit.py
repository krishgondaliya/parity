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


# The optional P-0005 repro: its own folder in the zip, never touching the main results


def finished_main_run(tmp_path: Path, smoke_ok: bool = True) -> tuple[Path, Path]:
    """A main run's output folder and results zip, as cell 5 leaves them."""
    out = tmp_path / "parity-gpu-out"
    run = kit.Kit(ROOT, out, [CALIBRATION.spec], "c" * 40, "d" * 64)
    run.facts.update(env_dir=kit.CU126_ENV, env_lock=f"{kit.CU126_ENV}/uv.lock")
    for name in kit.STEPS:
        run.steps.append(kit.Step(name, ok=smoke_ok or name != "smoke", seconds=1.0))
        if name == "smoke" and not smoke_ok:
            break
    (out / "results" / "Qwen__Qwen3-0.6B" / "verdicts" / "thinking").mkdir(parents=True)
    (out / "results" / "Qwen__Qwen3-0.6B" / "verdicts" / "thinking" / "t1.x__single_turn.json").write_text("{}")
    return out, run.finish()


def entries(zip_path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(zip_path) as zf:
        return {n: zf.read(n) for n in zf.namelist()}


def fake_repro(tmp_path: Path, body: str) -> Path:
    script = tmp_path / "fake_repro.py"
    script.write_text(body, encoding="utf-8")
    return script


PRINTS_A_DROP = (
    "import sys\n"
    "print('SFTConfig(pad_to_multiple_of=64)')\n"
    "print('batching: packing=False, padding_free=False (set explicitly)')\n"
    "print('before train_on_responses_only: collator DataCollatorForLanguageModeling(pad_to_multiple_of=64), batch width 64')\n"
    "print('after  train_on_responses_only: collator DataCollatorForSeq2Seq(pad_to_multiple_of=None), batch width 26')\n"
    "sys.exit(1)\n"
)


def test_the_repro_adds_its_own_folder_and_leaves_every_main_entry_byte_identical(tmp_path):
    out, zip_path = finished_main_run(tmp_path)
    before = entries(zip_path)
    code = kit.run_p0005(ROOT, out, zip_path, python=Path(sys.executable), repro=fake_repro(tmp_path, PRINTS_A_DROP))
    after = entries(zip_path)
    top = zip_path.stem
    assert code == 0
    assert {n: after[n] for n in before} == before  # the main results, untouched
    added = sorted(n.removeprefix(f"{top}/P-0005/") for n in after if n not in before)
    assert added == ["MANIFEST.json", "padded.log", "padding_free_unset.log", "run.json"]
    record = json.loads(after[f"{top}/P-0005/run.json"])
    assert record["environment"] == kit.CU126_ENV
    padded = record["variants"]["padded"]
    assert (padded["completed"], padded["exit_code"]) == (True, 1) and len(padded["lines"]) == 4
    assert "P-0005" not in json.loads(after[f"{top}/MANIFEST.json"])["files"]  # its own manifest, not the main one


def test_running_the_repro_again_replaces_its_folder_instead_of_adding_a_second(tmp_path):
    out, zip_path = finished_main_run(tmp_path)
    repro = fake_repro(tmp_path, PRINTS_A_DROP)
    kit.run_p0005(ROOT, out, zip_path, python=Path(sys.executable), repro=repro)
    first = sorted(entries(zip_path))
    kit.run_p0005(ROOT, out, zip_path, python=Path(sys.executable), repro=repro)
    assert sorted(entries(zip_path)) == first


def test_a_crashing_repro_is_recorded_and_the_main_results_still_zip(tmp_path):
    out, zip_path = finished_main_run(tmp_path)
    before = entries(zip_path)
    crash = fake_repro(tmp_path, "raise RuntimeError('CUDA error: no kernel image')\n")  # exits 1, like a drop
    code = kit.run_p0005(ROOT, out, zip_path, python=Path(sys.executable), repro=crash)
    after = entries(zip_path)
    record = json.loads(after[f"{zip_path.stem}/P-0005/run.json"])
    assert code == 1 and {n: after[n] for n in before} == before
    assert record["variants"]["padded"]["completed"] is False  # exit 1 alone is never read as "dropped"
    assert "no kernel image" in after[f"{zip_path.stem}/P-0005/padded.log"].decode()


def test_without_a_confirmed_gpu_route_the_repro_refuses_and_records_why(tmp_path):
    out, zip_path = finished_main_run(tmp_path, smoke_ok=False)
    before = entries(zip_path)
    code = kit.run_p0005(ROOT, out, zip_path, python=Path(sys.executable), repro=fake_repro(tmp_path, PRINTS_A_DROP))
    after = entries(zip_path)
    record = json.loads(after[f"{zip_path.stem}/P-0005/run.json"])
    assert code == 1 and {n: after[n] for n in before} == before
    assert "smoke" in record["error"] and record["variants"] == {}


def test_if_the_zip_rewrite_fails_the_original_zip_is_left_exactly_as_it_was(tmp_path, monkeypatch):
    out, zip_path = finished_main_run(tmp_path)
    original = zip_path.read_bytes()

    def broken_write(self, *args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(zipfile.ZipFile, "write", broken_write)
    code = kit.run_p0005(ROOT, out, zip_path, python=Path(sys.executable), repro=fake_repro(tmp_path, PRINTS_A_DROP))
    assert code == 1 and zip_path.read_bytes() == original
    assert not zip_path.with_name(zip_path.name + ".tmp").exists()


def test_no_results_zip_means_nothing_runs(tmp_path):
    assert kit.run_p0005(ROOT, tmp_path / "out", tmp_path / "missing.zip") == 2
    assert not (tmp_path / "out").exists()


def test_the_repro_outcome_needs_its_result_line():
    assert kit.p0005_outcome(1, "Traceback ...\nRuntimeError: boom")["completed"] is False
    done = kit.p0005_outcome(0, "noise\nafter  train_on_responses_only: collator X(pad_to_multiple_of=64), batch width 64")
    assert done["completed"] is True and done["lines"] == [done["lines"][0]]


def test_the_optional_cell_sits_between_the_run_and_the_download_and_needs_the_main_zip():
    nb = json.loads((KIT_DIR / "parity_gpu_run.ipynb").read_text(encoding="utf-8"))
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    run, optional, download = code[3], code[4], code[5]
    assert "colab_kit.py" in run and "--extra" not in run
    assert optional.startswith("# Cell 6 (optional)") and '"--extra", "p0005"' in optional and "RESULT_ZIP" in optional
    assert download.startswith("# Cell 7: download") and "files.download(RESULT_ZIP)" in download


def test_the_kit_points_at_the_committed_repro():
    assert (ROOT / kit.P0005_REPRO).is_file()
    assert "--padding-free-unset" in (ROOT / kit.P0005_REPRO).read_text(encoding="utf-8")
