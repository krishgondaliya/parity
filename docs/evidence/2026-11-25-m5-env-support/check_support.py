"""M5 prep, item 3: can each framework's pinned environment handle each Phase 0 target? No data path.

Run from the repo root, once per environment, then once more to summarize:
    envs/trl-1.18.2/.venv/bin/python        docs/evidence/2026-11-25-m5-env-support/check_support.py trl
    envs/axolotl-0.21.1/.venv/bin/python    docs/evidence/2026-11-25-m5-env-support/check_support.py axolotl
    envs/unsloth-2026.11.10/.venv/bin/python docs/evidence/2026-11-25-m5-env-support/check_support.py unsloth
    uv run python                            docs/evidence/2026-11-25-m5-env-support/check_support.py none
    uv run python                            docs/evidence/2026-11-25-m5-env-support/check_support.py --summarize

What it does in each environment (the human's decision, 2026-11-25: it may import Unsloth before the M5 predictions,
because it never touches the data path):
1. Import the framework first, the way its docs do (Unsloth registers some architectures at import). No dataset,
   no trainer, no collator, no template handling by the framework.
2. Per target, from the release's tokenizer and config files only (rule 6):
   - load the tokenizer with trust_remote_code=False;
   - render the 8 fixtures in each documented mode and hash the token ids, to compare across environments;
   - resolve the config with AutoConfig, and record the exact exception if it fails;
   - build the release's architecture on the `meta` device (module structure only, no memory, no weights): as a
     text causal LM (from the config, then from its text_config) and as an image-text-to-text model. This answers
     "can this environment build the architecture" without depending on any shrinking code of ours;
   - Axolotl only: its own model-config loader (`axolotl.loaders.utils.load_model_config`), the first thing
     `axolotl preprocess` does, for the exact error a user would see.
3. The same-family stand-in candidates (D1): is the model type registered, and does its default config build on meta?

Writes <framework>.json next to this file; `--summarize` writes summary.md from all four.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ENVS = {
    "trl": "envs/trl-1.18.2/uv.lock",
    "trl-vlm": "envs/trl-1.18.2/vlm/uv.lock",  # TRL's own vlm extra (the human's decision, 2026-11-26)
    "axolotl": "envs/axolotl-0.21.1/uv.lock",
    "unsloth": "envs/unsloth-2026.11.10/uv.lock",
    "none": "uv.lock",
}
# D1 stand-in candidates, fixed before any run: registered architectures of the same family, per path (a text
# causal LM, or an image-text-to-text model when a framework's documented path is the vision one). In order of
# preference: the first one an environment registers is its stand-in. DeepSeek: no V4-line vision class exists in
# any environment (DeepSeek-VL is a different model line), and the release's own config declares a causal LM.
STAND_INS = {
    "deepseek_v41": {"text": ("deepseek_v4",), "vision": ()},
    "hy_v4_flash": {"text": ("hy_v4",), "vision": ("hy_v4_vl", "hunyuan_vl")},
    "inkling2_loop": {"text": ("inkling_text",), "vision": ("inkling_mm_model",)},
}


def import_framework(name: str) -> dict:
    """Import the framework the way its docs and our adapters do, before transformers is touched."""
    if name == "unsloth":
        os.environ["UNSLOTH_DISABLE_STATISTICS"] = "1"  # Unsloth's documented opt-out
        import unsloth

        return {"unsloth": unsloth.__version__, "backend": getattr(unsloth, "DEVICE_TYPE", "unknown")}
    if name == "axolotl":
        os.environ["AXOLOTL_DO_NOT_TRACK"] = "1"
        os.environ["DO_NOT_TRACK"] = "1"
        import axolotl

        return {"axolotl": axolotl.__version__}
    if name in ("trl", "trl-vlm"):
        import trl

        return {"trl": trl.__version__}
    return {}


def error_record(exc: BaseException) -> dict:
    message = str(exc).replace(str(Path.home()), "~")[:2000]  # no home-directory paths in committed evidence
    return {"type": f"{type(exc).__module__}.{type(exc).__qualname__}", "message": message}


def ids_hash(ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16]


def meta_build(auto_class, config) -> dict:
    try:
        import torch
    except ImportError:  # the core environment has no torch: renders only
        return {"ok": False, "error": {"type": "skipped", "message": "torch is not installed in this environment"}}
    try:
        with torch.device("meta"):
            model = auto_class.from_config(config)
        return {"ok": True, "class": type(model).__name__, "params": sum(p.numel() for p in model.parameters())}
    except Exception as exc:  # noqa: BLE001 - recorded exactly, never hidden
        return {"ok": False, "error": error_record(exc)}


def check_target(target, framework: str) -> dict:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

    from parity.fixtures import FIXTURE_IDS, load_fixture
    from parity.references.template import model_chat_template, template_token_ids
    from parity.release import fetch_release_files
    from parity.serving import MODE_SPECS

    local = fetch_release_files(target)
    out: dict = {"spec": target.spec}

    try:
        tokenizer = AutoTokenizer.from_pretrained(local, trust_remote_code=False)
        out["tokenizer"] = {
            "ok": True,
            "class": type(tokenizer).__name__,
            "len": len(tokenizer),
            "bos": tokenizer.bos_token,
            "eos": tokenizer.eos_token,
            "pad": tokenizer.pad_token,
        }
        template = model_chat_template(tokenizer)
        renders = {}
        for mode, kwargs in MODE_SPECS[target.repo_id].kwargs.items():
            if kwargs is None:
                continue
            for fid in FIXTURE_IDS:
                fixture = load_fixture(fid)
                try:
                    renders[f"{mode}/{fid}"] = ids_hash(template_token_ids(tokenizer, fixture, template, **kwargs))
                except Exception as exc:  # noqa: BLE001
                    renders[f"{mode}/{fid}"] = "error: " + error_record(exc)["message"][:200]
        out["renders"] = renders
    except Exception as exc:  # noqa: BLE001
        out["tokenizer"] = {"ok": False, "error": error_record(exc)}

    try:
        config = AutoConfig.from_pretrained(local)
        text = getattr(config, "text_config", None)
        out["config"] = {
            "ok": True,
            "class": type(config).__name__,
            "model_type": config.model_type,
            "text_model_type": getattr(text, "model_type", None),
        }
        out["build"] = {
            "causal_lm(config)": meta_build(AutoModelForCausalLM, config),
            "image_text_to_text(config)": meta_build(AutoModelForImageTextToText, config),
        }
        if text is not None:
            out["build"]["causal_lm(text_config)"] = meta_build(AutoModelForCausalLM, text)
    except Exception as exc:  # noqa: BLE001
        out["config"] = {"ok": False, "error": error_record(exc)}

    # The vision path's processor (D1): what AutoProcessor builds from the release's own files, or the exact error.
    if (Path(local) / "processor_config.json").exists():
        import transformers
        from transformers import AutoProcessor

        declared = json.loads((Path(local) / "processor_config.json").read_text(encoding="utf-8"))["processor_class"]
        record: dict = {"declared": declared}
        try:
            getattr(transformers, declared)
            record["declared_available"] = True
        except AttributeError as exc:  # the exact error a framework resolving the declared class gets
            record["declared_available"] = False
            record["declared_error"] = error_record(exc)
        try:  # what AutoProcessor gives instead: with no such class it falls back to the tokenizer, silently
            got = type(AutoProcessor.from_pretrained(local)).__name__
            record["auto_processor"] = got
            record["ok"] = got == declared
        except Exception as exc:  # noqa: BLE001
            record["ok"] = False
            record["auto_processor_error"] = error_record(exc)
        out["processor"] = record
    else:
        out["processor"] = {"ok": None, "note": "the release ships no processor_config.json (no vision path)"}

    if framework == "axolotl":
        from axolotl.loaders.utils import load_model_config
        from axolotl.utils.dict import DictDefault

        try:
            loaded = load_model_config(DictDefault({"base_model": str(local)}))
            out["axolotl_load_model_config"] = {"ok": True, "model_type": getattr(loaded, "model_type", None)}
        except Exception as exc:  # noqa: BLE001
            out["axolotl_load_model_config"] = {"ok": False, "error": error_record(exc)}
    return out


def check_stand_ins() -> dict:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    autos = {"text": AutoModelForCausalLM, "vision": AutoModelForImageTextToText}
    out: dict = {}
    for release_type, paths in STAND_INS.items():
        for path, candidates in paths.items():
            for stand_in in candidates:
                entry: dict = {"for": release_type, "path": path, "registered": stand_in in CONFIG_MAPPING}
                if entry["registered"]:
                    entry["build"] = meta_build(autos[path], AutoConfig.for_model(stand_in))
                out[f"{release_type}/{path}/{stand_in}"] = entry
    return out


def run(framework: str) -> None:
    versions = import_framework(framework)
    import tokenizers
    import transformers

    try:
        import torch

        torch_version = torch.__version__
    except ImportError:
        torch_version = "not installed"

    from parity.targets import PHASE0_TARGETS

    lock = ENVS[framework]
    record = {
        "framework": framework,
        "python": sys.executable.replace(str(REPO) + "/", ""),
        "env_lock": lock,
        "env_lock_sha256": hashlib.sha256((REPO / lock).read_bytes()).hexdigest(),
        "versions": {
            **versions,
            "transformers": transformers.__version__,
            "tokenizers": tokenizers.__version__,
            "torch": torch_version,
        },
        "targets": {},
    }
    for target in PHASE0_TARGETS:
        try:
            record["targets"][target.repo_id] = check_target(target, framework)
        except Exception as exc:  # noqa: BLE001 - a crash in the check itself is ours, recorded as such
            record["targets"][target.repo_id] = {"harness_error": error_record(exc), "trace": traceback.format_exc()}
    record["stand_ins"] = check_stand_ins()
    (HERE / f"{framework}.json").write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for repo_id, t in record["targets"].items():
        print(repo_id, "tokenizer:", t.get("tokenizer", {}).get("ok"), "config:", t.get("config", {}).get("ok"))


# The Day-0 support table (D1): per framework x target, the environment it runs in, the path a text-only chat
# fine-tune takes (read from each framework's own code; docs/DECISIONS.md, 2026-11-26, and each adapter's PATHS),
# and what follows. TRL x Inkling runs in the vlm variant (the human's decision, 2026-11-26).
PAIRS = {
    ("trl", "thinkingmachines/Inkling-2-Mini"): ("trl-vlm", "vision"),
    ("trl", "tencent/Hy4-flash"): ("trl", "vision"),
    ("trl", "deepseek-ai/DeepSeek-V4.1-Flash-1017"): ("trl", "text"),
    ("axolotl", "thinkingmachines/Inkling-2-Mini"): ("axolotl", "vision"),
    ("axolotl", "tencent/Hy4-flash"): ("axolotl", "vision"),
    ("axolotl", "deepseek-ai/DeepSeek-V4.1-Flash-1017"): ("axolotl", "text"),
    ("unsloth", "thinkingmachines/Inkling-2-Mini"): ("unsloth", "vision"),
    ("unsloth", "tencent/Hy4-flash"): ("unsloth", "vision"),
    ("unsloth", "deepseek-ai/DeepSeek-V4.1-Flash-1017"): ("unsloth", "text"),
}
NOTES = {
    ("trl", "deepseek-ai/DeepSeek-V4.1-Flash-1017"): (
        "TRL's refusal is predicted from its source (trl.no_training_template: no generation markers, no stored "
        "training template), so its cells are expected not_supported"
    ),
}


def summarize() -> None:
    records = {fw: json.loads((HERE / f"{fw}.json").read_text(encoding="utf-8")) for fw in ENVS}
    frameworks = [fw for fw in ENVS if fw != "none"]
    targets = list(records["none"]["targets"])

    def label(fw: str) -> str:
        v = records[fw]["versions"]
        base = fw.removesuffix("-vlm")
        name = f"{base} {v[base]}" + (" (vlm extra)" if fw.endswith("-vlm") else "") if base in v else "core (no framework)"
        return f"{name}, transformers {v['transformers']}"

    def built(t: dict) -> str:
        if not t.get("config", {}).get("ok"):
            return "no"
        parts = [f"{k}: {v['class']}" for k, v in t.get("build", {}).items() if v.get("ok")]
        return "yes (" + "; ".join(parts) + ")" if parts else "config only, no class builds"

    out = [
        "# M5 support check: can each pinned environment handle each target?",
        "",
        "Generated by `check_support.py`. Tokenizer and config files only (rule 6); architectures built on the `meta` "
        "device (structure, no memory, no weights); the framework imported first, never its data path.",
        "",
        "## Does the environment build the release's architecture?",
        "",
        "| target | " + " | ".join(label(fw) for fw in frameworks) + " |",
        "| --- |" + " --- |" * len(frameworks),
    ]
    for repo_id in targets:
        out.append(f"| {repo_id} | " + " | ".join(built(records[fw]["targets"][repo_id]) for fw in frameworks) + " |")
    def first_line(err: dict) -> str:
        return f"`{err['type']}`: {err['message'].splitlines()[0]}"

    out += ["", "## Day-0 support, per pair (D1: the report card's \"loads the release's architecture\" field)", "",
            "| framework | target | environment | path | release's architecture | release's processor | M5 |",
            "| --- | --- | --- | --- | --- | --- | --- |"]
    for (fw, repo_id), (env, path) in PAIRS.items():
        t = records[env]["targets"][repo_id]
        arch = "builds" if t.get("config", {}).get("ok") else "no: " + first_line(t["config"]["error"])[:90] + "..."
        pr = t.get("processor", {})
        if path == "text":
            proc, judged = "n/a (text path)", None
        elif pr.get("ok"):
            proc, judged = f"{pr['auto_processor']} builds", None
        else:
            err = pr.get("declared_error") or pr.get("auto_processor_error")
            proc = f"can't build {pr['declared']}: {first_line(err)[:120]}"
            judged = "not judged: can't build the release's processor"
        if judged is None:
            judged = "judged on the release's architecture" if t.get("config", {}).get("ok") else "judged on a stand-in"
        note = NOTES.get((fw, repo_id))
        if note:
            judged += f"; {note}"
        out.append(f"| {fw} | {repo_id} | `{ENVS[env]}` | {path} | {arch} | {proc} | {judged} |".replace("\n", " "))
    out += ["", "## Exact errors (first line; the full text is in `<framework>.json`)", ""]
    for repo_id in targets:
        for fw in frameworks:
            t = records[fw]["targets"][repo_id]
            if "harness_error" in t:
                out.append(f"- **{repo_id}, {fw}: HARNESS ERROR (ours)** {first_line(t['harness_error'])}")
                continue
            c = t["config"]
            if not c["ok"]:
                out.append(f"- **{repo_id}, {fw}, AutoConfig:** {first_line(c['error'])}")
            for k, b in t.get("build", {}).items():
                if not b["ok"]:
                    out.append(f"- {repo_id}, {fw}, {k}: {first_line(b['error'])}")
            a = t.get("axolotl_load_model_config")
            if a and not a["ok"]:
                out.append(f"- **{repo_id}, axolotl, load_model_config:** {first_line(a['error'])}")
    out += ["", "## The vision path's processor (what AutoProcessor builds from the release's files)", "",
            "| target | " + " | ".join(label(fw) for fw in frameworks) + " |", "| --- |" + " --- |" * len(frameworks)]
    for repo_id in targets:
        cells = []
        for fw in frameworks:
            pr = records[fw]["targets"][repo_id].get("processor", {})
            if pr.get("ok") is None:
                cells.append("n/a")
            elif pr["ok"]:
                cells.append(f"{pr['auto_processor']} (the declared class)")
            elif not pr["declared_available"]:
                got = pr.get("auto_processor") or first_line(pr["auto_processor_error"])
                cells.append(f"can't build {pr['declared']}: {first_line(pr['declared_error'])}; AutoProcessor gives {got}")
            else:
                cells.append(f"can't build {pr['declared']}: AutoProcessor raises {first_line(pr['auto_processor_error'])}")
        out.append(f"| {repo_id} | " + " | ".join(c.replace("|", "/") for c in cells) + " |")
    out += ["", "## Tokenizers", "", "| target | " + " | ".join(label(fw) for fw in ENVS) + " |", "| --- |" + " --- |" * len(ENVS)]
    for repo_id in targets:
        cells = []
        for fw in ENVS:
            tok = records[fw]["targets"][repo_id].get("tokenizer", {})
            cells.append(f"{tok['class']} ({tok['len']})" if tok.get("ok") else "FAILS")
        out.append(f"| {repo_id} | " + " | ".join(cells) + " |")
    out += ["", "## Fixture renders: same token ids in every environment?", ""]
    for repo_id in targets:
        base = records["none"]["targets"][repo_id].get("renders", {})
        for fw in frameworks:
            other = records[fw]["targets"][repo_id].get("renders", {})
            diff = sorted(k for k in base if other.get(k) != base[k])
            verdict = "identical" if not diff and other else ("differ: " + ", ".join(diff) if other else "not rendered")
            out.append(f"- {repo_id}, {fw}: {len(base)} renders, {verdict}")
    out += [
        "",
        "## Stand-in candidates (D1), default config of each, built on meta",
        "",
        "| release type | path | stand-in | " + " | ".join(frameworks) + " |",
        "| --- | --- | --- |" + " --- |" * len(frameworks),
    ]
    for key, entry in records["none"]["stand_ins"].items():
        cells = []
        for fw in frameworks:
            e = records[fw]["stand_ins"][key]
            if not e["registered"]:
                cells.append("not registered")
            else:
                b = e["build"]
                cells.append(b["class"] if b["ok"] else f"fails: {first_line(b['error'])[:160]}")
        out.append(f"| {entry['for']} | {entry['path']} | {key.rsplit('/', 1)[1]} | " + " | ".join(cells) + " |")
    out += ["", "## Environments", ""]
    for fw, r in records.items():
        out.append(f"- {fw}: `{r['env_lock']}` (sha256 {r['env_lock_sha256'][:16]}), {json.dumps(r['versions'])}")
    out.append("")
    (HERE / "summary.md").write_text("\n".join(out), encoding="utf-8")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg == "--summarize":
        summarize()
    elif arg in ENVS:
        run(arg)
    else:
        raise SystemExit(f"usage: check_support.py {{{'|'.join(ENVS)}}} | --summarize")
