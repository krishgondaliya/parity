# How to run the P-0005 repro before posting

**It runs on CPU. No GPU is needed.** The script needs only Unsloth (with the TRL, transformers and torch it pins), `datasets`, `huggingface_hub` and the standard library. It downloads Qwen3-0.6B's tokenizer and config files only (ungated, no token needed); the model is a tiny random one built locally. It runs in seconds once those files are cached, and exits **1** when the problem reproduces.

These commands were run exactly as written (with `WORK` set to a scratch folder) on 2026-11-25, on a machine with no GPU.

## 1. Set up a fresh environment with the pinned versions

```bash
WORK=~/p0005                      # anywhere outside the parity repo
mkdir -p "$WORK"
cp ~/parity/findings/P-0005/repro.py ~/parity/findings/P-0005/requirements.txt "$WORK/"
cd "$WORK"
uv venv --python 3.12 .venv-2026.11.10
VIRTUAL_ENV="$WORK/.venv-2026.11.10" uv pip install -r requirements.txt
```

`requirements.txt` holds the exact versions of Parity's Unsloth lockfile (`envs/unsloth-2026.11.10/uv.lock`, also copied here as `uv.lock`), without Parity itself: unsloth 2026.11.10, unsloth-zoo 2026.11.10, trl 1.15.0, transformers 5.17.0, torch 2.14.1. Torch comes as the default Linux CUDA wheel, so the first install downloads a few GB; the script itself runs on CPU.

## 2. Turn telemetry off

```bash
export UNSLOTH_DISABLE_STATISTICS=1      # Unsloth's usage statistics (being set at all turns them off)
export HF_HUB_DISABLE_TELEMETRY=1        # Hugging Face Hub telemetry
export DO_NOT_TRACK=1                    # the same, by the shared convention huggingface_hub also reads
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1   # not telemetry: never send a stored Hugging Face token (the repo is ungated)
```

Parity's own Unsloth adapter also sets `UNSLOTH_DISABLE_STATISTICS=1` before importing Unsloth. Parity's GPU kit sets it as well, along with `HF_HUB_DISABLE_TELEMETRY` and `HF_HUB_DISABLE_IMPLICIT_TOKEN`.

## 3. Run it on unsloth 2026.11.10

Run from `$WORK`, outside the repo, so nothing from Parity can be imported.

```bash
cd "$WORK"
.venv-2026.11.10/bin/python repro.py; echo "exit code: $?"
```

The script prints these three lines, then `exit code: 1`:

```
before train_on_responses_only: DataCollatorForLanguageModeling, batch width 64
after: DataCollatorForSeq2Seq(pad_to_multiple_of=None)
expected batch width: a multiple of 64; actual: 26
```

Around them, Unsloth prints its own lines on stdout. They start with "Unsloth 2026.11.10: no GPU found, training on CPU", and include "Unsloth: Dropping columns the model is not fed: ['text']". Their thread counts and memory figures depend on the machine. Torch warnings may also appear on stderr. None of them matter.

On a machine with an NVIDIA GPU, Unsloth takes its GPU route instead. There, a longer variant of this script with the same calls gave 64, then 25 (Tesla T4, `docs/evidence/2026-11-25-p0005-gpu/`); this exact script wasn't run on GPU.

## 4. Optional: check it on the newest release and on unsloth-zoo's main branch

```bash
cd "$WORK"
uv venv --python 3.12 .venv-2026.11.12
VIRTUAL_ENV="$WORK/.venv-2026.11.12" uv pip install "unsloth==2026.11.12" "unsloth-zoo==2026.11.12"
.venv-2026.11.12/bin/python repro.py; echo "exit code: $?"

uv venv --python 3.12 .venv-main
VIRTUAL_ENV="$WORK/.venv-main" uv pip install "unsloth==2026.11.12" "unsloth-zoo @ git+https://github.com/unslothai/unsloth-zoo@cad81ef2dc557b1060896d3b294ef6b9a7f9c1ea"
.venv-main/bin/python repro.py; echo "exit code: $?"
```

Both give the same three lines and `exit code: 1`. Both resolve to trl 1.15.0, transformers 5.17.0 and torch 2.14.1. At `cad81ef2`, the code that swaps the collator is byte-identical to 2026.11.10's (`unsloth_zoo/dataset_utils.py` L2547-L2583). unsloth-zoo's main has since moved one commit, which changes only a test file; the issue links `cad81ef2`.

## Then post

1. Open a new issue at https://github.com/unslothai/unsloth-zoo/issues with the "Bug / Issue" template.
2. Paste `findings/P-0005/issue.md` into it: keep the template's six questions, and replace its default "[Bug]" title with the draft's title. The notes at the top of the draft say the same.
3. Don't attach anything; the draft says the pinned requirements are available on request.
4. Afterwards, put the issue URL in `findings/P-0005/record.md` under "Upstream".
