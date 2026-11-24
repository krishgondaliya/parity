# How to run the P-0002 repro before posting

The script needs only Axolotl, transformers and the standard library. It runs on CPU in seconds, downloads tokenizer and config files only (three ungated repos, no token needed), and exits **1** when the problem reproduces.

These commands were run exactly as written (with `WORK` set to a scratch folder) on 2026-11-24.

## 1. Set up a fresh environment with the pinned versions

```bash
WORK=~/p0002                      # anywhere outside the parity repo
mkdir -p "$WORK"
cp ~/parity/findings/P-0002/repro.py ~/parity/findings/P-0002/requirements.txt "$WORK/"
cd "$WORK"
uv venv --python 3.12 .venv-0.21.1
VIRTUAL_ENV="$WORK/.venv-0.21.1" uv pip install -r requirements.txt
```

`requirements.txt` is the exact resolution of `uv pip install axolotl==0.21.1` that Parity's lockfile is pinned to: axolotl 0.21.1, transformers 5.20.1, torch 2.14.1. Torch comes as the default Linux CUDA wheel, so the first install downloads a few GB; the script itself runs on CPU.

## 2. Turn Axolotl's telemetry off

Axolotl's telemetry is opt-out. These are the same two variables Parity's adapter sets before importing Axolotl. The script also sets them itself, before its Axolotl import.

```bash
export AXOLOTL_DO_NOT_TRACK=1
export DO_NOT_TRACK=1
```

## 3. Run it on axolotl 0.21.1

Run from `$WORK`, outside the repo, so nothing from Parity can be imported.

```bash
cd "$WORK"
.venv-0.21.1/bin/python repro.py; echo "exit code: $?"
```

Expected output, then `exit code: 1`:

```
HuggingFaceTB/SmolLM2-135M-Instruct: <|im_end|> at [19, 27, 35, 47, 53], labeled [False, False, True, False, False]; empty turn's <|im_end|> labeled: False
unsloth/Llama-3.2-1B-Instruct: <|eot_id|> at [25, 33, 40, 52, 57], labeled [False, False, True, False, False]; empty turn's <|eot_id|> labeled: False
mistralai/Mistral-7B-Instruct-v0.3: </s> at [8, 19], labeled [True, False]; empty turn's </s> labeled: False
```

Library warnings may be printed above it on stderr; they don't matter.

## 4. Optional: check it on Axolotl's main branch

```bash
cd "$WORK"
uv venv --python 3.12 .venv-main
VIRTUAL_ENV="$WORK/.venv-main" uv pip install "axolotl @ git+https://github.com/axolotl-ai-cloud/axolotl@de8549a4406327ab5311f7187a8afaa232645487"
.venv-main/bin/python repro.py; echo "exit code: $?"
```

Same output and `exit code: 1` (main at `de8549a` resolves to axolotl 0.21.2.dev0 with transformers 5.22.0 and torch 2.15.0).

## Then post

Paste `findings/P-0002/issue.md` into Axolotl's Bug report form, field by field, and attach `requirements.txt`. Afterwards, put the issue URL in `findings/P-0002/record.md` under "Upstream".
