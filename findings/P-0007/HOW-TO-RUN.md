# How to run the P-0007 repro before posting

The script needs only Axolotl, transformers and the standard library. It runs on CPU in seconds, downloads one ungated
tokenizer (EXAONE-4.0-1.2B, pinned; tokenizer files only, no weights, no token needed), and exits **1** when the
problem reproduces.

These commands were run exactly as written (with `WORK=~/p0007`) on 2026-11-26.

## 1. Set up a fresh environment with the pinned versions

```bash
WORK=~/p0007                      # anywhere outside the parity repo
mkdir -p "$WORK"
cp ~/parity/findings/P-0007/repro.py ~/parity/findings/P-0007/requirements.txt "$WORK/"
cd "$WORK"
uv venv --python 3.12 .venv-0.21.1
VIRTUAL_ENV="$WORK/.venv-0.21.1" uv pip install -r requirements.txt
```

`requirements.txt` is the exact resolution of `uv pip install axolotl==0.21.1` that Parity's lockfile pins (axolotl
0.21.1, transformers 5.20.1, torch 2.14.1), the same file as P-0002's. Torch comes as the default Linux CUDA wheel,
so the first install downloads a few GB; the script itself runs on CPU.

## 2. Turn Axolotl's telemetry off

Axolotl's telemetry is opt-out. The script also sets these itself, before its Axolotl import.

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

Expected output (after Axolotl's and torch's log lines on stderr), then `exit code: 1`:

```
mappings: {'role': 'role', 'content': 'content'} | analyzer found: ['content', 'role']
--- the template's render ---
[|user|]
Weather in Paris?[|endofturn|]
[|assistant|]
<think>

</think>

<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>[|endofturn|]
[|tool|]
<tool_result>{"result": "18C"}</tool_result>[|endofturn|]
[|assistant|]
<think>
The tool says 18C.
</think>

It's 18C.[|endofturn|]

--- Axolotl's row ---
[|user|]
Weather in Paris?[|endofturn|]
[|assistant|]
[|endofturn|]
[|tool|]
<tool_result>{"result": "18C"}</tool_result>[|endofturn|]
[|assistant|]
<think>

</think>

It's 18C.[|endofturn|]
```

## 4. Run it on Axolotl's main branch

```bash
cd "$WORK"
uv venv --python 3.12 .venv-main
VIRTUAL_ENV="$WORK/.venv-main" uv pip install "axolotl @ git+https://github.com/axolotl-ai-cloud/axolotl@de8549a4406327ab5311f7187a8afaa232645487"
.venv-main/bin/python repro.py; echo "exit code: $?"
```

On main at `de8549a` (axolotl 0.21.2.dev0, transformers 5.22.0, torch 2.15.0), the same output, then
`exit code: 1`.

## What was checked, 2026-11-26

- 3 runs on 0.21.1 and 3 on main: exit 1 every time, with the output above, identical across the six runs. (An earlier version printed True/False lines for the two marks; the issue-reviewer asked for the texts themselves, and this version was re-run 3+3.)
- `findings/P-0007/path_check.py` (in Parity's Axolotl environment): on this conversation, the full path
  (`load_cfg` + `load_datasets`, what `axolotl preprocess` runs, with EXAONE-4.0's config and tokenizer files as
  `base_model`) gives the same input ids and labels as the script's direct strategy call. With `reasoning_content` and
  `tool_calls` listed in `message_property_mappings`, the row equals the template's render.
