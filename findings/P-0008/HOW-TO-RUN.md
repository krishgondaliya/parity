# How to run the P-0008 repro before posting

The script needs transformers, jinja2 and the standard library, plus TRL for one optional line. It runs on CPU in a
few seconds, downloads Inkling-2-Mini's tokenizer and template files only (pinned revision, ungated, no token needed),
and exits **1** when the problem reproduces.

These commands were run exactly as written (with `WORK` set to a scratch folder) on 2026-11-27.

## 1. Set up a fresh environment with the pinned versions

```bash
WORK=~/p0008                      # anywhere outside the parity repo
mkdir -p "$WORK"
cp ~/parity/findings/P-0008/repro.py ~/parity/findings/P-0008/requirements.txt "$WORK/"
cd "$WORK"
uv venv --python 3.12 .venv
VIRTUAL_ENV="$WORK/.venv" uv pip install -r requirements.txt
```

`requirements.txt` is the exact resolution of `transformers==5.24.0` and `trl==1.18.2` against Parity's TRL `vlm`
lockfile (`findings/P-0008/uv.lock`): transformers 5.24.0, tokenizers 0.23.3, jinja2 3.1.6, trl 1.18.2, torch 2.15.0.
Torch comes as the default Linux CUDA wheel, so the first install downloads a few GB; the script itself runs on CPU.

**Smaller alternative** (what a model-repo reader is likely to use): only the template part, no TRL and no torch.

```bash
WORK=~/p0008-min                  # anywhere outside the parity repo
mkdir -p "$WORK"
cp ~/parity/findings/P-0008/repro.py "$WORK/"
cd "$WORK"
uv venv --python 3.12 .venv
VIRTUAL_ENV="$WORK/.venv" uv pip install transformers==5.24.0 jinja2==3.1.6
```

jinja2 must be pinned too: with transformers alone, `apply_chat_template` raises
`ImportError: apply_chat_template requires jinja2`, which also exits 1. So check the output, not just the exit code.

## 2. Run it

Run from `$WORK`, outside the repo, so nothing from Parity can be imported.

```bash
cd "$WORK"
.venv/bin/python repro.py; echo "exit code: $?"
```

Expected output with the pinned environment, then `exit code: 1`:

```
expected (tml-renderers 0.5.0): no empty text message for any form
content "": 0 empty text message(s)
content null: 0 empty text message(s)
content [{"type": "text", "text": ""}]: 1 empty text message(s)
TRL turns content "" into [{'type': 'text', 'text': ''}]: 1 empty text message(s)
```

With the smaller alternative, the same first four lines and `exit code: 1`: without TRL, the last line is skipped.
Library warnings may be printed above it on stderr; they don't matter.

## 3. No newer version to check

The model's main (`c9d0283`, 2026-11-24) has the same `chat_template.jinja`, `tokenizer_config.json` and
`tokenizer.json` as the pinned revision (`7a67ae2`), byte for byte, and Inkling-2-Small and Inkling-2-Small-MXFP8 ship
the same `chat_template.jinja`. TRL's `prepare_multimodal_messages` is unchanged on its main (`c2d5ff4`, 2026-11-26).

## Then post

On https://huggingface.co/thinkingmachines/Inkling-2-Mini/discussions, click **New discussion**. Paste the **Title**
from `findings/P-0008/issue.md`, then everything under **Body** as the discussion's text (it is Markdown). Nothing
to attach: the repro is inline. Afterwards, put the discussion URL in `findings/P-0008/record.md` under "Upstream".

## What was checked

- 2026-11-27: the commands above, exactly as written (with `WORK` set to a scratch folder), both environments. The
  pinned one printed the five lines and exited 1; the smaller one printed the first four lines and exited 1. Earlier
  the same day: 3 runs in the pinned environment, identical output, exit 1 each time.
- 2026-11-26: 3 runs in the pinned environment (before the review edits that pinned jinja2 in the script's comment and
  removed an unneeded `copy`): exit 1 every time, the same output.
- The expected line comes from Thinking Machines' tml-renderers 0.5.0, run in a sandbox on this exact conversation:
  `docs/evidence/2026-11-26-p0008-renderer/repro_results.json`.
