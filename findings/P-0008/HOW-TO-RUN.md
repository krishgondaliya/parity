# How to run the P-0008 repro before posting

The script needs transformers and the standard library, plus TRL for one optional line. It runs on CPU in seconds,
downloads Inkling-2-Mini's tokenizer and template files only (pinned revision, ungated, no token needed), and exits
**1** when the problem reproduces.

These commands were run exactly as written (with `WORK=~/p0008`) on 2026-11-26.

## 1. Set up a fresh environment with the pinned versions

```bash
WORK=~/p0008                      # anywhere outside the parity repo
mkdir -p "$WORK"
cp ~/parity/findings/P-0008/repro.py ~/parity/findings/P-0008/requirements.txt "$WORK/"
cd "$WORK"
uv venv --python 3.12 .venv
VIRTUAL_ENV="$WORK/.venv" uv pip install -r requirements.txt
```

`requirements.txt` resolves `transformers==5.24.0` and `trl==1.18.2` against Parity's TRL `vlm` lockfile
(`findings/P-0008/uv.lock`), so every package is at the version Parity ran (torch 2.15.0). TRL is only needed for
the optional last line; `pip install transformers==5.24.0 jinja2==3.1.6` alone runs the template part. transformers
alone doesn't: `apply_chat_template` raises `ImportError` without jinja2, which also exits 1, so check the output.

## 2. Run it

```bash
cd "$WORK"
.venv/bin/python repro.py; echo "exit code: $?"
```

Expected output, then `exit code: 1`:

```
expected (tml-renderers 0.5.0): no empty text message for any form
content "": 0 empty text message(s)
content null: 0 empty text message(s)
content [{"type": "text", "text": ""}]: 1 empty text message(s)
TRL turns content "" into [{'type': 'text', 'text': ''}]: 1 empty text message(s)
```

Without TRL installed, the last line is skipped and the exit code is still 1.

## What was checked, 2026-11-27 (after the issue-reviewer's edits: jinja2 pinned in the comment, `copy` removed)

- 3 runs in the fresh pinned environment: exit 1 every time, the five lines above, identical across the runs.
- A venv with only `transformers==5.24.0` and `jinja2==3.1.6`: the first four lines, exit 1. With transformers alone:
  `ImportError: apply_chat_template requires jinja2`.

## What was checked, 2026-11-26

- 3 runs in the fresh environment (transformers 5.24.0, trl 1.18.2, torch 2.15.0): exit 1 every time, with the output above. One more run with TRL made unimportable: the first four lines, exit 1.
- No fixed version to test: `chat_template.jinja`, `tokenizer_config.json` and `tokenizer.json` are byte-identical between the pinned revision and the model's main (`c9d0283`, 2026-11-24), and `prepare_multimodal_messages` is identical on TRL's main (`c2d5ff4`, 2026-11-26).
- The expected line comes from Thinking Machines' tml-renderers 0.5.0, run sandboxed: `docs/evidence/2026-11-26-p0008-renderer/`.
