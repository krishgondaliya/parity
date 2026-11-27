# GPU run: Unsloth on a free Colab GPU

This runs Parity's six Tier 1 checks on Unsloth's real GPU route, in both serving modes, and gives you one zip to bring back. An optional cell also runs the P-0005 repro on the same GPU. Unsloth's own notebooks target Colab's free T4 GPU, so that's where this runs.

- **Time:** about 30 to 45 minutes for a full run, most of it waiting; the checks themselves took 19 minutes on a T4. A P-0005-only run (`RUN = "p0005"`, below) skips the checks and takes a few minutes.
- **Needs:** a Google account, to use Colab. The kit itself uses no tokens and no logins: the models are ungated, nothing from `.env` goes in the kit, and the kit turns off Hugging Face's token lookup.
- **Cost:** free. Colab's free GPUs aren't always available; if you get none, try again later.

## What the kit is

Three files, built on your computer from the committed code (`git archive` of HEAD, never a copy of your working folder):

| File | What it is |
| --- | --- |
| `parity_gpu_run.ipynb` | The Colab notebook: 7 short cells (cell 6 is optional). |
| `parity-src-<commit>.zip` | The code at that commit. Git writes the commit id into the zip, and the notebook checks it. |
| `KIT.txt` | The commit, the zip's sha256, and when it was built. |

In Colab, the notebook installs Unsloth's environment from Parity's lockfile into its own virtual environment. It picks the lockfile your GPU's driver can run (CUDA 13, or the CUDA 12.6 variant). It refuses to run if Unsloth doesn't take its GPU route, then runs the checks and zips the verdicts, `environment.json` (every deviation from Unsloth's documented install) and the logs.

## Steps

### A. Build the kit (on your computer, in WSL)

1. Open your WSL terminal and go to the repo: `cd ~/parity`
2. Make sure everything is committed: `git status` should say "nothing to commit". The kit is built from the last commit only, so uncommitted changes would be left out. The build refuses if tracked files have changes.
3. Build it: `uv run python docs/gpu-run/build_kit.py`. For target models, add `--models org/name@<sha>,...` (I'll give you the exact command); it refuses a target whose predictions aren't committed, and writes the list into the kit's notebook.
4. It prints a line starting `folder (Windows):`, like `\\wsl.localhost\Ubuntu-22.04\home\krishg\parity\dist\gpu-kit\<commit>`. Copy that path.
5. Open **File Explorer** in Windows, paste the path into the address bar at the top, and press Enter. You should see the three files above. Keep this window open.

### B. Open the notebook in Colab (in your browser)

6. Go to https://colab.research.google.com and sign in with your Google account.
7. An "Open notebook" window appears. (If it doesn't, use the menu **File > Upload notebook**.) Click the **Upload** tab, then **Browse**. Pick `parity_gpu_run.ipynb` from the kit folder. Tip: paste the Windows path from step 4 into the file-name box of the file picker and press Enter to get there.
8. The notebook opens. Colab may warn "This notebook was not authored by Google": that's expected, since you wrote it. Choose **Run anyway** when it asks.

### C. Pick the GPU (do this before running anything)

9. Menu **Runtime > Change runtime type**.
10. Under "Hardware accelerator", choose **T4 GPU**, then click **Save**.
11. Top right, click **Connect** if it isn't connected yet. When it's ready, it shows RAM and Disk bars.

### D. Run the cells, in order

Each code cell has a ▶ (play) button at its top left. Click it and wait until the cell finishes (the spinner stops and a green check appears) before starting the next one.

12. **Cell 2 (MODELS and RUN):** ▶. Nothing prints. Leave `MODELS` as it is: the kit wrote the models to run (the calibration model unless I said otherwise). `RUN` says what to run:
    - `RUN = "full"` (the default): the six checks, about 20 minutes, with cell 6 optional;
    - `RUN = "p0005"`: only P-0005, a few minutes: install, the GPU check, then the P-0005 runs. No checks run and no verdicts are made. Use this when you're asked for a P-0005-only trip: change the word between the quotes, then run the cell.
13. **Cell 3 (GPU check):** ▶. A table prints with **Tesla T4** in it. If it says "No GPU", go back to step 9.
14. **Cell 4 (upload):** ▶. A **Choose Files** button appears under the cell. Click it and pick `parity-src-<commit>.zip` from the kit folder. Wait for "100% done". The cell then prints `Code from commit ...`: check that it matches the commit in `KIT.txt`.
15. **Cell 5 (the run):** ▶. With `RUN = "full"` this takes about 20 to 40 minutes; with `RUN = "p0005"`, a few minutes, and the `checks` step below is replaced by the P-0005 runs. **Keep the browser tab open and the computer awake.** As soon as each model finishes, its results are saved as a zip of their own (`parity-gpu-model-<commit>-<model>-<time>.zip`, in `/content`), and the output says so. Nothing downloads during the run: a browser can block a page's repeated downloads (Chrome did, 2026-11-26), so cell 7 makes one download at the end. If the session might end early, take a model's zip by hand from the Files panel as soon as it's saved (as in step 17). It prints each step as it goes:
    - `platform`: which GPU and driver;
    - `choose`: which lockfile the driver allows;
    - `uv` and `sync`: installing, the longest step (several GB of downloads);
    - `freeze`: checks every installed version against the lockfile;
    - `smoke`: torch uses the GPU, and Unsloth takes its GPU route;
    - `checks`: the six checks, both modes.

    It ends with either `DONE:` or `STOPPED EARLY:`, followed by the results zip's name. **Both are fine to bring back.** A stopped run's zip holds the logs that say why.
16. **Cell 6 (optional, P-0005):** ▶, or skip straight to cell 7. With `RUN = "p0005"`, skip it: cell 5 already did this. It takes a few minutes and runs four things in the same environment:
    - the P-0005 repro with padded batches;
    - the repro with batching left to Unsloth;
    - the repro with one real training step, recording the batch training actually uses;
    - Parity's adapter at the padding size the check asks for, with and without the assistant-only step.

    Its output goes into the same results zip, in its own `P-0005` folder, apart from the verdicts.
    - It ends with `P-0005 done` or `P-0005 did not complete`. **Both are fine.**
    - It never changes the main results. If it fails, the zip from cell 5 stays exactly as it was. If it works, the zip gets the extra folder.
17. **Cell 7 (download):** ▶. Your browser makes one download, `parity-gpu-results-<commit>-<time>.zip`, which holds every model's results. If the browser asks to allow downloads, allow it.
    - Cell 7 then prints every zip to keep: the results zip, then the per-model zips. If nothing downloaded, or the browser blocked it, take each listed file by hand: click the **folder icon** in Colab's left sidebar, go to `/content` (you may need to click `..` once), click the file's **⋮** menu, and choose **Download**. The results zip is the one that matters; the per-model zips are what to bring back if there is no results zip.
    - Cell 5's output ends with the same list (`KEEP=` lines), in case cell 7 never runs.

### E. Bring the zip back

18. In File Explorer, go to `\\wsl.localhost\Ubuntu-22.04\home\krishg\parity\dist`. Make a folder named `gpu-results` there, and move the downloaded zips into it (the results zip, and any per-model zips you took). **Don't unzip it or change it.** Its manifest holds the sha256 of every file, and I check them.
19. Tell Claude Code: "The GPU results zip is at `dist/gpu-results/<file name>.zip`".
20. Back in Colab: menu **Runtime > Disconnect and delete runtime**, so the free GPU isn't held.

## If something goes wrong

| What you see | What to do |
| --- | --- |
| "Cannot connect to GPU backend", or a usage-limit message | No free GPU right now. Try again later. |
| Cell 3 says "No GPU" | Steps 9 to 11, then run cells 2 and 3 again. |
| Cell 4 says "Upload exactly one parity-src-<commit>.zip" | Run cell 4 again and pick only that one file. |
| Cell 4 says "This zip wasn't made by build_kit.py" | Rebuild the kit (part A) and use the new files together. |
| Colab disconnected during cell 5 or 6 | Reconnect, then run cells 2 to 7 again (cell 4 needs the upload again). |
| Cell 5 ends with `STOPPED EARLY` | Skip cell 6 (it refuses without a confirmed GPU route), run cell 7, and bring the zip back. |
| Cell 6 says "Run cell 5 first" | Run cell 5, then cell 6 again. Or skip cell 6: it's optional. |
| Cell 6 ends with `P-0005 did not complete` | Fine: run cell 7. The reason is in the zip's `P-0005/run.json` and logs. |
| Cell 7 says "No results zip was written" | Copy everything cell 5 printed into a text file and bring that back instead. |

## What the kit never does

- It never uses a token or login, and nothing from `.env` is in it.
- It never downloads model weights: only tokenizer, template and config files (rule 6). The model is a tiny random-weight one.
- It never posts anything anywhere. The results come back only as the zip you download.
- It never falls back to CPU. `UNSLOTH_DISABLE_CPU_FALLBACK=1` is set, and the smoke step stops the run if Unsloth isn't on its GPU route.

## For later runs (M5 targets)

- The same kit runs other models: build it with `--models`, or edit the `MODELS` list in cell 2. Each entry is `org/name@<40-character commit>`, and every target needs its committed Unsloth predictions in the code zip, or `run_checks.py` refuses it.
- Two conditions first, for each target:
  - its `predictions.md` is committed;
  - the Unsloth adapter supports that model's chat format. Today its assistant-only step uses Qwen's ChatML parts, so other model families need adapter work first.
- After any code change, build a fresh kit (part A), so the zip matches the commit.

## What happens to the zip (Claude's side)

The zip is treated as untrusted data and unpacked into a new, empty folder. Then:

1. Every file's sha256 is checked against the zip's `MANIFEST.json`. The optional `P-0005/` folder has its own `MANIFEST.json`, since it was added after the main run.
2. The commit must be in this repo. `git archive` of that commit is rebuilt, and its sha256 must equal the one Colab recorded, which shows that the code that ran is exactly that commit.
3. `environment.json` must say the GPU route ran (the smoke facts), and its lockfile, mismatches and deviations are read.
4. The P-0005 repro's output, if present, goes to `/triage` for P-0005, separately from the verdicts.
5. The verdicts are compared with `docs/evidence/2026-11-25-m3-unsloth-gpu/predictions.md`, every fail goes through `/triage`, and the evidence is committed under `docs/evidence/2026-11-25-m3-unsloth-gpu/`.

## Files

- `docs/gpu-run/build_kit.py`: builds the kit from HEAD (part A).
- `docs/gpu-run/parity_gpu_run.ipynb`: the notebook.
- `docs/gpu-run/colab_kit.py`: what cell 5 runs, using only Python's standard library. Steps: platform, choose, uv, sync, freeze, smoke, checks. Cell 6 runs it again with `--extra p0005`.
- `findings/P-0005/triage_repro.py` (padded, `--padding-free-unset`, `--train-step`) and `findings/P-0005/adapter_check.py`: what P-0005 runs, from cell 6 or with `RUN = "p0005"` (`colab_kit.py --only p0005`).
- `docs/gpu-run/run_checks.py`: the checks inside Unsloth's environment (`--allow-cpu` exists only to test the kit without a GPU).
- `envs/unsloth-2026.11.10/uv.lock`: the main lockfile (torch 2.14.1 for CUDA 13.0).
- `envs/unsloth-2026.11.10/cu126/uv.lock`: the CUDA 12.6 variant, for drivers older than CUDA 13. It differs from the main lock only in torch, torchvision and their CUDA libraries (a test holds it to that).
