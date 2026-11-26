# Finding templates

Copy into `findings/<P-000N>/record.md` and `findings/<P-000N>/issue.md`.

The README is rendered from each record's Status, Severity and Upstream lines (`parity.readme`), so keep their format. After editing a record, run `uv run python -m parity.readme`.

---

## record.md

```markdown
# P-000N: <one-line symptom>

- Status: candidate | confirmed | fixed | retracted | difference | harness error | closed: not_configurable
- Severity: low | medium | high (<why>), by effect x reach (docs/DECISIONS.md). Required unless the status rules it out
- Date found: YYYY-MM-DD
- Model: <org/name>@<revision>
- Framework: <name>@<version>   (lockfile: findings/P-000N/<file>.lock)
- Check: <check id>   Fixture: <fixture id>   Kind: bug | difference
- Verdict file: runs/<run-id>/<file>.json

## Evidence
- Expected: <tokens/labels at positions ...>
- Got: <...>
- First divergence: <index> (8 decoded tokens each side)

## Triage (from /triage)
- Calibration green on known-good model: yes/no
- Adapter uses documented public path + option: yes/no (link to docs)
- Fixture valid for this model: yes/no
- Reference consistent (template vs model card vs official renderer): yes/no
- Documented as intended behavior: yes/no (link)
- Duplicates searched: <query>, results: <links or none>

## Skeptic verdict
<paste the skeptic subagent's output>

## Repro
- findings/P-000N/repro.py: fails 3/3 on pinned version: yes/no
- On fixed version / main: passes / fails / not tested

## Upstream
- Issue URL (posted by human): <url> (posted YYYY-MM-DD) | not posted; draft in findings/P-000N/issue.md | none
- Maintainer response: none yet | YYYY-MM-DD, <link>, "<quote>"   (one line per response)
- Fixed in: not yet | <version or commit>, <PR or commit URL>, re-run passes: yes/no   (one line per fix)
```

---

## issue.md

```markdown
**Title:** [<area>] <one-line symptom> for <model> (<framework> <version>)

**What happens**
With `<setting>`, <what goes wrong, in one or two sentences>.

**Repro** (CPU, no full weights downloaded)
<short standalone script>

**Expected vs actual**
Expected <tokens or labels at positions ...>. Actual: <...>.

**Reference used**
<e.g. the model's own chat template via apply_chat_template>

**Environment**
<framework version>, transformers <version>, Python <version>, lockfile attached.
```
