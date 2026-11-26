"""The README's numbers and statuses, re-derived from the records they come from.

README.md is rendered by `parity.readme` from docs/readme/README.in.md. These tests fail when it drifts:
- README.md must equal a fresh render, so every claim in it matches the records as they are today;
- the main claims are re-derived here a second way, reading the ledger, the records, the verdict JSON and the
  check code directly, and then looked up in the README;
- a number the template states in prose must be a rendered claim, or a constant whose source file is named below;
- the public text stays clean: no internal planning words, local paths, secrets file, email addresses, broken links.
"""

import json
import re
import subprocess
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree

import pytest

from parity import readme
from parity.checks.tier1 import TIER1_CHECKS
from parity.fixtures import FIXTURE_IDS, load_fixture
from parity.readme import table_cells

REPO = Path(__file__).resolve().parents[1]
README = REPO / "README.md"
TEMPLATE = REPO / "docs" / "readme" / "README.in.md"
LEDGER = REPO / "findings" / "LEDGER.md"
BLUEPRINT = "docs/plan/02-detailed-blueprint.md"

FINDINGS_HEADER = "| ID | Framework | What happens | Severity | Status | Upstream issue |"
RULED_OUT_HEADER = "| ID | Framework | What we saw | Outcome | Counted against the framework? |"
CALIBRATION_HEADER = "| Framework | Route | Verdicts | Pass |"
PREDICTIONS_HEADER = "| Predictions | Cells as predicted |"
CHECKS_HEADER = "| Check | What it asks | Planted faults it must catch |"

# Numbers the README states that are plan or timing constants, not derived from records: each phrase must appear in
# the README, and its source phrase in the named file.
CITED = [
    ("4 times the measured noise floor", BLUEPRINT, r"> 4 \cdot \max"),
    ("(3 runs each)", BLUEPRINT, "3 runs each"),
    ("all 5 of 5 runs", BLUEPRINT, "run both sides 5 times"),
    ("95% Wilson interval", BLUEPRINT, "95% Wilson interval"),
    ("3 loop runs from the frozen suite", BLUEPRINT, "3 loop runs from v0 with different seeds"),
    ("z = 1.96", BLUEPRINT, "z = 1.96"),
    ("over 10 iterations", "docs/DECISIONS.md", "over 10 loop iterations"),
    ("about 30 to 45 minutes", "docs/gpu-run/GPU-RUN.md", "about 30 to 45 minutes for a full run"),
    ("took 19 minutes on a T4", "docs/gpu-run/GPU-RUN.md", "the checks themselves took 19 minutes on a T4"),
    ("about 17 minutes", "CLAUDE.md", "axolotl`, ~17 min"),
    ("about 3 minutes on a laptop CPU", "CLAUDE.md", "run.py trl` (~3 min"),
    ("Cell 4 uploads", "docs/gpu-run/GPU-RUN.md", "**Cell 4 (upload):**"),
    ("Cell 5 ends with", "docs/gpu-run/GPU-RUN.md", "**Cell 5 (the run):**"),
    ("cell 7 downloads", "docs/gpu-run/GPU-RUN.md", "**Cell 7 (download):**"),
]

# Identifiers that contain digits but state no number.
IDENTIFIERS = [
    r"\d{4}-\d{2}-\d{2}",  # dates
    r"\d+(?:\.\d+)+",  # versions
    r"\b(?:P|KD|H)-\d+\b",  # finding, known-difference and harness-error ids
    r"#\d+",  # issue numbers
    r"\bQwen3(?:-0\.6B)?\b",
    r"\b0\.6B\b",
    r"\bT4\b",
    r"\bfp(?:16|32|64)\b",
    r"\bbf16\b",
    r"\bx86_64\b",
    r"\bWSL2\b",
    r"\bt1\b",
    r"\bsha256\b",
    r"(?m)^\s*\d+\. ",  # numbered-list markers
]


def readme_text() -> str:
    return README.read_text(encoding="utf-8")


def table_after(text: str, header: str) -> list[list[str]]:
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(header))
    rows = []
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        rows.append(table_cells(line))
    return rows


def rows_by_id(rows: list[list[str]]) -> dict[str, list[str]]:
    return {re.match(r"\[(P-\d{4})\]", r[0]).group(1): r for r in rows}


def heading_body(text: str, heading: str) -> str:
    return text.split(heading + "\n", 1)[1].split("\n#", 1)[0].strip()


def prose(text: str) -> str:
    """The text a reader reads as prose: no fenced blocks, inline code, link targets, HTML tags or comments."""
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.S)
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"`[^`\n]*`", " ", text)
    text = re.sub(r"\]\([^)]*\)", "] ", text)
    return re.sub(r"<[^>]+>", " ", text)


def ledger_ids() -> list[str]:
    return re.findall(r"^\| \[(P-\d{4})\]\(P-\d{4}/record\.md\)", LEDGER.read_text(encoding="utf-8"), re.M)


def record(finding_id: str) -> str:
    return (REPO / "findings" / finding_id / "record.md").read_text(encoding="utf-8")


def status_line(finding_id: str) -> str:
    [line] = re.findall(r"^- Status: (.+)$", record(finding_id).split("\n## ", 1)[0], re.M)
    return line.replace("*", "").strip()


def upstream_values(finding_id: str, name: str) -> list[str]:
    body = record(finding_id).split("\n## Upstream\n", 1)[1].split("\n## ", 1)[0]
    return re.findall(rf"^- {re.escape(name)}: (.+)$", body, re.M)


# 1. The README is a fresh render.


def test_readme_matches_a_fresh_render():
    drift = "README.md has drifted from its records: run `uv run python -m parity.readme`"
    assert readme_text() == readme.rendered(), drift


# 2. Findings: every ledger entry, in the right table, with its record's status, severity and issue.


def test_every_ledger_finding_is_in_exactly_one_table_with_its_records_status():
    text = readme_text()
    found = rows_by_id(table_after(text, FINDINGS_HEADER))
    ruled = rows_by_id(table_after(text, RULED_OUT_HEADER))
    assert sorted([*found, *ruled]) == sorted(ledger_ids()) and not set(found) & set(ruled)
    for fid in ledger_ids():
        raw = status_line(fid)
        status = raw.lower()
        if status.startswith(("candidate", "confirmed", "fixed")):
            row = found[fid]
            expected = {"candidate": "candidate", "confirmed": "confirmed by a maintainer", "fixed": "fixed upstream"}
            assert row[4] == next(v for k, v in expected.items() if status.startswith(k)), fid
        else:
            outcome = ruled[fid][3]
            if status.startswith("difference"):
                assert outcome.startswith("known difference")
                assert set(re.findall(r"KD-\d{4}", outcome)) == set(re.findall(r"KD-\d{4}", raw)), fid
            elif status.startswith("harness error"):
                assert outcome.startswith("harness error (ours)"), fid
            elif status.startswith("closed: not_configurable"):
                assert outcome == "closed: not configurable", fid
            else:
                assert status.startswith("retracted") and outcome == "retracted", fid


def unquote_tokens(text: str) -> str:
    return text.replace("`<|", "<|").replace("|>`", "|>")


def test_framework_and_what_come_from_the_ledger():
    text = readme_text()
    rows = {**rows_by_id(table_after(text, FINDINGS_HEADER)), **rows_by_id(table_after(text, RULED_OUT_HEADER))}
    for line in LEDGER.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| [P-"):
            continue
        cells = table_cells(line)
        fid = re.match(r"\[(P-\d{4})\]", cells[0]).group(1)
        name, _, rest = cells[1].partition(" ")
        assert rows[fid][1] == f"{readme.DISPLAY.get(name, name)} {rest}".strip(), fid
        # The README puts bare special tokens in code; otherwise the text is the ledger's, word for word.
        assert unquote_tokens(rows[fid][2]) == unquote_tokens(cells[2]), fid


def test_each_findings_severity_is_its_records():
    for fid, row in rows_by_id(table_after(readme_text(), FINDINGS_HEADER)).items():
        [severity] = re.findall(r"^- Severity: (.+)$", record(fid).split("\n## ", 1)[0], re.M)
        assert row[3] == severity.strip(), fid


def test_upstream_issue_column_follows_each_record():
    for fid, row in rows_by_id(table_after(readme_text(), FINDINGS_HEADER)).items():
        [issue] = upstream_values(fid, "Issue URL (posted by human)")
        url = re.search(r"https://\S+", issue)
        if url:
            assert f"]({url.group(0)})" in row[5], fid
        elif issue.startswith("not posted"):
            assert row[5].startswith("drafted, not posted") and f"](findings/{fid}/issue.md)" in row[5], fid
        else:
            assert row[5] == issue, fid


def test_upstream_fixes_and_acknowledgments_fill_in_from_the_records():
    text = readme_text()
    fixes = heading_body(text, "### Upstream fixes")
    thanks = heading_body(text, "### Maintainer acknowledgments")
    fixed, replies, awaiting = [], [], []
    for fid in ledger_ids():
        for value in upstream_values(fid, "Fixed in"):
            if not value.lower().startswith(("not yet", "n/a", "none")):
                fixed.append(re.search(r"https://\S+?(?=,|$)", value).group(0))
        for value in upstream_values(fid, "Maintainer response"):
            if not value.lower().startswith("none"):
                replies.append(re.search(r"https://\S+?(?=,|$)", value).group(0))
        [issue] = upstream_values(fid, "Issue URL (posted by human)")
        url = re.search(r"https://\S+", issue)
        kind = status_line(fid).lower()
        if url and kind.startswith(("candidate", "confirmed", "fixed")) and not any(
            not v.lower().startswith(("not yet", "n/a", "none")) for v in upstream_values(fid, "Fixed in")
        ):
            awaiting.append(url.group(0))

    assert fixes.splitlines()[0] == "Merged fixes and fix commits, from each record's \"Fixed in\" lines."
    if fixed:
        assert all(f"]({u})" in fixes for u in fixed)
    else:
        assert "\nNone yet.\n" in f"\n{fixes}\n"
    for url in awaiting:
        assert f"]({url})" in fixes.split("Reported upstream, awaiting a fix:", 1)[1]
    if replies:
        assert all(f"]({u})" in thanks for u in replies)
    else:
        assert thanks.splitlines()[-1] == "None yet."


def test_open_candidates_and_posted_issues_counts():
    text = readme_text()
    open_ids = rows_by_id(table_after(text, FINDINGS_HEADER))
    posted = [fid for fid in ledger_ids() if "https://" in upstream_values(fid, "Issue URL (posted by human)")[0]]
    assert f'alt="candidate findings: {len(open_ids)}"' in text
    assert f'alt="issues posted upstream: {len(posted)}"' in text


def test_harness_error_count_is_the_ledgers():
    ledger = LEDGER.read_text(encoding="utf-8")
    rows = re.findall(r"^\| H-\d+ \|", ledger, re.M)
    assert f"## Harness errors caught: {len(rows)}" in ledger
    text = readme_text()
    assert f"**{len(rows)} harness errors caught**" in text
    assert f'alt="harness errors caught: {len(rows)}"' in text


# 3. Calibration, from the verdict JSON.


def calibration_verdicts(framework: str) -> list[dict]:
    folder = REPO / readme.CALIBRATION_DIR / framework / "verdicts"
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(folder.rglob("*.json"))]


def is_green(verdicts: list[dict]) -> bool:
    parity = {(v["serving_mode"], v["fixture"]): v for v in verdicts if v["check"] == "t1.template_parity"}
    for v in verdicts:
        if v["result"] == "fail" and not v["root_cause"]:
            return False
        if any(not p["caught"] for p in v["plants"]):
            return False
        if v["result"] == "inconclusive":
            gate = parity.get((v["serving_mode"], v["fixture"]))
            framework_side = "blocked_by_template_parity" in v["flags"] or (v["reason"] or "").startswith(
                "the framework's tokens differ from its own declared template's render"
            )
            if not (framework_side and gate and gate["result"] == "fail" and gate["root_cause"]):
                return False
    return True


@pytest.mark.parametrize("framework", readme.CALIBRATED)
def test_calibration_row_matches_the_verdicts(framework):
    verdicts = calibration_verdicts(framework)
    results = Counter(v["result"] for v in verdicts)
    linked = Counter(v["root_cause"] for v in verdicts if v["result"] == "fail")
    plants = [p for v in verdicts for p in v["plants"]]
    [row] = [r for r in table_after(readme_text(), CALIBRATION_HEADER) if f"/{framework})" in r[0]]

    assert int(row[2]) == len(verdicts) == len(FIXTURE_IDS) * len(TIER1_CHECKS) * 2
    assert int(row[3]) == results["pass"]
    assert int(row[4].split(" ", 1)[0]) == results["fail"]
    assert dict((rc, int(n)) for rc, n in re.findall(r"([A-Z]+-\d{4}) ×(\d+)", row[4])) == dict(linked)
    for result in ("not_applicable", "not_configurable"):
        shown = re.search(rf"(\d+) {result.replace('_', ' ')}", row[5])
        assert (int(shown.group(1)) if shown else 0) == results[result], result
    assert row[6] == f"{sum(p['caught'] for p in plants)} of {len(plants)}"
    assert row[7] == ("yes" if is_green(verdicts) else "**no**")
    assert set(results) <= {"pass", "fail", "not_applicable", "not_configurable", "inconclusive"}


@pytest.mark.parametrize("framework", [f for f in readme.CALIBRATED if f != "unsloth"])
def test_expected_run_output_is_what_the_verdicts_give(framework):
    verdicts = calibration_verdicts(framework)
    lines = []
    for mode in ("thinking", "non_thinking"):
        counts = Counter((v["result"], v["kind"]) for v in verdicts if v["serving_mode"] == mode)
        counts = sorted(counts.items(), key=str)
        lines.append(f"{mode} {counts}")
    causes = Counter(v["root_cause"] for v in verdicts if v["result"] == "fail" and v["root_cause"])
    causes = dict(sorted(causes.items()))
    untriaged = sum(1 for v in verdicts if v["result"] == "fail" and not v["root_cause"])
    fails = sum(1 for v in verdicts if v["result"] == "fail")
    lines.append(f"fail verdicts: {fails}; root causes: {causes}; untriaged {untriaged}")
    assert "```text\n" + "\n".join(lines) + "\n```" in readme_text()


def test_expected_report_output_is_what_the_verdicts_give():
    text = readme_text()
    for framework in readme.CALIBRATED:
        verdicts = calibration_verdicts(framework)
        [label] = {v["framework"] if v["backend"] is None else f"{v['framework']} ({v['backend']})" for v in verdicts}
        state = "green" if is_green(verdicts) else "NOT GREEN"
        untriaged = sum(1 for v in verdicts if v["result"] == "fail" and not v["root_cause"])
        missed = sum(1 for v in verdicts for p in v["plants"] if not p["caught"])
        inconclusive = sum(1 for v in verdicts if v["result"] == "inconclusive")  # none of these runs has one
        assert inconclusive == 0, framework
        line = f"{label}: {state}; untriaged {untriaged}, harness inconclusive 0, plants missed {missed}"
        assert f"\n{line}\n" in text, line
    green = sum(is_green(calibration_verdicts(f)) for f in readme.CALIBRATED)
    assert f'alt="calibration: green on {green} of {len(readme.CALIBRATED)}"' in text


# 4. The checks table, from the check code.


def test_checks_table_matches_the_checks_and_their_planted_faults(qwen3, official):
    rows = {re.match(r"\[`([^`]+)`\]", r[0]).group(1): r for r in table_after(readme_text(), CHECKS_HEADER)}
    assert list(rows) == [c.id for c in TIER1_CHECKS]
    total = real = 0
    for check in TIER1_CHECKS:
        origins = {}
        for fid in FIXTURE_IDS:
            fixture = load_fixture(fid)
            if check.not_applicable(fixture) is None:
                for p in check.plants(qwen3, fixture, official.declare):
                    origins[p.name] = p.origin
        parts = rows[check.id][2].split("<br>")
        shown = [re.match(r"`([a-z_]+)`: ", part).group(1) for part in parts]
        assert sorted(shown) == sorted(origins), check.id
        tagged = {name for name, part in zip(shown, parts) if "([real-world](https://" in part}
        assert tagged == {n for n, o in origins.items() if o == "real-world"}, check.id
        total += len(origins)
        real += sum(o == "real-world" for o in origins.values())
    text = readme_text()
    assert f"{total} in all, of which {real} reproduce a bug someone actually hit" in text
    assert f'alt="planted faults: {total} ({real} real-world)"' in text


def test_fixture_and_known_difference_counts():
    text = readme_text()
    assert f"Fixtures<br/>{len(FIXTURE_IDS)} test conversations" in text
    kds = json.loads((REPO / "src/parity/checks/known_differences.json").read_text(encoding="utf-8"))["entries"]
    assert f"{len(kds)} entries)" in text


# 5. Predictions: hit rates recounted, and committed before their results.


def prediction_folders() -> list[Path]:
    found = (REPO / "docs" / "evidence").glob("*/predictions.md")
    return sorted(p.parent for p in found if (p.parent / "compare.md").is_file())


@pytest.mark.parametrize("folder", prediction_folders(), ids=lambda p: p.name)
def test_prediction_hit_rate_is_recounted_from_the_comparison(folder):
    compare = (folder / "compare.md").read_text(encoding="utf-8")
    header = next(line for line in compare.splitlines() if line.startswith("|") and "| Match |" in line)
    match = table_cells(header).index("Match")
    rows = [table_cells(line) for line in compare.splitlines() if line.startswith("| ") and "---" not in line][1:]
    hits = sum(r[match] == "yes" for r in rows)
    assert all(r[match] in ("yes", "**no**") for r in rows)
    assert f"{hits} of {len(rows)} cells as predicted" in compare
    rel = folder.relative_to(REPO).as_posix()
    [row] = [r for r in table_after(readme_text(), PREDICTIONS_HEADER) if f"]({rel}/predictions.md)" in r[3]]
    assert row[1] == f"{hits} of {len(rows)}"


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True).stdout.strip()


@pytest.mark.parametrize("folder", prediction_folders(), ids=lambda p: p.name)
def test_predictions_were_committed_before_their_results(folder):
    try:
        if git("rev-parse", "--is-shallow-repository") == "true":
            pytest.skip("shallow clone: history incomplete")
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    rel = folder.relative_to(REPO).as_posix()
    predicted = git("log", "--diff-filter=A", "--format=%H", "--", f"{rel}/predictions.md").splitlines()[-1]
    compared = git("log", "--diff-filter=A", "--format=%H", "--", f"{rel}/compare.md").splitlines()[-1]
    assert predicted != compared
    subprocess.run(["git", "-C", str(REPO), "merge-base", "--is-ancestor", predicted, compared], check=True)


# 6. No unexplained numbers in the template's prose; cited constants match their sources.


@pytest.mark.parametrize(("phrase", "source", "source_phrase"), CITED, ids=[c[0] for c in CITED])
def test_cited_constants_appear_in_their_sources(phrase, source, source_phrase):
    assert phrase in readme_text()
    assert source_phrase in (REPO / source).read_text(encoding="utf-8"), source


def test_template_prose_states_no_number_that_isnt_a_claim_or_cited():
    text = prose(readme.PLACEHOLDER.sub(" ", TEMPLATE.read_text(encoding="utf-8")))
    for phrase, _, _ in CITED:
        text = text.replace(phrase, " ")
    for pattern in IDENTIFIERS:
        text = re.sub(pattern, " ", text)
    assert re.findall(r"\S*\d\S*", text) == []


# 7. Public hygiene.


def test_no_internal_planning_words_in_the_prose():
    text = prose(readme_text())
    for pattern in (r"\bphases?\b", r"\bgates?\b", r"\bkill[- ]test\b", r"\bmilestones?\b", r"\bM\d\b", r"\bTier \d\b"):
        assert not re.search(pattern, text, re.I), pattern


def test_no_local_paths_secrets_file_or_email_addresses():
    text = readme_text()
    for pattern in (
        r"(?<![\w/-])\.env\b",
        r"/home/",
        r"(?<![\w.])~/",
        r"\\\\wsl",
        r"wsl\.localhost",
        r"[A-Za-z]:\\",
        r"/mnt/",
        r"PARITY_HIDDEN_VAULT",
        r"[\w.+-]+@[\w-]+\.[A-Za-z]{2,}",
    ):
        assert not re.search(pattern, text), pattern


def slug(heading: str) -> str:
    """GitHub's heading anchor: lower case, punctuation dropped, spaces to hyphens."""
    return re.sub(r"[^\w\- ]", "", heading.strip().lower()).replace(" ", "-")


def anchors(path: Path) -> set[str]:
    return {slug(h) for h in re.findall(r"^#{1,6} (.+)$", path.read_text(encoding="utf-8"), re.M)}


def test_relative_links_and_anchors_resolve():
    text = readme_text()
    targets = re.findall(r"\]\(([^)\s]+)\)", text) + re.findall(r'src="([^"]+)"', text)
    for target in targets:
        if target.startswith(("https://", "http://")):
            continue
        path, _, anchor = target.partition("#")
        file = (REPO / path) if path else README
        assert file.exists(), target
        if anchor:
            assert anchor in anchors(file), target


def test_banner_is_self_contained_and_brings_its_own_background():
    path = REPO / "docs" / "assets" / "parity-banner.svg"
    raw = path.read_text(encoding="utf-8")
    root = ElementTree.fromstring(raw)
    tags = {el.tag.split("}", 1)[1] for el in root.iter()}
    assert root.get("viewBox") and "title" in tags and "desc" in tags
    assert not tags & {"image", "script", "foreignObject", "use", "style"}
    assert not re.search(r"href|@import|@font-face|url\((?!#)", raw)
    card = next(el for el in root if el.tag.endswith("rect"))
    assert card.get("fill") and card.get("width") and card.get("stroke")
    assert 'src="docs/assets/parity-banner.svg"' in readme_text()
