"""The public README, rendered from the records it cites.

README.md is generated. `uv run python -m parity.readme` renders `docs/readme/README.in.md`, filling every
`{{name}}` with a number, status or table derived from the repo's own records, and writes README.md.
`--check` writes nothing and exits 1 if README.md has drifted from them. tests/test_readme_claims.py re-derives
the main claims a second way and fails on drift too.

Sources:
- findings/LEDGER.md: each finding's framework, what happens, and whether it counts against the framework; the
  harness-error count.
- findings/<id>/record.md: the finding's status, severity, and the Upstream fields of findings/_TEMPLATE.md (the
  issue, maintainer responses, fixes). So the README's upstream sections fill in as the records do.
- The calibration verdicts (CALIBRATION_DIR): counts, green status, and the planted faults as run.
- The Tier 1 check modules: what each check asks, and its planted faults in their documented order.
- docs/evidence/*/predictions.md and compare.md: pre-registered predictions and their hit rates.
- src/parity/checks/known_differences.json, envs/*/environment.json, .python-version.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

REPO = Path(__file__).resolve().parents[2]
TEMPLATE = REPO / "docs" / "readme" / "README.in.md"
README = REPO / "README.md"
LEDGER = REPO / "findings" / "LEDGER.md"
KNOWN_DIFFERENCES = REPO / "src" / "parity" / "checks" / "known_differences.json"
CALIBRATION_DIR = "docs/evidence/2026-11-25-m4-calibration"
CALIBRATION_REPORT = "runs/calibration-2026-11-25.md"
CALIBRATED = ("trl", "axolotl", "unsloth")  # the report's order
MODES = ("thinking", "non_thinking")
DISPLAY = {"trl": "TRL", "axolotl": "Axolotl", "unsloth": "Unsloth"}
PLACEHOLDER = re.compile(r"\{\{([a-z][a-z0-9_.]*)\}\}")
WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve")

# How a record's Status line starts -> (kind, how the README shows it). The first match wins.
STATUS_KINDS = (
    ("closed: not_configurable", "not_configurable", "closed: not configurable"),
    ("harness error", "harness_error", "harness error (ours)"),
    ("difference", "difference", "known difference"),
    ("retracted", "retracted", "retracted"),
    ("fixed", "fixed", "fixed upstream"),
    ("confirmed", "confirmed", "confirmed by a maintainer"),
    ("candidate", "candidate", "candidate"),
)
RULED_OUT = frozenset({"not_configurable", "harness_error", "difference", "retracted"})


# Markdown helpers.


def table_cells(line: str) -> list[str]:
    """The cells of one Markdown table row. A pipe inside a code span, or an escaped pipe, stays in its cell."""
    body = line.strip()
    body = body[1:] if body.startswith("|") else body
    body = body[:-1] if body.endswith("|") and not body.endswith("\\|") else body
    cells, cell, in_code, i = [], [], False, 0
    while i < len(body):
        ch = body[i]
        if ch == "\\" and body[i + 1 : i + 2] == "|":
            cell.append("|")
            i += 2
            continue
        if ch == "`":
            in_code = not in_code
        if ch == "|" and not in_code:
            cells.append("".join(cell).strip())
            cell = []
        else:
            cell.append(ch)
        i += 1
    cells.append("".join(cell).strip())
    return cells


def section(text: str, heading: str) -> str:
    """The body under the first line starting with `heading`, up to the next heading of the same level."""
    level = heading.split(" ", 1)[0]
    lines, inside = [], False
    for line in text.splitlines():
        if inside and line.startswith(level + " "):
            break
        if inside:
            lines.append(line)
        elif line.startswith(heading):
            inside = True
    if not inside:
        raise ValueError(f"no section {heading!r}")
    return "\n".join(lines)


def cell(text: str) -> str:
    """Text made safe for a table cell: special tokens in code, pipes escaped."""
    text = re.sub(r"(?<!`)(<\|[A-Za-z_]+\|>)(?!`)", r"`\1`", text)
    return text.replace("|", "\\|")


def link_text(url: str) -> str:
    """owner/repo#N for a GitHub issue or pull request, else the URL."""
    m = re.fullmatch(r"https://github\.com/([^/]+/[^/]+)/(?:issues|pull)/(\d+)(#\S+)?", url)
    if not m:
        return url
    return f"{m.group(1)}#{m.group(2)}" + (" (comment)" if m.group(3) else "")


def word(n: int) -> str:
    return WORDS[n] if 0 <= n < len(WORDS) else str(n)


def first_url(text: str) -> str | None:
    m = re.search(r"https://[^\s)>\]]+", text)
    return m.group(0).rstrip(".,;:") if m else None


def rel(path: Path) -> str:
    return path.relative_to(REPO).as_posix()


# Findings: the ledger plus each record.


@dataclass(frozen=True)
class LedgerRow:
    id: str
    framework: str
    what: str
    status: str
    counted: str


@dataclass(frozen=True)
class Response:
    date: str
    url: str
    quote: str


@dataclass(frozen=True)
class Fix:
    what: str
    url: str
    rerun: str | None


@dataclass(frozen=True)
class Upstream:
    issue_url: str | None  # a posted issue
    posted: str | None  # its date, when the record gives one
    draft: str | None  # a drafted issue not posted yet
    note: str  # the Issue URL field as written
    responses: tuple[Response, ...]
    fixes: tuple[Fix, ...]


@dataclass(frozen=True)
class Finding:
    id: str
    record: str
    framework: str
    what: str
    counted: str
    kind: str
    outcome: str
    severity: str | None
    upstream: Upstream

    @property
    def ruled_out(self) -> bool:
        return self.kind in RULED_OUT


def ledger_rows(text: str) -> list[LedgerRow]:
    rows = []
    for line in section(text, "## Findings").splitlines():
        if not line.startswith("| ["):
            continue
        cells = table_cells(line)
        m = re.fullmatch(r"\[(P-\d{4})\]\((P-\d{4})/record\.md\)", cells[0])
        if not m or m.group(1) != m.group(2) or len(cells) != 5:
            raise ValueError(f"ledger row not understood: {line[:80]}")
        rows.append(LedgerRow(m.group(1), *cells[1:]))
    return rows


def harness_errors(text: str) -> int:
    """The ledger's harness-error count. Its heading and its table must agree."""
    m = re.search(r"^## Harness errors caught: (\d+)$", text, re.M)
    if not m:
        raise ValueError("the ledger has no 'Harness errors caught' heading")
    rows = re.findall(r"^\| H-(\d+) \|", text, re.M)
    if int(m.group(1)) != len(rows) or [int(r) for r in rows] != list(range(1, len(rows) + 1)):
        raise ValueError(f"harness errors: the heading says {m.group(1)}, the table has {len(rows)} rows")
    return len(rows)


def classify(status: str) -> tuple[str, str]:
    plain = status.replace("*", "").strip()
    for prefix, kind, shown in STATUS_KINDS:
        if plain.lower().startswith(prefix):
            if kind == "difference":
                ids = sorted(set(re.findall(r"KD-\d{4}", plain)))
                shown += f" ({', '.join(ids)})" if ids else ""
            if kind == "harness_error":
                ids = re.findall(r"\bH-\d+\b", plain)
                shown += f", {ids[0]}" if ids else ""
            return kind, shown
    raise ValueError(f"status not understood: {status[:80]!r}")


def field(text: str, name: str) -> list[str]:
    return [v.strip() for v in re.findall(rf"^- {re.escape(name)}: (.+)$", text, re.M)]


def parse_upstream(record: str, finding_id: str) -> Upstream:
    body = section(record, "## Upstream")
    issue = field(body, "Issue URL (posted by human)")
    if len(issue) != 1:
        raise ValueError(f"{finding_id}: its Upstream section needs one 'Issue URL (posted by human):' line")
    note = issue[0]
    url = first_url(note)
    posted = re.search(r"posted (\d{4}-\d{2}-\d{2})", note)
    draft = None
    if url is None and note.lower().startswith("not posted"):
        m = re.search(r"findings/P-\d{4}/issue\.md", note)
        if not m:
            raise ValueError(f"{finding_id}: a 'not posted' issue names its draft, findings/<id>/issue.md")
        draft = m.group(0)
    elif url is None and not note.lower().startswith("none"):
        raise ValueError(f"{finding_id}: Issue URL is a URL, 'not posted; draft in ...' or 'none': {note!r}")

    responses = []
    for value in field(body, "Maintainer response"):
        if value.lower().startswith("none"):
            continue
        m = re.fullmatch(r'(\d{4}-\d{2}-\d{2}), (https://\S+), "(.+)"', value)
        if not m:
            raise ValueError(f'{finding_id}: a maintainer response reads: YYYY-MM-DD, <url>, "<quote>": {value!r}')
        responses.append(Response(*m.groups()))
    fixes = []
    for value in field(body, "Fixed in"):
        if value.lower().startswith(("not yet", "n/a", "none")):
            continue
        m = re.fullmatch(r"(.+?), (https://\S+?)(?:, re-run passes: (.+))?", value)
        if not m:
            raise ValueError(f"{finding_id}: a fix reads: <version or commit>, <url>[, re-run passes: ...]: {value!r}")
        fixes.append(Fix(*m.groups()))
    if not field(body, "Maintainer response") or not field(body, "Fixed in"):
        raise ValueError(f"{finding_id}: its Upstream section needs 'Maintainer response:' and 'Fixed in:' lines")
    return Upstream(url, posted.group(1) if posted else None, draft, note, tuple(responses), tuple(fixes))


def display_framework(ledger_framework: str) -> str:
    name, _, rest = ledger_framework.partition(" ")
    return f"{DISPLAY.get(name, name)} {rest}".strip()


def findings() -> list[Finding]:
    out = []
    for row in ledger_rows(LEDGER.read_text(encoding="utf-8")):
        path = REPO / "findings" / row.id / "record.md"
        text = path.read_text(encoding="utf-8")
        head = text.split("\n## ", 1)[0]
        status = field(head, "Status")
        if len(status) != 1:
            raise ValueError(f"{row.id}: its record needs one '- Status:' line")
        kind, outcome = classify(status[0])
        severity = field(head, "Severity")
        if kind not in RULED_OUT and len(severity) != 1:
            raise ValueError(f"{row.id}: a {kind} finding's record needs one '- Severity:' line")
        out.append(
            Finding(
                id=row.id,
                record=rel(path),
                framework=display_framework(row.framework),
                what=row.what,
                counted=row.counted,
                kind=kind,
                outcome=outcome,
                severity=severity[0] if severity else None,
                upstream=parse_upstream(text, row.id),
            )
        )
    return out


def issue_cell(f: Finding) -> str:
    u = f.upstream
    if u.issue_url:
        posted = f", posted {u.posted}" if u.posted else ""
        reply = "; maintainer replied (below)" if u.responses else "; no maintainer reply yet"
        return f"[{link_text(u.issue_url)}]({u.issue_url}){posted}{reply}"
    if u.draft:
        return f"drafted, not posted ([draft]({u.draft}))"
    return cell(u.note)


def findings_table(items: list[Finding]) -> str:
    rows = [
        "| ID | Framework | What happens | Severity | Status | Upstream issue |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for f in items:
        if not f.ruled_out:
            rows.append(
                f"| [{f.id}]({f.record}) | {cell(f.framework)} | {cell(f.what)} | {cell(f.severity or '')} "
                f"| {f.outcome} | {issue_cell(f)} |"
            )
    return "\n".join(rows)


def ruled_out_table(items: list[Finding]) -> str:
    rows = [
        "| ID | Framework | What we saw | Outcome | Counted against the framework? |",
        "| --- | --- | --- | --- | --- |",
    ]
    for f in items:
        if f.ruled_out:
            rows.append(
                f"| [{f.id}]({f.record}) | {cell(f.framework)} | {cell(f.what)} | {f.outcome} | {cell(f.counted)} |"
            )
    return "\n".join(rows)


def upstream_fixes(items: list[Finding]) -> str:
    lines = []
    for f in items:
        for x in f.upstream.fixes:
            rerun = f"; Parity's re-run passes: {x.rerun}" if x.rerun else ""
            fix = f"fixed in {x.what}, [{link_text(x.url)}]({x.url}){rerun}"
            lines.append(f"- [{f.id}]({f.record}), {cell(f.framework)}: {fix}")
    if not lines:
        lines = ["None yet."]
    awaiting = [f for f in items if f.upstream.issue_url and not f.upstream.fixes and not f.ruled_out]
    if awaiting:
        lines += ["", "Reported upstream, awaiting a fix:", ""]
        for f in awaiting:
            posted = f", posted {f.upstream.posted}" if f.upstream.posted else ""
            issue = f"[{link_text(f.upstream.issue_url)}]({f.upstream.issue_url})"
            lines.append(f"- {issue} ([{f.id}]({f.record}), {cell(f.framework)}){posted}")
    return "\n".join(lines)


def acknowledgments(items: list[Finding]) -> str:
    lines = [
        f'- [{f.id}]({f.record}), [{link_text(r.url)}]({r.url}) ({r.date}): "{r.quote}"'
        for f in items
        for r in f.upstream.responses
    ]
    return "\n".join(lines) if lines else "None yet."


# The checks and their planted faults.


@dataclass(frozen=True)
class PlantInfo:
    name: str
    description: str
    origin: str
    source_url: str | None


@dataclass(frozen=True)
class CheckInfo:
    id: str
    question: str
    module: str
    plants: tuple[PlantInfo, ...]


def documented_plants(doc: str) -> list[str]:
    """Plant names in a check module's docstring, in order: the '- name' lines after 'Planted fault'."""
    after = doc.split("Planted fault", 1)[1]
    return re.findall(r"^- ([a-z_]+)\b", after, re.M)


def checks(verdicts: list[Any]) -> list[CheckInfo]:
    """The six checks in suite order, with each planted fault as the calibration verdicts recorded it."""
    from parity.checks.tier1 import TIER1_CHECKS

    seen: dict[str, dict[str, PlantInfo]] = {}
    for v in verdicts:
        for p in v.plants:
            info = PlantInfo(p["plant"], p["description"], p["origin"], first_url(p.get("source") or ""))
            seen.setdefault(v.check, {}).setdefault(p["plant"], info)
    out = []
    for check in TIER1_CHECKS:
        module = sys.modules[type(check).__module__]
        doc = module.__doc__ or ""
        question = doc.split("\n", 1)[0].split(": ", 1)[1]
        order = documented_plants(doc)
        ran = seen.get(check.id, {})
        if sorted(order) != sorted(ran):
            raise ValueError(f"{check.id}: documented plants {order} differ from the calibration's {sorted(ran)}")
        path = Path(module.__file__).resolve()
        out.append(CheckInfo(check.id, question[0].upper() + question[1:], rel(path), tuple(ran[n] for n in order)))
    return out


def checks_table(items: list[CheckInfo]) -> str:
    rows = ["| Check | What it asks | Planted faults it must catch |", "| --- | --- | --- |"]
    for c in items:
        faults = []
        for p in c.plants:
            tag = f" ([real-world]({p.source_url}))" if p.origin == "real-world" and p.source_url else ""
            faults.append(f"`{p.name}`: {p.description}{tag}")
        rows.append(f"| [`{c.id}`]({c.module}) | {cell(c.question)} | {cell('<br>'.join(faults))} |")
    return "\n".join(rows)


# Calibration.


@dataclass(frozen=True)
class CalibrationRun:
    framework: str  # trl, axolotl, unsloth
    folder: str
    verdicts: tuple[Any, ...]
    status: Any  # parity.engine.calibration.Status

    @property
    def plants(self) -> list[dict]:
        return [p for v in self.verdicts for p in v.plants]

    @property
    def route(self) -> str:
        """CPU, or the GPU the run's environment.json recorded (GPU runs come back with one)."""
        backends = {v.backend for v in self.verdicts}
        if backends <= {None, "cpu"}:
            return "CPU"
        env = REPO / self.folder / "environment.json"
        gpu = json.loads(env.read_text(encoding="utf-8"))["platform"]["gpu"] if env.is_file() else "unrecorded"
        return f"GPU ({gpu})"


def calibration() -> list[CalibrationRun]:
    from parity.engine.calibration import attribute
    from parity.engine.verdict import read_verdict

    runs = []
    for framework in CALIBRATED:
        folder = REPO / CALIBRATION_DIR / framework
        verdicts = tuple(read_verdict(p) for p in sorted((folder / "verdicts").rglob("*.json")))
        [status] = attribute(verdicts)
        runs.append(CalibrationRun(framework, rel(folder), verdicts, status))
    return runs


def run_stdout(verdicts: tuple[Any, ...]) -> str:
    """What CALIBRATION_DIR/run.py prints for these verdicts (its last lines, same code)."""
    from parity.engine.root_causes import count_fails

    lines = []
    for mode in MODES:
        counts = sorted(Counter((v.result, v.kind) for v in verdicts if v.serving_mode == mode).items(), key=str)
        lines.append(f"{mode} {counts}")
    count = count_fails(verdicts)
    lines.append(f"fail verdicts: {count.cells}; root causes: {count.root_causes}; untriaged {len(count.untriaged)}")
    return "\n".join(lines)


def report_stdout(runs: list[CalibrationRun]) -> str:
    """What CALIBRATION_DIR/report.py prints after it writes the report."""
    lines = []
    for r in runs:
        s = r.status
        lines.append(
            f"{s.framework}: {'green' if s.green else 'NOT GREEN'}; untriaged {len(s.untriaged_fails)}, "
            f"harness inconclusive {len(s.harness_inconclusives)}, plants missed {len(s.missed_plants)}"
        )
    return "\n".join([*lines, f"wrote {CALIBRATION_REPORT}"])


def calibration_table(runs: list[CalibrationRun]) -> str:
    rows = [
        "| Framework | Route | Verdicts | Pass | Fail, each linked to a known difference or a triaged finding "
        "| Not judged | Planted-fault runs caught | Green |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in runs:
        s = r.status
        route = r.route
        linked = ", ".join(f"{rc} ×{len(cells)}" for rc, cells in sorted(s.linked_fails.items()))
        not_judged = ", ".join(
            f"{s.results[k]} {k.replace('_', ' ')}" for k in ("not_applicable", "not_configurable") if s.results.get(k)
        )
        caught = sum(1 for p in r.plants if p.get("caught"))
        name, _, version = r.verdicts[0].framework.partition("@")
        name = f"{DISPLAY.get(name, name)} {version}"
        rows.append(
            f"| [{name}]({r.folder}) | {route} | {len(r.verdicts)} | {s.results.get('pass', 0)} "
            f"| {s.results.get('fail', 0)} ({linked or 'none'}) | {not_judged or 'none'} | {caught} of {len(r.plants)} "
            f"| {'yes' if s.green else '**no**'} |"
        )
    return "\n".join(rows)


# Predictions, pins and registries.


@dataclass(frozen=True)
class Prediction:
    folder: str
    title: str
    hits: int
    cells: int
    caveat: str | None


def predictions() -> list[Prediction]:
    out = []
    for compare in sorted((REPO / "docs" / "evidence").glob("*/compare.md")):
        pred = compare.parent / "predictions.md"
        if not pred.is_file():
            continue
        m = re.search(r"(\d+) of (\d+) cells as predicted", compare.read_text(encoding="utf-8"))
        if not m:
            raise ValueError(f"{rel(compare)}: no 'N of M cells as predicted' line")
        text = pred.read_text(encoding="utf-8")
        title = text.split("\n", 1)[0].removeprefix("# Predictions: ").strip()
        caveat = re.search(r"\*\*(Not [^*]+?), stated plainly\.\*\*", text)
        hits, cells = int(m.group(1)), int(m.group(2))
        out.append(Prediction(rel(compare.parent), title, hits, cells, caveat.group(1) if caveat else None))
    return out


def predictions_table(items: list[Prediction]) -> str:
    rows = ["| Predictions | Cells as predicted | Caveat, stated in the file | Files |", "| --- | --- | --- | --- |"]
    for p in items:
        rows.append(
            f"| {cell(p.title)} | {p.hits} of {p.cells} | {p.caveat or '—'} "
            f"| [predictions]({p.folder}/predictions.md), [compared]({p.folder}/compare.md) |"
        )
    return "\n".join(rows)


def environments() -> list[dict]:
    envs = []
    for path in sorted((REPO / "envs").glob("*/environment.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        variants = sorted(rel(p) for p in path.parent.glob("*/uv.lock"))
        envs.append({**data, "variants": variants})
    order = {name: i for i, name in enumerate(CALIBRATED)}
    return sorted(envs, key=lambda e: order.get(e["framework"], len(order)))


def lockfiles_list(envs: list[dict]) -> str:
    lines = []
    for e in envs:
        extra = "".join(f", [`{v}`]({v})" for v in e["variants"])
        lines.append(f"- {DISPLAY[e['framework']]} {e['version']}: [`{e['lockfile']}`]({e['lockfile']}){extra}")
    for lock in sorted((REPO / "findings").glob("*/uv.lock")):
        finding = lock.parent.name
        lines.append(f"- {finding}'s standalone repro: [`{rel(lock)}`]({rel(lock)})")
    return "\n".join(lines)


def known_differences() -> list[dict]:
    return json.loads(KNOWN_DIFFERENCES.read_text(encoding="utf-8"))["entries"]


def badge(label: str, message: str, color: str) -> str:
    def esc(s: str) -> str:
        return quote(s.replace("-", "--").replace("_", "__"), safe="")

    return f'<img alt="{label}: {message}" src="https://img.shields.io/badge/{esc(label)}-{esc(message)}-{color}">'


# The claims, and rendering.


def claims() -> dict[str, str]:
    from parity.fixtures import FIXTURE_IDS
    from parity.targets import CALIBRATION

    items = findings()
    runs = calibration()
    all_verdicts = [v for r in runs for v in r.verdicts]
    check_infos = checks(all_verdicts)
    plants = [p for c in check_infos for p in c.plants]
    envs = environments()
    kds = known_differences()
    preds = predictions()
    ledger = LEDGER.read_text(encoding="utf-8")
    errors = harness_errors(ledger)
    open_findings = [f for f in items if not f.ruled_out]
    posted = [f for f in items if f.upstream.issue_url]
    green = sum(r.status.green for r in runs)
    plants_run = sum(len(r.plants) for r in runs)
    plants_caught = sum(1 for r in runs for p in r.plants if p.get("caught"))
    versions = {e["framework"]: e["version"] for e in envs}
    framework_names = [f"{DISPLAY[f]} {versions[f]}" for f in CALIBRATED]
    python = (REPO / ".python-version").read_text(encoding="utf-8").strip()

    values: dict[str, Any] = {
        "python": python,
        "fixtures.count": len(FIXTURE_IDS),
        "fixtures.count_word": word(len(FIXTURE_IDS)),
        "fixtures.list": ", ".join(f"`{f}`" for f in FIXTURE_IDS),
        "checks.count_word": word(len(check_infos)),
        "checks.table": checks_table(check_infos),
        "plants.total": len(plants),
        "plants.real_world": sum(p.origin == "real-world" for p in plants),
        "frameworks.count_word": word(len(CALIBRATED)),
        "frameworks.names": ", ".join(framework_names[:-1]) + " and " + framework_names[-1],
        "version.trl": versions["trl"],
        "version.axolotl": versions["axolotl"],
        "version.unsloth": versions["unsloth"],
        "calibration.model": CALIBRATION.spec,
        "calibration.model_name": CALIBRATION.repo_id,
        "calibration.dir": CALIBRATION_DIR,
        "calibration.report": CALIBRATION_REPORT,
        "calibration.table": calibration_table(runs),
        "calibration.green": green,
        "calibration.frameworks": len(runs),
        "calibration.verdicts": len(all_verdicts),
        "calibration.plants_run": plants_run,
        "calibration.plants_caught": plants_caught,
        "calibration.report_stdout": report_stdout(runs),
        "harness_errors": errors,
        "findings.open": len(open_findings),
        "findings.ruled_out": len(items) - len(open_findings),
        "findings.posted": len(posted),
        "findings.table": findings_table(items),
        "findings.ruled_out_table": ruled_out_table(items),
        "upstream.fixes": upstream_fixes(items),
        "upstream.acknowledgments": acknowledgments(items),
        "kd.count": len(kds),
        "predictions.table": predictions_table(preds),
        "lockfiles": lockfiles_list(envs),
    }
    for r in runs:
        if r.framework != "unsloth":
            values[f"calibration.stdout.{r.framework}"] = run_stdout(r.verdicts)
    status_color = "2ea44f" if green == len(runs) else "d73a49"
    values["badges"] = "\n".join(
        [
            badge("python", python, "3776ab"),
            badge("frameworks", " · ".join(framework_names), "informational"),
            badge("calibration", f"green on {green} of {len(runs)}", status_color),
            badge("planted faults", f"{len(plants)} ({values['plants.real_world']} real-world)", "informational"),
            badge("candidate findings", str(len(open_findings)), "orange"),
            badge("issues posted upstream", str(len(posted)), "informational"),
            badge("harness errors caught", str(errors), "lightgrey"),
        ]
    )
    return {k: str(v) for k, v in values.items()}


def render(template: str, values: dict[str, str]) -> str:
    missing = sorted({m.group(1) for m in PLACEHOLDER.finditer(template)} - set(values))
    if missing:
        raise KeyError(f"the template names claims that don't exist: {missing}")
    out = PLACEHOLDER.sub(lambda m: values[m.group(1)], template)
    if "{{" in out:
        raise ValueError("a placeholder was left unrendered")
    return out


def rendered() -> str:
    return render(TEMPLATE.read_text(encoding="utf-8"), claims())


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    text = rendered()
    if "--check" in argv:
        current = README.read_text(encoding="utf-8") if README.is_file() else ""
        if current != text:
            print("README.md has drifted from its records: run `uv run python -m parity.readme`")
            return 1
        print("README.md matches its records")
        return 0
    README.write_text(text, encoding="utf-8")
    print(f"wrote {rel(README)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
