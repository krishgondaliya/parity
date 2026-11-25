#!/usr/bin/env python3
"""Parity guard hook (PreToolUse).

Blocks three things, whatever the permission settings say:
1. Posting anything upstream (issues, PRs, comments, reviews, write API calls).
   Drafts go to findings/<id>/issue.md and the human posts them.
2. Any tool call that touches the hidden bug vault.
3. Predictions first: creating or running anything in a framework environment
   (envs/<framework>-<version>) before a committed docs/evidence/*/predictions.md
   names that framework and version in its title. The predictions must be written
   from the framework's docs alone, before any code touches the environment
   (the human's decision, 2026-11-24; docs/DECISIONS.md).

Claude Code sends the tool call as JSON on stdin. To deny, this script prints a
JSON decision with hookSpecificOutput.permissionDecision = "deny" and exits 0.
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HIDDEN_VAULT_MARKER = "parity-vault-hidden"

# Tools that run shell commands. On Windows, Claude Code has a PowerShell tool
# alongside Bash, so both must be checked.
SHELL_TOOLS = ("Bash", "PowerShell")

# "gh" or "gh.exe" (Windows), as a command word.
GH = r"\bgh(?:\.exe)?"

UPSTREAM_WRITE_PATTERNS = [
    GH + r"\s+issue\s+(create|comment|edit|close|reopen|delete|transfer|lock)\b",
    GH + r"\s+pr\s+(create|comment|review|merge|edit|close|reopen)\b",
    GH + r"\s+api\b.*(-X|--method)\s*(POST|PATCH|PUT|DELETE)\b",
    GH + r"\s+api\b.*\s(-f|-F|--field|--raw-field)\s",
]


# A framework environment path, e.g. envs/unsloth-2026.11.10 or envs\\axolotl-0.21.1.
ENV_PATH = re.compile(r"\benvs[/\\]([a-z][a-z0-9_.]*?)-(\d[0-9A-Za-z.+_]*)", re.IGNORECASE)
# Something that creates or runs code in it: an interpreter, a test runner, a package tool, the env's own bin.
EXECUTES = re.compile(r"\b(python3?|pytest|uv|pip3?|venv)\b|\.venv[/\\](bin|Scripts)[/\\]", re.IGNORECASE)
# Environments that predate the rule and have no predictions file. Never add to this list.
GRANDFATHERED = {
    "trl-1.18.2": "predates the predictions-first rule (2026-11-24); its M3 run had no predictions file",
}


def repo_root() -> Path:
    for var in ("PARITY_GUARD_REPO", "CLAUDE_PROJECT_DIR"):
        if os.environ.get(var):
            return Path(os.environ[var])
    return Path(__file__).resolve().parents[2]


def committed_predictions(root: Path, framework: str, version: str) -> bool:
    """Is there a predictions.md in HEAD whose title names this framework and version?"""
    listing = subprocess.run(
        ["git", "-C", str(root), "ls-tree", "-r", "--name-only", "HEAD", "docs/evidence"],
        capture_output=True, text=True, timeout=30,
    )
    if listing.returncode != 0:
        return False
    title = re.compile(rf"\b{re.escape(framework)}\s+{re.escape(version)}\b", re.IGNORECASE)
    for path in listing.stdout.splitlines():
        if not path.endswith("/predictions.md"):
            continue
        shown = subprocess.run(
            ["git", "-C", str(root), "show", f"HEAD:{path}"], capture_output=True, text=True, timeout=30
        )
        first = shown.stdout.splitlines()[:1] if shown.returncode == 0 else []
        if first and title.search(first[0]):
            return True
    return False


def check_predictions_first(command: str) -> None:
    if not EXECUTES.search(command):
        return
    for framework, version in sorted(set(ENV_PATH.findall(command))):
        env = f"{framework.lower()}-{version}"
        if env in GRANDFATHERED:
            continue
        if not committed_predictions(repo_root(), framework, version):
            deny(
                f"Predictions first: nothing may create or run code in envs/{env} until a committed "
                f"docs/evidence/<date>-<milestone>-{framework.lower()}/predictions.md, titled "
                f"'# Predictions: <Framework> {version} on <model>', exists. Write it from the framework's docs "
                "alone, commit it, then set up the environment. No probes before that commit."
            )


def deny(reason: str) -> None:
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }))
    sys.exit(0)


def main() -> None:
    try:
        event = json.load(sys.stdin)
    except Exception:
        sys.exit(0)  # never break the session on a malformed event

    tool = event.get("tool_name", "")
    tool_input = event.get("tool_input", {}) or {}
    blob = json.dumps(tool_input)

    if HIDDEN_VAULT_MARKER in blob:
        deny(
            "Parity rule 9: the hidden vault is off limits. Only the evaluation "
            "harness reads it through PARITY_HIDDEN_VAULT. Do not read, list or "
            "reference it."
        )

    if tool in SHELL_TOOLS:
        command = str(tool_input.get("command", ""))
        for pattern in UPSTREAM_WRITE_PATTERNS:
            if re.search(pattern, command, flags=re.IGNORECASE):
                deny(
                    "Parity rule 1: never post upstream. Write the draft to "
                    "findings/<id>/issue.md; the human reviews and posts it."
                )
        check_predictions_first(command)

    sys.exit(0)


if __name__ == "__main__":
    main()
