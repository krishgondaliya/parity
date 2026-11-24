"""The guard hook (.claude/hooks/guard.py) must deny upstream writes and any
reference to the hidden vault, for both shell tools, and allow normal work.

Each case feeds the hook the JSON event Claude Code sends on stdin and reads its
decision. A deny is a JSON object with permissionDecision "deny"; an allow is no
output. The exit code is 0 either way.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
GUARD = ROOT / ".claude" / "hooks" / "guard.py"
SHELL_TOOLS = ("Bash", "PowerShell")


def _guard_module():
    spec = importlib.util.spec_from_file_location("parity_guard", GUARD)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# The vault folder name is read from guard.py instead of being spelled out here,
# because the guard blocks any tool call that contains it, including the edit
# that writes this file. test_vault_marker_matches_gitignore pins its value
# against a second source.
VAULT = _guard_module().HIDDEN_VAULT_MARKER


def run_guard(tool_name: str, tool_input: dict) -> tuple[str, str | None]:
    event = {
        "session_id": "test",
        "hook_event_name": "PreToolUse",
        "tool_name": tool_name,
        "tool_input": tool_input,
    }
    proc = subprocess.run(
        [sys.executable, str(GUARD)],
        input=json.dumps(event),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    if not proc.stdout.strip():
        return "allow", None
    output = json.loads(proc.stdout)["hookSpecificOutput"]
    assert output["hookEventName"] == "PreToolUse"
    return output["permissionDecision"], output["permissionDecisionReason"]


# Deny: posting upstream, from either shell tool.

UPSTREAM_WRITES = [
    "gh issue create --title test --body test",
    "gh pr comment 12 --body looks-good",
    "gh api -X POST repos/owner/repo/issues",
    "gh api repos/owner/repo/issues -f title=test",
    "gh.exe issue create --title test --body test",
    "gh.exe pr comment 12 --body looks-good",
]


@pytest.mark.parametrize("tool", SHELL_TOOLS)
@pytest.mark.parametrize("command", UPSTREAM_WRITES)
def test_upstream_writes_are_denied(tool, command):
    decision, reason = run_guard(tool, {"command": command})
    assert decision == "deny"
    assert "rule 1" in reason


# Deny: anything that mentions the hidden vault, whatever the tool.

VAULT_CALLS = [
    ("Read", {"file_path": f"C:\\Users\\someone\\{VAULT}\\pairs\\V-0001.json"}),
    ("Bash", {"command": f"ls ~/{VAULT}"}),
    ("PowerShell", {"command": f"Get-ChildItem C:\\Users\\someone\\{VAULT}"}),
    ("Grep", {"pattern": "bug", "path": f"/home/someone/{VAULT}"}),
    ("Glob", {"pattern": f"**/{VAULT}/**"}),
    ("Edit", {"file_path": f"/home/someone/{VAULT}/notes.md", "old_string": "a", "new_string": "b"}),
    ("Write", {"file_path": "notes.md", "content": f"see ~/{VAULT}/pairs"}),
]


@pytest.mark.parametrize("tool, tool_input", VAULT_CALLS, ids=[c[0] for c in VAULT_CALLS])
def test_any_hidden_vault_reference_is_denied(tool, tool_input):
    decision, reason = run_guard(tool, tool_input)
    assert decision == "deny"
    assert "rule 9" in reason


def test_vault_marker_matches_gitignore():
    assert f"{VAULT}/" in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()


# Allow: normal read-only work.

ALLOWED_SHELL = [
    "git status",
    "gh issue view 5471 --repo huggingface/trl",
    "gh search issues assistant_only_loss --repo huggingface/trl",
    'gh api "search/issues?q=repo:huggingface/trl+assistant_only_loss"',
]


@pytest.mark.parametrize("tool", SHELL_TOOLS)
@pytest.mark.parametrize("command", ALLOWED_SHELL)
def test_read_only_shell_commands_are_allowed(tool, command):
    assert run_guard(tool, {"command": command}) == ("allow", None)


def test_reading_a_source_file_is_allowed():
    path = ROOT / "src" / "parity" / "release.py"
    assert run_guard("Read", {"file_path": str(path)}) == ("allow", None)
