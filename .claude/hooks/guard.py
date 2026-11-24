#!/usr/bin/env python3
"""Parity guard hook (PreToolUse).

Blocks two things, whatever the permission settings say:
1. Posting anything upstream (issues, PRs, comments, reviews, write API calls).
   Drafts go to findings/<id>/issue.md and the human posts them.
2. Any tool call that touches the hidden bug vault.

Claude Code sends the tool call as JSON on stdin. To deny, this script prints a
JSON decision with hookSpecificOutput.permissionDecision = "deny" and exits 0.
"""
import json
import re
import sys

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

    sys.exit(0)


if __name__ == "__main__":
    main()
