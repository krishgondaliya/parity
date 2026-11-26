"""P-0007: which message fields JinjaTemplateAnalyzer.get_message_vars finds, on four toy templates.

Run from the repo root, in the Axolotl environment:
    envs/axolotl-0.21.1/.venv/bin/python docs/evidence/2026-11-26-p0007-reach/analyzer_probe.py

Each template reads one field, `x`, a different way. A fresh analyzer per template, as ChatTemplatePrompter builds one
(prompt_strategies/chat_template.py get_chat_template_msg_variables). Also records a second call on the same
analyzer instance. Writes analyzer_probe.json.
"""

import json
from pathlib import Path

from axolotl.prompt_strategies.jinja_template_analyzer import JinjaTemplateAnalyzer

HERE = Path(__file__).resolve().parent
TEMPLATES = {
    "loop variable `message`, attribute": "{% for message in messages %}{{ message.x }}{% endfor %}",
    "loop variable `message`, .get()": '{% for message in messages %}{{ message.get("x") }}{% endfor %}',
    "loop variable `msg`, attribute": "{% for msg in messages %}{{ msg.x }}{% endfor %}",
    "`set msg = messages[i]`, attribute": "{% for i in range(messages|length) %}{% set msg = messages[i] %}{{ msg.x }}{% endfor %}",
}


def main() -> None:
    out = {}
    for name, template in TEMPLATES.items():
        analyzer = JinjaTemplateAnalyzer(template)
        first = sorted(analyzer.get_message_vars("messages"))
        second = sorted(analyzer.get_message_vars("messages"))
        out[name] = {"template": template, "found": first, "found_on_a_second_call": second}
        print(f"{name}: {first} (second call on the same instance: {second})")
    (HERE / "analyzer_probe.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
