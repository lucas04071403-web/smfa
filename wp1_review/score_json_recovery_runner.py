"""Run the unchanged official scorer with a strict-JSON apostrophe repair.

DashScope occasionally returns ``\'`` inside JSON double-quoted strings.
That escape is invalid in JSON (apostrophes need no escaping), so the
repository parser returns the raw text and drops an otherwise valid score.
Only that illegal escape is repaired; prompt, model, rubric, and validation
remain the official ones.
"""

import json
import re
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "S-MLLMUn"))
import eval_meaningful


def parse_json_with_apostrophe_escape_repair(content):
    match = re.search(r"```json\n(.*?)```", content, re.DOTALL)
    source = match.group(1).strip() if match else content
    try:
        return json.loads(source)
    except json.JSONDecodeError as initial:
        def repair(match):
            slashes = match.group(1)
            return "\\" * (len(slashes) if len(slashes) % 2 == 0 else len(slashes) - 1) + "'"

        repaired = re.sub(r"(\\+)'", repair, source)
        if repaired == source:
            raise initial
        return json.loads(repaired)


def main():
    eval_meaningful.convert_markdown_json_to_json = parse_json_with_apostrophe_escape_repair
    sys.argv = [str(ROOT / "reproduce_smfa/score.py"), *sys.argv[1:]]
    runpy.run_path(str(ROOT / "reproduce_smfa/score.py"), run_name="__main__")


if __name__ == "__main__":
    main()
