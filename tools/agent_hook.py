"""Claude Code PostToolUse hook: check a Python file the moment an agent edits it.

Configured in .claude/settings.json. Reads the hook payload on stdin; for a .py file inside
the project it runs `tools/check.py --files <file>` (format and lint ratchet) and refuses
edits to files that a finished study pins. On a problem it exits 2 with the findings on
stderr, which Claude Code shows to the agent so it fixes them before moving on. Every other
case exits 0 without output, so the hook costs nothing for docs, data or results.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import guard  # noqa: E402  (tools/ is not a package; guard is its sibling)


def edited_file(payload: dict) -> Path | None:
    tool_input = payload.get("tool_input") or {}
    path = tool_input.get("file_path") or (payload.get("tool_response") or {}).get("filePath")
    return Path(path).resolve() if path else None


def main() -> int:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    try:
        payload = json.load(sys.stdin)
    except ValueError:
        return 0
    path = edited_file(payload)
    if path is None or path.suffix != ".py" or not path.is_relative_to(ROOT) or not path.exists():
        return 0
    relative = path.relative_to(ROOT).as_posix()
    if relative in guard.pinned_files():
        print(
            f"{relative} is pinned by a finished study (tools/guard.py, pinned-source). Revert this edit with "
            "`git checkout -- <file>` and put the new behaviour in a new module or a new study.",
            file=sys.stderr,
        )
        return 2
    run = subprocess.run(
        [sys.executable, str(ROOT / "tools/check.py"), "--files", relative],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if run.returncode == 0:
        return 0
    print(
        f"tools/check.py found problems in {relative} after this edit; fix them now:\n{run.stdout}{run.stderr}",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
