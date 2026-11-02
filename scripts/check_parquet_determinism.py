"""Lint source tree for bare to_parquet calls.

Every to_parquet call must go through deterministic_parquet or be preceded
by a # parquet-lint: ignore comment on the same line.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SRC_DIR = Path("spam_fraud_detector")
PATTERN = re.compile(r"\.to_parquet\s*\(")
IGNORE_MARKER = "parquet-lint: ignore"


def check(strict: bool = False) -> tuple[int, list[str]]:
    messages = []
    for py_file in sorted(SRC_DIR.rglob("*.py")):
        try:
            lines = py_file.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            continue
        for lineno, line in enumerate(lines, 1):
            if PATTERN.search(line):
                if IGNORE_MARKER in line:
                    continue
                # check if previous line has the ignore marker
                if lineno > 1 and IGNORE_MARKER in lines[lineno - 2]:
                    continue
                # check if deterministic_parquet is on the same line or previous
                if "deterministic_parquet" in line:
                    continue
                if lineno > 1 and "deterministic_parquet" in lines[lineno - 2]:
                    continue
                messages.append(f"{py_file}:{lineno}: bare to_parquet call (use deterministic_parquet or add # parquet-lint: ignore)")
    messages.sort()
    return (0 if not messages else 1), messages


def main() -> int:
    parser = argparse.ArgumentParser(description="Check parquet determinism.")
    parser.add_argument("--strict", action="store_true", help="Fail on any bare to_parquet call.")
    parser.add_argument("--json", action="store_true", help="Machine-readable JSON output.")
    args = parser.parse_args()

    code, messages = check()
    if args.json:
        print(json.dumps({"ok": code == 0, "messages": messages}, indent=2))
    else:
        if code == 0:
            print("check-parquet-determinism: OK")
        else:
            for m in messages:
                print(f"FAIL {m}")
    return code if args.strict else 0


if __name__ == "__main__":
    sys.exit(main())