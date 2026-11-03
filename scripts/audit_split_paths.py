"""Audit: SplitManager writes to 01_split/, no stale paths."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC_DIR = Path("spam_fraud_detector")
CORE_DIR = SRC_DIR / "core"


def audit() -> tuple[int, list[str]]:
    msgs = []
    # Check source files for stale path references
    stale_patterns = ['"splits/split.json"', '"splits"', "06_features/{train", "06_features/train.parquet"]
    for py_file in sorted(CORE_DIR.rglob("*.py")):
        try:
            lines = py_file.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            continue
        for lineno, line in enumerate(lines, 1):
            for pat in stale_patterns:
                if pat in line:
                    # skip docstrings/comments
                    stripped = line.strip()
                    if stripped.startswith(("#", '"""', "'")):
                        continue
                    msgs.append(f"{py_file}:{lineno}: stale path reference '{pat}'")

    # Check for stale directories on disk
    for domain in ("spam", "fraud"):
        stale_split = Path("data/processed") / domain / "splits"
        if stale_split.exists():
            msgs.append(f"stale directory exists: {stale_split}")
        stale_features = Path("data/processed") / domain / "06_features"
        for split_name in ("train", "val", "test"):
            stale_parquet = stale_features / f"{split_name}.parquet"
            if stale_parquet.exists():
                msgs.append(f"stale split parquet: {stale_parquet}")

    return (0 if not msgs else 1), msgs


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit split paths.")
    parser.add_argument("--json", action="store_true", help="Machine-readable JSON output.")
    args = parser.parse_args()

    code, msgs = audit()
    if args.json:
        print(json.dumps({"ok": code == 0, "messages": msgs}, indent=2))
    else:
        if code == 0:
            print("audit_split_paths: OK")
        else:
            for m in msgs:
                print(f"FAIL {m}")
    return code


if __name__ == "__main__":
    sys.exit(main())
