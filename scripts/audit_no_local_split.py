"""Audit: comparer.py has no local train_test_split and uses SplitManager."""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

SRC = Path("spam_fraud_detector/core/comparer.py")


def audit() -> tuple[int, list[str]]:
    msgs = []
    if not SRC.exists():
        return 1, [f"file not found: {SRC}"]
    src = SRC.read_text(encoding="utf-8")
    tree = ast.parse(src)

    # Check for train_test_split calls
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name == "train_test_split":
                msgs.append(f"train_test_split() found at line {node.lineno}")

    # Check SplitManager is imported
    if "SplitManager" not in src:
        msgs.append("SplitManager not imported in comparer.py")

    # Check load_split is called
    if "load_split" not in src:
        msgs.append("SplitManager.load_split() not called in comparer.py")

    return (0 if not msgs else 1), msgs


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit no local split.")
    parser.add_argument("--json", action="store_true", help="Machine-readable JSON output.")
    args = parser.parse_args()

    code, msgs = audit()
    if args.json:
        print(json.dumps({"ok": code == 0, "messages": msgs}, indent=2))
    else:
        if code == 0:
            print("audit_no_local_split: OK")
        else:
            for m in msgs:
                print(f"FAIL {m}")
    return code


if __name__ == "__main__":
    sys.exit(main())