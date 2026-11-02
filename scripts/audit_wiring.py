"""Audit: compare_anomalies() called in compare_groups(); record_explainability called."""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

COMPARER = Path("spam_fraud_detector/core/comparer.py")
MAIN = Path("spam_fraud_detector/main.py")
RESULT_STORE = Path("spam_fraud_detector/core/result_store.py")


def _has_call_in_function(src: str, func_name: str, call_name: str) -> bool:
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    name = None
                    if isinstance(child.func, ast.Name):
                        name = child.func.id
                    elif isinstance(child.func, ast.Attribute):
                        name = child.func.attr
                    if name == call_name:
                        return True
    return False


def audit() -> tuple[int, list[str]]:
    msgs = []
    # comparer.py
    if COMPARER.exists():
        src = COMPARER.read_text(encoding="utf-8")
        if "compare_anomalies" not in src:
            msgs.append("comparer.py: compare_anomalies() not defined")
        elif not _has_call_in_function(src, "compare_groups", "compare_anomalies"):
            msgs.append("comparer.py: compare_anomalies() defined but not called in compare_groups()")

    # result_store.py
    if RESULT_STORE.exists():
        src = RESULT_STORE.read_text(encoding="utf-8")
        if "def record_explainability" not in src:
            msgs.append("result_store.py: record_explainability() not defined")

    # main.py
    if MAIN.exists():
        src = MAIN.read_text(encoding="utf-8")
        if "record_explainability" not in src:
            msgs.append("main.py: record_explainability() not called")

    return (0 if not msgs else 1), msgs


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit wiring.")
    parser.add_argument("--json", action="store_true", help="Machine-readable JSON output.")
    args = parser.parse_args()

    code, msgs = audit()
    if args.json:
        print(json.dumps({"ok": code == 0, "messages": msgs}, indent=2))
    else:
        if code == 0:
            print("audit_wiring: OK")
        else:
            for m in msgs:
                print(f"FAIL {m}")
    return code


if __name__ == "__main__":
    sys.exit(main())