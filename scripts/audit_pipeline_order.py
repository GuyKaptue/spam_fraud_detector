"""Audit: DataPipeline.run() executes stages in 01_split -> 07_features order."""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

SRC = Path("spam_fraud_detector/core/data_pipeline.py")
STAGE_ORDER = ["01_split", "02_cleaned", "03_modeled", "04_analyzed",
               "05_kpis", "06_quality", "07_features"]


def audit() -> tuple[int, list[str]]:
    if not SRC.exists():
        return 1, [f"file not found: {SRC}"]
    src = SRC.read_text(encoding="utf-8")
    tree = ast.parse(src)

    # Find DataPipeline.run method
    run_func = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "run":
            run_func = node
            break

    if run_func is None:
        return 1, ["DataPipeline.run() not found"]

    # Collect string literals in the function body
    literals = []
    for node in ast.walk(run_func):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            literals.append((node.lineno, node.value))

    # Check that 01_split appears first among stage names
    stage_mentions = [(ln, val) for ln, val in literals if val in STAGE_ORDER]
    if not stage_mentions:
        # Maybe stages are in a list variable
        for node in ast.walk(run_func):
            if isinstance(node, ast.List):
                vals = []
                for elt in node.elts:
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                        vals.append(elt.value)
                if any(v in STAGE_ORDER for v in vals):
                    stage_mentions.extend([(node.lineno, v) for v in vals if v in STAGE_ORDER])

    if not stage_mentions:
        return 1, ["no stage names found in DataPipeline.run()"]

    # Find the layers list
    for node in ast.walk(run_func):
        if isinstance(node, ast.List):
            vals = []
            for elt in node.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    vals.append(elt.value)
            if vals == STAGE_ORDER:
                # Found the exact stage order
                return (0, ["pipeline order: OK"])

    # If not found, check if at least the first stage is 01_split
    # and report what we found
    return (1, [f"stage order list not found in exact form; expected {STAGE_ORDER}"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit pipeline stage order.")
    parser.add_argument("--json", action="store_true", help="Machine-readable JSON output.")
    args = parser.parse_args()

    code, msgs = audit()
    if args.json:
        print(json.dumps({"ok": code == 0, "messages": msgs}, indent=2))
    else:
        if code == 0:
            print("audit_pipeline_order: OK")
        else:
            for m in msgs:
                print(f"FAIL {m}")
    return code


if __name__ == "__main__":
    sys.exit(main())