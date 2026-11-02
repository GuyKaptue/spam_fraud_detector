#!/usr/bin/env python3
"""Guard: verify anomaly explainability produced zero failures."""
import sys
import pathlib

p = pathlib.Path("reports/fraud_detection/docs/explainability")
if not p.exists():
    print("no anomaly explainability dir")
    sys.exit(1)

files = list(p.glob("*__anomaly__*.json"))
failed = [f for f in files if "FAILED" in f.name]
if failed:
    print(f"anomaly explain failed for: {[f.name for f in failed]}")
    sys.exit(1)
print(f"anomaly explain OK: {len(files)} case files")
