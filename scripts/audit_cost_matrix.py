"""Audit cost matrix wiring."""
from __future__ import annotations
import json
import sys
from pathlib import Path

def main():
    msgs = []
    for domain in ["spam", "fraud"]:
        p = Path(f"data/processed/{domain}/05_kpis/kpis.json")
        if not p.exists():
            msgs.append(f"missing {p}")
            continue
        d = json.loads(p.read_text())
        cm = d.get("cost_matrix", {})
        if "cost_fn" not in cm or "cost_fp" not in cm:
            msgs.append(f"{p}: cost_matrix missing cost_fn/cost_fp")
        if not (isinstance(cm.get("cost_fn"), int) and isinstance(cm.get("cost_fp"), int)):
            msgs.append(f"{p}: cost_fn/cost_fp not int")
        if cm.get("cost_fn", 0) <= 0 or cm.get("cost_fp", 0) <= 0:
            msgs.append(f"{p}: non-positive cost")
    # DataPipeline writes kpis.json
    dp = Path("spam_fraud_detector/core/data_pipeline.py").read_text()
    if "kpis.json" not in dp:
        msgs.append("data_pipeline: kpis.json not written")
    if msgs:
        for m in msgs:
            print("FAIL", m)
        sys.exit(1)
    print("audit_cost_matrix: OK")

if __name__ == "__main__":
    main()