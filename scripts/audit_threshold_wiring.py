"""Audit threshold wiring."""
from __future__ import annotations

import sys
from pathlib import Path

EXPLAINER = Path("spam_fraud_detector/core/result_store.py")
SELECTOR = Path("spam_fraud_detector/core/selector.py")

def main():
    msgs = []
    src = EXPLAINER.read_text()
    if "threshold_policy" not in src:
        msgs.append("result_store: threshold_policy missing")
    if "fact_thresholds_at_star" not in src:
        msgs.append("result_store: fact_thresholds_at_star missing")
    if "dim_threshold_policy" not in src:
        msgs.append("result_store: dim_threshold_policy missing")
    ssrc = SELECTOR.read_text()
    if "selection_policy" not in ssrc:
        msgs.append("selector: selection_policy missing")
    if msgs:
        for m in msgs:
            print("FAIL", m)
        sys.exit(1)
    print("audit_threshold_wiring: OK")

if __name__ == "__main__":
    main()