"""Verify anomaly layer produced bundles."""
from __future__ import annotations

import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()
    errors = []
    outputs = Path("outputs")
    for domain in ["spam_detection", "fraud_detection"]:
        d = outputs / "models" / domain / "all"
        bundles = list(d.glob("*__anomaly.joblib")) if d.exists() else []
        if len(bundles) < 3:
            errors.append(f"{domain}: only {len(bundles)} anomaly bundles")
    dtp = outputs / "dashboard" / "dim_group.parquet"
    if dtp.exists():
        import pandas as pd
        df = pd.read_parquet(dtp)
        if "anomaly" not in df["group"].values if "group" in df.columns else True:
            errors.append("dim_group missing anomaly row")
    fts = outputs / "dashboard" / "fact_metrics.parquet"
    if fts.exists():
        import pandas as pd
        df = pd.read_parquet(fts)
        if "anomaly" not in df.get("group", pd.Series()).values:
            errors.append("fact_metrics missing anomaly rows")
    fail_log = outputs / "logs" / "anomaly_failures.log"
    if fail_log.exists() and fail_log.read_text().strip():
        errors.append("anomaly_failures.log non-empty")
    if errors:
        for e in errors:
            print("FAIL", e)
        return 1 if args.strict else 0
    print("verify-anomaly-layer: OK")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())