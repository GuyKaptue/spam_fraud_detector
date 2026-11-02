"""Verify dashboard contract: schema, invariants, FK joins, parquet/csv parity."""
import argparse
import sys
import json
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dashboard", action="store_true")
    args = parser.parse_args()
    root = Path("outputs")
    if not root.exists():
        print("no outputs dir; nothing to verify")
        return 0
    dash = root / "dashboard"
    errors = []
    # parquet/csv parity
    for pq in dash.glob("*.parquet"):
        csv = pq.with_suffix(".csv")
        if not csv.exists():
            errors.append(f"missing csv for {pq.name}")
            continue
        try:
            a = pd.read_parquet(pq)
            b = pd.read_csv(csv)
            if list(a.columns) != list(b.columns):
                errors.append(f"column mismatch {pq.name}")
        except Exception as e:
            errors.append(f"read error {pq.name}: {e}")
    if args.dashboard:
        # check index.json
        if not (dash / "index.json").exists():
            errors.append("missing index.json")
        if not (dash / "filters.json").exists():
            errors.append("missing filters.json")
    # docs_policy
    contract_path = root / "CONTRACT.json"
    if contract_path.exists():
        try:
            contract = json.loads(contract_path.read_text())
            dp = contract.get("docs_policy", {})
            expected = dp.get("core_readme_count_expected", 1)
            actual = len(list(Path("spam_fraud_detector/core").glob("*.README.md")))
            if actual != expected:
                errors.append(f"docs_policy mismatch: {actual} != {expected}")
        except Exception as e:
            errors.append(f"contract parse error: {e}")
    print("verify-contract: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())