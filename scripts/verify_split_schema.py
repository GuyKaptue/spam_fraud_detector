"""Verify split schema: train/val/test share columns and dtypes."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify split schema.")
    parser.add_argument("--domain", default="spam", help="Domain (spam or fraud).")
    parser.add_argument("--data-root", default="data", help="Data root.")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as errors.")
    parser.add_argument("--json", action="store_true", help="Machine-readable JSON output.")
    args = parser.parse_args()

    domain = args.domain
    data_root = Path(args.data_root)
    split_dir = data_root / "processed" / domain / "01_split"

    errors = []
    warnings = []

    # check split parquets exist
    frames = {}
    for split_name in ("train", "val", "test"):
        path = split_dir / f"{split_name}.parquet"
        if not path.exists():
            errors.append(f"missing split file: {path}")
            continue
        frames[split_name] = pd.read_parquet(path)

    if errors:
        if args.json:
            print(json.dumps({"ok": False, "errors": errors, "warnings": warnings}, indent=2))
        else:
            for e in errors:
                print(f"FAIL {e}")
        return 1

    # check _row_id present
    for split_name, df in frames.items():
        if "_row_id" not in df.columns:
            errors.append(f"{split_name}: missing _row_id column")

    # check same columns
    col_sets = {s: set(df.columns) for s, df in frames.items()}
    all_cols = set()
    for s, cols in col_sets.items():
        all_cols |= cols
    for s, cols in col_sets.items():
        if cols != all_cols:
            missing = all_cols - cols
            extra = cols - all_cols
            if missing:
                errors.append(f"{s}: missing columns {missing}")
            if extra:
                errors.append(f"{s}: extra columns {extra}")

    # check same dtypes
    for col in all_cols:
        dtypes = {s: str(df[col].dtype) for s, df in frames.items() if col in df.columns}
        unique_dtypes = set(dtypes.values())
        if len(unique_dtypes) > 1:
            errors.append(f"column '{col}' dtype mismatch: {dtypes}")

    # check non-empty
    for split_name, df in frames.items():
        if len(df) == 0:
            errors.append(f"{split_name}: empty split")

    # check split.json
    split_json_path = split_dir / "split.json"
    if not split_json_path.exists():
        errors.append(f"missing split.json: {split_json_path}")
    else:
        try:
            manifest = json.loads(split_json_path.read_text())
            if "row_id_hashes" not in manifest:
                warnings.append("split.json missing row_id_hashes (deprecated row_index_hashes may be present)")
            if manifest.get("data_layout_version") != "v3.8":
                warnings.append(f"unexpected data_layout_version: {manifest.get('data_layout_version')}")
        except json.JSONDecodeError as e:
            errors.append(f"invalid split.json: {e}")

    if args.strict and warnings:
        errors.extend(warnings)
        warnings = []

    if args.json:
        print(json.dumps({"ok": len(errors) == 0, "errors": errors, "warnings": warnings}, indent=2))
    else:
        if errors:
            for e in errors:
                print(f"FAIL {e}")
        elif warnings:
            for w in warnings:
                print(f"WARN {w}")
        else:
            print("verify-split-schema: OK")

    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())