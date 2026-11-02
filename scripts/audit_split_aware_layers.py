"""Audit: every layer 02_*..07_* has split-aware outputs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DOMAINS = ("spam", "fraud")
LAYERS = ["02_cleaned", "03_modeled", "04_analyzed", "05_kpis", "06_quality", "07_features"]


def audit(domain: str = "both", strict: bool = False) -> tuple[int, list[str]]:
    msgs = []
    domains = ("spam", "fraud") if domain == "both" else (domain,)

    for d in domains:
        split_dir = Path("data/processed") / d / "01_split"
        if not (split_dir / "split.json").exists():
            # fresh clone before make data — skip
            continue
        for layer in LAYERS:
            layer_dir = Path("data/processed") / d / layer
            if not layer_dir.exists():
                if strict:
                    msgs.append(f"{d}/{layer}: missing layer directory")
                continue
            # check for split-aware outputs
            has_subfolders = all((layer_dir / s).exists() for s in ("train", "val", "test"))
            has_parquets = all((layer_dir / f"{s}.parquet").exists() for s in ("train", "val", "test"))
            if not has_subfolders and not has_parquets:
                msgs.append(f"{d}/{layer}: no split-aware outputs (need train/val/test subfolders or parquets)")

    return (0 if not msgs else 1), msgs


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit split-aware layers.")
    parser.add_argument("--domain", default="both", help="Domain: spam, fraud, or both.")
    parser.add_argument("--strict", action="store_true", help="Treat missing as failure.")
    parser.add_argument("--json", action="store_true", help="Machine-readable JSON output.")
    args = parser.parse_args()

    code, msgs = audit(domain=args.domain, strict=args.strict)
    if args.json:
        print(json.dumps({"ok": code == 0, "messages": msgs}, indent=2))
    else:
        if code == 0:
            print("audit_split_aware_layers: OK")
        else:
            for m in msgs:
                print(f"FAIL {m}")
    return code


if __name__ == "__main__":
    sys.exit(main())