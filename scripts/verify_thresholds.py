"""Verify cost-sensitive threshold selection."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    errors = []
    outputs = Path("outputs")
    for meta_path in outputs.rglob("*.meta.json"):
        meta = json.loads(meta_path.read_text())
        tp = meta.get("threshold_policy")
        if tp == "expected_cost":
            thr_path = meta_path.parent / (meta_path.name.replace(".meta.json", ".thresholds.json"))
            if not thr_path.exists():
                errors.append(f"missing {thr_path}")
                continue
            thr = json.loads(thr_path.read_text())
            if abs(thr.get("threshold_star", 0) - meta.get("threshold", 0)) > 1e-9:
                errors.append(f"threshold mismatch {meta_path}")
            cm = thr.get("cost_matrix", {})
            cf, cp = cm.get("cost_fn"), cm.get("cost_fp")
            tp_ = thr.get("tp_at_star")
            fp_ = thr.get("fp_at_star")
            fn_ = thr.get("fn_at_star")
            tn_ = thr.get("tn_at_star")
            n = (tp_ or 0) + (fp_ or 0) + (fn_ or 0) + (tn_ or 0)
            if n > 0:
                ec = (cf * fn_ + cp * fp_) / n
                if abs(ec - thr.get("expected_cost_at_star", 0)) > 1e-9:
                    errors.append(f"expected_cost mismatch {meta_path}")
            sweep = thr.get("sweep", [])
            if sweep:
                min_ec = min(r.get("expected_cost", float("inf")) for r in sweep)
                star_ec = thr.get("expected_cost_at_star")
                if abs(min_ec - star_ec) > 1e-9:
                    errors.append(f"threshold_star not argmin {meta_path}")

    fts_path = outputs / "dashboard" / "fact_thresholds_at_star.parquet"
    if fts_path.exists():
        fts = pd.read_parquet(fts_path)
        if fts.empty:
            errors.append("fact_thresholds_at_star empty")
    else:
        errors.append("missing fact_thresholds_at_star.parquet")

    dtp_path = outputs / "dashboard" / "dim_threshold_policy.parquet"
    if dtp_path.exists():
        dtp = pd.read_parquet(dtp_path)
        if len(dtp) != 2:
            errors.append(f"dim_threshold_policy rows={len(dtp)}, expected 2")
    else:
        errors.append("missing dim_threshold_policy.parquet")

    if args.json:
        print(json.dumps({"ok": not errors, "errors": errors}, indent=2))
    else:
        if errors:
            for e in errors:
                print("FAIL", e)
            return 1 if args.strict else 0
        print("verify-thresholds: OK")
    return 0 if not errors else (1 if args.strict else 0)


if __name__ == "__main__":
    raise SystemExit(main())