"""Collect every artifact under outputs/, reports/, mlruns/ into a single JSON inventory.

This inventory is the single source of truth for all documented claims. Every
number, figure, and statement in the documentation must cite a path from this
inventory. If a claim cites a path not present here, `make docs-verify` fails.

Inventory schema:
{
  "generated_at": ISO-8601 timestamp,
  "git_commit": git commit hash,
  "dataset_hash": {"spam": "sha256:...", "fraud": "sha256:..."},
  "models": {
    "spam_detection": {
      "<model_name>__<group>": {
        "bundle_path": "outputs/models/spam_detection/all/<name>__<group>.joblib",
        "meta_path": "outputs/models/spam_detection/all/<name>__<group>.meta.json",
        "eval": {"test": {...}},
        "threshold": 0.34,
        "threshold_policy": "expected_cost",
        "explainability": {"persona": "spam", "summary_path": "...", "casefiles": [...]}
      },
      ...
    },
    "fraud_detection": {...}
  },
  "stats": {
    "spam_detection": {
      "ab_test_path": "outputs/stats/spam_detection/ab_test.json",
      "comparisons": [...],
      "summary_md_path": "outputs/stats/spam_detection/summary.md"
    },
    "fraud_detection": {...}
  },
  "dashboard": {
    "fact_metrics": "outputs/dashboard/fact_metrics.parquet",
    "fact_confusion": "outputs/dashboard/fact_confusion.parquet",
    "fact_thresholds_at_star": "outputs/dashboard/fact_thresholds_at_star.parquet",
    "fact_stats": "outputs/dashboard/fact_stats.parquet",
    "dim_explainability": "outputs/dashboard/dim_explainability.parquet",
    "selection": "outputs/dashboard/selection.parquet"
  },
  "figures": {
    "spam_detection": ["reports/spam_detection/figures/f1_scores.png", ...],
    "fraud_detection": [...]
  },
  "explainability": {
    "spam_detection": {"casefiles": [...], "summaries": [...]},
    "fraud_detection": {...}
  },
  "mlflow": {
    "runs": [{"run_id": "...", "run_name": "...", "params": {...}, "metrics": {...}}]
  }
}
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List


def _hash_file(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_json(path: Path) -> dict | None:
    """Read a JSON file, returning None if it doesn't exist or is invalid."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.loads(f.read())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def collect_domain_models(domain: str, models_dir: Path) -> dict:
    """Collect all model bundles for a domain."""
    domain_dir = models_dir / domain / "all"
    if not domain_dir.exists():
        return {}

    models = {}
    for bundle_path in sorted(domain_dir.glob("*.joblib")):
        meta_path = bundle_path.with_suffix(".meta.json")
        meta = _read_json(meta_path)
        if meta is None:
            continue

        model_name = bundle_path.stem  # e.g., "XGBoost__unscaled"

        eval_path = bundle_path.with_suffix(".eval.json")
        eval_data = _read_json(eval_path)

        thresholds_path = bundle_path.with_suffix(".thresholds.json")
        thresholds_data = _read_json(thresholds_path)

        confusion_path = bundle_path.with_suffix(".confusion.json")
        confusion_data = _read_json(confusion_path)

        curve_path = bundle_path.with_suffix(".curve.json")
        curve_data = _read_json(curve_path)

        models[model_name] = {
            "bundle_path": str(bundle_path.relative_to(Path("outputs"))),
            "meta_path": str(meta_path.relative_to(Path("outputs"))),
            "thresholds_path": str(thresholds_path.relative_to(Path("outputs"))) if thresholds_path.exists() else None,
            "meta": meta,
            "eval": eval_data,
            "thresholds": thresholds_data,
            "confusion": confusion_data,
            "curve": curve_data,
            "artifact_bytes": meta.get("artifact_bytes") if meta else None,
            "artifact_sha256": meta.get("artifact_sha256") if meta else None,
            "threshold": meta.get("threshold") if meta else None,
            "threshold_policy": meta.get("threshold_policy") if meta else None,
            "feature_count": meta.get("feature_count") if meta else None,
            "feature_columns": meta.get("feature_columns") if meta else None,
            "group": meta.get("group") if meta else None,
            "model_name": meta.get("model_name") if meta else model_name,
        }

    return models


def collect_domain_stats(domain: str, outputs_root: Path) -> dict:
    """Collect stats artifacts for a domain."""
    stats_dir = outputs_root / "stats" / domain
    ab_test_path = stats_dir / "ab_test.json"
    summary_path = stats_dir / "summary.md"

    ab_test = _read_json(ab_test_path)
    summary_md = summary_path.read_text(encoding="utf-8") if summary_path.exists() else None

    return {
        "ab_test_path": str(ab_test_path.relative_to(outputs_root)) if ab_test_path.exists() else None,
        "summary_md_path": str(summary_path.relative_to(outputs_root)) if summary_path.exists() else None,
        "ab_test": ab_test,
        "summary_md": summary_md,
    }


def collect_domain_dashboard(domain: str, outputs_root: Path) -> dict:
    """Collect dashboard artifacts for a domain."""
    dashboard_dir = outputs_root / "dashboard"
    if not dashboard_dir.exists():
        return {}

    tables = {}
    for tname in ["fact_metrics", "fact_confusion", "fact_thresholds", "fact_thresholds_at_star",
                  "fact_stats", "fact_stats_summary", "fact_curves", "fact_predictions",
                   "dim_explainability", "dim_domain", "dim_group", "dim_model",
                   "dim_split", "dim_threshold_policy", "manifest", "selection"]:
        pq_path = dashboard_dir / f"{tname}.parquet"
        csv_path = dashboard_dir / f"{tname}.csv"
        if pq_path.exists():
            tables[tname] = {
                "parquet": str(pq_path.relative_to(outputs_root)),
                "csv": str(csv_path.relative_to(outputs_root)) if csv_path.exists() else None,
            }

    return tables


def collect_domain_explainability(domain: str, outputs_root: Path) -> dict:
    """Collect explainability artifacts for a domain."""
    exp_dir = outputs_root / "reports" / domain / "docs" / "explainability"
    if not exp_dir.exists():
        return {}

    casefiles = []
    hypos = []
    summaries = []

    for f in sorted(exp_dir.glob("*__case_*.json")):
        meta = _read_json(f)
        if meta:
            casefiles.append({
                "path": str(f.relative_to(outputs_root)),
                "meta": meta,
            })

    for f in sorted(exp_dir.glob("*__hypo_*.json")):
        meta = _read_json(f)
        if meta:
            hypos.append({
                "path": str(f.relative_to(outputs_root)),
                "meta": meta,
            })

    for f in sorted(exp_dir.glob("*__summary.png")):
        summaries.append(str(f.relative_to(outputs_root)))

    for f in sorted(exp_dir.glob("*__case_*.png")):
        pass  # Already covered by casefiles JSON

    return {
        "casefiles": casefiles,
        "hypotheses": hypos,
        "summaries": summaries,
    }


def collect_mlflow() -> dict:
    """Collect MLflow run information."""
    mlruns_dir = Path("mlruns")
    if not mlruns_dir.exists():
        return {"runs": []}

    runs = []
    for run_dir in sorted(mlruns_dir.glob("**/meta.yaml")):
        try:
            import yaml
            meta = yaml.safe_load(run_dir.read_text())
            runs.append({
                "run_id": run_dir.parent.name,
                "run_name": meta.get("run_name", run_dir.parent.name),
                "status": meta.get("status", "unknown"),
                "artifact_uri": meta.get("artifact_uri", ""),
                "start_time": meta.get("start_time"),
                "end_time": meta.get("end_time"),
                "params": meta.get("params", {}),
                "metrics": meta.get("metrics", {}),
            })
        except Exception:
            continue

    return {"runs": runs}


def collect_all() -> dict:
    """Collect all evidence into a single inventory."""
    outputs_root = Path("outputs")
    reports_root = Path("reports")

    inventory = {
        "schema_version": "1.0.0",
        "generated_at": "",
        "git_commit": "",
        "dataset_hash": {},
        "models": {},
        "stats": {},
        "dashboard": {},
        "figures": {},
        "explainability": {},
        "mlflow": {},
    }

    # Git commit
    import subprocess
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        inventory["git_commit"] = result.stdout.strip()
    except Exception:
        pass

    # Dataset hashes
    for domain in ["spam", "fraud"]:
        dataset_path = Path("data") / "processed" / domain / "01_split"
        if dataset_dir := dataset_path:
            for split_file in dataset_dir.glob("*.parquet"):
                h = hashlib.sha256()
                with open(split_file, "rb") as f:
                    for chunk in iter(lambda: f.read(8192), b""):
                        h.update(chunk)
                inventory["dataset_hash"][domain] = h.hexdigest()

    # Models
    for domain in ["spam_detection", "fraud_detection"]:
        inventory["models"][domain] = collect_domain_models(
            domain, outputs_root / "models"
        )

    # Stats
    for domain in ["spam_detection", "fraud_detection"]:
        inventory["stats"][domain] = collect_domain_stats(domain, outputs_root)

    # Dashboard
    inventory["dashboard"] = collect_domain_dashboard("spam_detection", outputs_root)
    inventory["dashboard"].update(collect_domain_dashboard("fraud_detection", outputs_root))

    # Explainability
    for domain in ["spam_detection", "fraud_detection"]:
        inventory["explainability"][domain] = collect_domain_explainability(domain, outputs_root)

    # MLflow
    inventory["mlflow"] = collect_mlflow()

    # Timestamp
    from datetime import datetime, timezone
    inventory["generated_at"] = datetime.now(timezone.utc).isoformat()

    return inventory


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Collect all artifacts into evidence inventory")
    parser.add_argument("--output", default="spam_fraud_detector/reports/docs/EVIDENCE_INVENTORY.json",
                        help="Output path for inventory JSON")
    args = parser.parse_args()

    inventory = collect_all()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(inventory, f, indent=2)

    print(f"Evidence inventory written to {output_path}")
    print(f"  Models: {sum(len(v) for v in inventory['models'].values())}")
    print(f"  Stats domains: {list(inventory['stats'].keys())}")
    print(f"  Dashboard tables: {len(inventory['dashboard'])}")
    print(f"  Explainability domains: {list(inventory['explainability'].keys())}")
    print(f"  MLflow runs: {len(inventory['mlflow']['runs'])}")


if __name__ == "__main__":
    main()