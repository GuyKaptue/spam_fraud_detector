"""Export A/B test results to MLflow.

Logs ab_test.json metrics to MLflow for tracking and comparison across runs.
Gracefully degrades if MLflow is not installed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def export_mlflow(domain: str = "spam_detection") -> bool:
    """Export ab_test.json to MLflow."""
    ab_path = Path("outputs/stats") / domain / "ab_test.json"
    if not ab_path.exists():
        # Try staging
        for p in Path("outputs").parent.glob("outputs.staging.*"):
            candidate = p / "stats" / domain / "ab_test.json"
            if candidate.exists():
                ab_path = candidate
                break
        else:
            print(f"export-mlflow: no ab_test.json found for domain '{domain}'")
            return False

    ab = json.loads(ab_path.read_text(encoding="utf-8"))

    try:
        import mlflow
    except ImportError:
        print("export-mlflow: mlflow not installed, skipping")
        return 0

    with mlflow.start_run(run_name=f"ab_test_{domain}"):
        mlflow.log_param("domain", domain)
        mlflow.log_param("n_comparisons", len(ab.get("comparisons", [])))
        config = ab.get("config", {})
        mlflow.log_param("method_default", config.get("method_default", ""))
        mlflow.log_param("family_size", config.get("family_size", 0))
        mlflow.log_param("alpha_family", config.get("alpha_family", 0.0))
        mlflow.log_param("min_discordant_chi2", config.get("min_discordant_chi2", 0))

        for c in ab.get("comparisons", []):
            run_name = f"{c.get('model_a', '?')} vs {c.get('model_b', '?')}"
            with mlflow.start_run(run_name=run_name, nested=True):
                mlflow.log_param("model_a", c.get("model_a", ""))
                mlflow.log_param("model_b", c.get("model_b", ""))
                mlflow.log_param("test", c.get("test", ""))
                mlflow.log_metric("statistic", c.get("statistic", 0.0))
                mlflow.log_metric("p_value", c.get("p_value", 0.0))
                mlflow.log_metric("p_holm_adjusted", c.get("p_holm_adjusted", 0.0))
                mlflow.log_metric("n", c.get("n", 0))
                mlflow.log_metric("n_discordant", c.get("n_discordant", 0))
                mlflow.log_metric("n01", c.get("n01", 0))
                mlflow.log_metric("n10", c.get("n10", 0))
                mlflow.log_metric("cohen_h", c.get("cohen_h", 0.0))
                mlflow.log_metric("risk_difference", c.get("risk_difference", 0.0))
                mlflow.log_param("winner", c.get("winner", ""))
                mlflow.log_param("family_rank", c.get("family_rank", 0))
                mlflow.log_param("significant_holm_0.05", c.get("significant_holm_0.05", False))
                mlflow.log_param("significant_holm_0.01", c.get("significant_holm_0.01", False))
                mlflow.log_param("winner_significant", c.get("winner_significant", False))

        print(f"export-mlflow: logged {len(ab.get('comparisons', []))} comparisons for domain '{domain}'")
        return True


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Export A/B test results to MLflow.")
    parser.add_argument("--domain", default="spam_detection", help="Domain to export.")
    args = parser.parse_args()

    return 0 if export_mlflow(args.domain) else 1


if __name__ == "__main__":
    sys.exit(main())