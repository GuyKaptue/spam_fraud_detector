"""Single-writer persistence: bundles + dashboard tables in one atomic transaction."""
from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import pyarrow  # noqa: F401

from .exceptions import ContractViolation
from .utils import atomic_write_json, atomic_write_text, hash_file, get_git_commit, get_package_versions

SCHEMA_VERSION = "1.0.0"
KNOWN_METRICS = [
    "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
    "recall_at_prec_0.90", "balanced_accuracy", "mcc", "brier", "log_loss",
]
COUNT_KEYS = {"tp", "fp", "tn", "fn", "n", "n_positive"}
CURVE_MAX_POINTS = 2001


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _downsample(xs, ys, max_points=CURVE_MAX_POINTS):
    xs = list(xs)
    ys = list(ys)
    if len(xs) <= max_points:
        return xs, ys
    idx = np.linspace(0, len(xs) - 1, max_points, dtype=int)
    return [xs[i] for i in idx], [ys[i] for i in idx]


@dataclass
class RunContext:
    domain: str
    git_commit: str = ""
    dataset_hash: str = ""
    seed: int = 42
    python_version: str = ""
    package_versions: dict = field(default_factory=dict)
    framework_version: str = "0.3.0"
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self):
        if not self.git_commit:
            self.git_commit = get_git_commit()
        if not self.python_version:
            import sys
            self.python_version = sys.version.split()[0]
        if not self.package_versions:
            self.package_versions = get_package_versions()


class ResultStore:
    def __init__(self, out_root, ctx: RunContext) -> None:
        self.out_root = Path(out_root) if not isinstance(out_root, Path) else out_root
        self.ctx = ctx
        self.staging = self.out_root.parent / ("outputs.staging." + uuid.uuid4().hex)
        self.staging.mkdir(parents=True, exist_ok=True)
        self.created_at = _utc_now()
        self._metrics = []
        self._confusion = []
        self._thresholds = []
        self._thresholds_at_star = []
        self._curves = []
        self._manifest = []
        self._skipped = []
        self._selection = None
        self._stats = None
        self._stats_summary_md = ""
        self.domain = ctx.domain
        (self.staging / "models" / ctx.domain / "all").mkdir(parents=True, exist_ok=True)
        (self.staging / "models" / ctx.domain / "top3").mkdir(parents=True, exist_ok=True)
        (self.staging / "predictions" / ctx.domain / "all").mkdir(parents=True, exist_ok=True)
        (self.staging / "predictions" / ctx.domain / "top3").mkdir(parents=True, exist_ok=True)
        (self.staging / "stats" / ctx.domain).mkdir(parents=True, exist_ok=True)
        (self.staging / "dashboard" / "fact_predictions").mkdir(parents=True, exist_ok=True)
        (self.staging / "runs").mkdir(parents=True, exist_ok=True)

    def _bundle_dir(self):
        return self.staging / "models" / self.ctx.domain / "all"

    def _base(self):
        return {"schema_version": self.ctx.schema_version, "git_commit": self.ctx.git_commit,
                "dataset_hash": self.ctx.dataset_hash, "created_at": self.created_at}

    def _threshold_sweep(self, y_true, y_score, cost_fn=1, cost_fp=1):
        y_true = np.asarray(y_true).astype(int)
        y_score = np.asarray(y_score, dtype=float)
        n = len(y_true)
        rows = []
        for i in range(101):
            t = round(i / 100.0, 2)
            pred = (y_score >= t).astype(int)
            tp = int(((pred == 1) & (y_true == 1)).sum())
            fp = int(((pred == 1) & (y_true == 0)).sum())
            tn = int(((pred == 0) & (y_true == 0)).sum())
            fn = int(((pred == 0) & (y_true == 1)).sum())
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            tpr = rec
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            expected_cost = (cost_fn * fn + cost_fp * fp) / n if n > 0 else 0.0
            expected_cost_cfn_10_cfp_1 = int(10 * fn + 1 * fp)
            rows.append({
                "threshold": t, "precision": float(prec), "recall": float(rec),
                "f1": float(f1), "tpr": float(tpr), "fpr": float(fpr),
                "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                "cost_fn": cost_fn, "cost_fp": cost_fp,
                "expected_cost": float(expected_cost),
                "expected_cost_cfn_10_cfp_1": expected_cost_cfn_10_cfp_1,
            })
        star = min(rows, key=lambda r: r["expected_cost"])["threshold"]
        return rows, star

    def _compute_curves(self, y_true, y_score):
        from sklearn.metrics import roc_curve, precision_recall_curve
        from sklearn.calibration import calibration_curve
        y_true = np.asarray(y_true).astype(int)
        y_score = np.asarray(y_score, dtype=float)
        fpr, tpr, _ = roc_curve(y_true, y_score)
        prec, rec, _ = precision_recall_curve(y_true, y_score)
        try:
            prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=10, strategy="quantile")
            cal = {"prob_pred": [float(x) for x in prob_pred],
                   "prob_true": [float(x) for x in prob_true]}
        except Exception:
            cal = {"prob_pred": [], "prob_true": []}
        fpr_d, tpr_d = _downsample(fpr, tpr)
        prec_d, rec_d = _downsample(prec, rec)
        return {"roc": {"fpr": [float(x) for x in fpr_d], "tpr": [float(x) for x in tpr_d]},
                "pr": {"precision": [float(x) for x in prec_d], "recall": [float(x) for x in rec_d]},
                "calibration": cal}

    # ------------------------------------------------------------------ #
    # Bundle persistence
    # ------------------------------------------------------------------ #

    def write_model(
        self,
        name: str,
        group: str,
        model: Any = None,
        hyperparams: Optional[dict] = None,
        eval_val: Optional[dict] = None,
        eval_test: Optional[dict] = None,
        y_true_val: Optional[Any] = None,
        y_score_val: Optional[Any] = None,
        y_true_test: Optional[Any] = None,
        y_score_test: Optional[Any] = None,
        row_ids_test: Optional[list] = None,
        threshold: Optional[float] = None,
        threshold_policy: Optional[str] = None,
        confusion: Optional[dict] = None,
        curves: Optional[dict] = None,
        feature_count: Optional[int] = None,
        train_rows: Optional[int] = None,
        test_rows: Optional[int] = None,
        feature_columns: Optional[list] = None,
    ):
        """Persist one fitted model bundle plus its metrics/curves/thresholds."""
        domain = self.ctx.domain
        stem = f"{name}__{group}"
        bundle_dir = self._bundle_dir()
        bundle_dir.mkdir(parents=True, exist_ok=True)

        # --- serialize the model ---
        model_path = bundle_dir / f"{stem}.joblib"
        joblib.dump(model, model_path)
        artifact_bytes = model_path.stat().st_size
        artifact_sha256 = hash_file(model_path)

        # --- threshold resolution ---
        was_auto = threshold_policy is None or str(threshold_policy).strip() == ""
        policy = (threshold_policy or "auto").strip()
        if policy.startswith("fixed:"):
            try:
                threshold = float(policy.split(":", 1)[1])
            except ValueError:
                pass
            policy = "fixed"
        elif policy == "auto":
            policy = "expected_cost"

        # cost matrix (defaults; may be overridden by kpis.json)
        cost_fn, cost_fp = 1, 1

        threshold_computed = False
        if threshold is None and y_true_val is not None and y_score_val is not None:
            kpis_path = Path("data") / "processed" / domain / "05_kpis" / "kpis.json"
            try:
                kpis = json.loads(kpis_path.read_text())
                cost_fn = int(kpis.get("cost_matrix", {}).get("cost_fn", 1))
                cost_fp = int(kpis.get("cost_matrix", {}).get("cost_fp", 1))
                if policy == "expected_cost" and kpis.get("threshold_policy"):
                    policy = kpis["threshold_policy"]
            except Exception:
                pass
            rows, star = self._threshold_sweep(y_true_val, y_score_val, cost_fn=cost_fn, cost_fp=cost_fp)
            threshold = float(star)
            threshold_computed = True

        if threshold_computed:
            pass  # policy stays as resolved (expected_cost)
        elif threshold is not None:
            threshold = float(threshold)
            if was_auto:
                policy = "fixed"
        else:
            threshold = 0.5
            policy = "fixed"

        # --- eval metrics blocks ---
        def _safe_eval(e):
            e = dict(e or {})
            return {
                k: v for k, v in e.items()
                if k in COUNT_KEYS or k in KNOWN_METRICS
            }

        eval_block = {
            "domain": domain,
            "group": group,
            "model_name": name,
            "schema_version": SCHEMA_VERSION,
            "validation": _safe_eval(eval_val) if eval_val else {},
            "test": _safe_eval(eval_test) if eval_test else {},
        }

        # --- confusion block ---
        if confusion is None:
            confusion = {}
        conf_block = {
            "domain": domain,
            "group": group,
            "model_name": name,
            "schema_version": SCHEMA_VERSION,
            "threshold": threshold,
            "validation": confusion.get("validation", {}),
            "test": confusion.get("test", {}),
        }

        # --- curves block ---
        if curves is None:
            curves = {}
            if y_true_val is not None and y_score_val is not None:
                curves["validation"] = self._compute_curves(y_true_val, y_score_val)
            thresholds_block = None
            if y_true_test is not None and y_score_test is not None:
                curves["test"] = self._compute_curves(y_true_test, y_score_test)
        curve_block = {
            "domain": domain,
            "group": group,
            "model_name": name,
            "schema_version": SCHEMA_VERSION,
            "validation": curves.get("validation", {}),
            "test": curves.get("test", {}),
        }

        # --- thresholds sweep + star ---
        sweep_rows = []
        threshold_star = threshold
        expected_cost_at_star = None
        tp_at_star = fp_at_star = fn_at_star = tn_at_star = None
        if y_true_val is not None and y_score_val is not None:
            rows, star = self._threshold_sweep(y_true_val, y_score_val,
                                               cost_fn=cost_fn, cost_fp=cost_fp)
            threshold_star = float(star)
            sweep_rows = rows
            star_row = min(rows, key=lambda r: r["expected_cost"])
            expected_cost_at_star = star_row["expected_cost"]
            tp_at_star = star_row["tp"]
            fp_at_star = star_row["fp"]
            fn_at_star = star_row["fn"]
            tn_at_star = star_row["tn"]
        elif y_true_test is not None and y_score_test is not None:
            rows, star = self._threshold_sweep(y_true_test, y_score_test,
                                               cost_fn=cost_fn, cost_fp=cost_fp)
            threshold_star = float(star)
            sweep_rows = rows
            star_row = min(rows, key=lambda r: r["expected_cost"])
            expected_cost_at_star = star_row["expected_cost"]
            tp_at_star = star_row["tp"]
            fp_at_star = star_row["fp"]
            fn_at_star = star_row["fn"]
            tn_at_star = star_row["tn"]

        thresholds_block = {
            "schema_version": SCHEMA_VERSION,
            "domain": domain,
            "group": group,
            "model_name": name,
            "cost_matrix": {"cost_fn": cost_fn, "cost_fp": cost_fp},
            "threshold": threshold,
            "threshold_star": threshold_star,
            "expected_cost_at_star": expected_cost_at_star,
            "sweep": sweep_rows,
        }

        # --- predictions parquet ---
        if y_true_test is not None and y_score_test is not None:
            preds = self._predictions_df(
                y_true_test, y_score_test, threshold, "test", fold=-1,
                row_ids=row_ids_test,
            )
            pred_path = bundle_dir / f"{stem}.parquet"
            from spam_fraud_detector.core.utils import deterministic_parquet
            deterministic_parquet(preds, pred_path)
            preds_dash_dir = self.staging / "dashboard" / "fact_predictions"
            preds_dash_dir.mkdir(parents=True, exist_ok=True)
            deterministic_parquet(preds, preds_dash_dir / f"{domain}__{stem}.parquet")

        # --- write bundle files ---
        atomic_write_json(model_path.with_suffix(".meta.json"), self._meta_block(
            name, group, hyperparams, threshold, policy, artifact_bytes,
            artifact_sha256, feature_count, train_rows, test_rows, feature_columns))
        atomic_write_json(model_path.with_suffix(".eval.json"), eval_block)
        atomic_write_json(model_path.with_suffix(".curve.json"), curve_block)
        atomic_write_json(model_path.with_suffix(".confusion.json"), conf_block)
        atomic_write_json(model_path.with_suffix(".thresholds.json"), thresholds_block or {})

        # --- accumulate dashboard rows ---
        self._metrics.extend(self._metrics_rows(name, group, eval_val, eval_test, threshold))
        self._curves.extend(self._curve_rows(name, group, curve_block))
        self._thresholds.extend(self._threshold_rows(name, group, sweep_rows, threshold_star,
                                                     cost_fn, cost_fp))
        self._thresholds_at_star.append(self._threshold_at_star_row(
            name, group, threshold_star, tp_at_star, fp_at_star, fn_at_star, tn_at_star,
            expected_cost_at_star, cost_fn, cost_fp))
        self._confusion.extend(self._confusion_rows(name, group, conf_block, threshold))

        entry = {
            "schema_version": SCHEMA_VERSION,
            "git_commit": self.ctx.git_commit,
            "dataset_hash": self.ctx.dataset_hash,
            "created_at": self.created_at,
            "domain": domain,
            "model": name,
            "group": group,
            "skipped": False,
            "skip_reason": None,
            "artifact": str(Path("models") / domain / "all" / f"{stem}.joblib"),
            "meta": str(Path("models") / domain / "all" / f"{stem}.meta.json"),
            "eval": str(Path("models") / domain / "all" / f"{stem}.eval.json"),
            "curve": str(Path("models") / domain / "all" / f"{stem}.curve.json"),
            "confusion": str(Path("models") / domain / "all" / f"{stem}.confusion.json"),
            "artifact_sha256": artifact_sha256,
            "artifact_bytes": int(artifact_bytes),
            "split_manifest_hash": "",
            "feature_pipeline_hash": "",
            "quality_gate_hash": "",
            "test_access_count": 0,
            "explainability": None,
        }
        self._manifest.append(entry)
        return entry

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _predictions_df(self, y_true, y_score, threshold, split, fold, row_ids=None):
        y_true = np.asarray(y_true).ravel().astype(int)
        y_score = np.asarray(y_score, dtype=float).ravel()
        y_pred = (y_score >= threshold).astype(int)
        n = len(y_true)
        if row_ids is None:
            row_ids = np.arange(n, dtype=int)
        return pd.DataFrame({
            "row_id": row_ids,
            "y_true": y_true,
            "y_score": y_score,
            "y_pred": y_pred,
            "split": [split] * n,
            "fold": [fold] * n,
        })

    def _meta_block(self, name, group, hyperparams, threshold, policy,
                    artifact_bytes, artifact_sha256, feature_count, train_rows, test_rows,
                    feature_columns=None):
        return {
            "schema_version": SCHEMA_VERSION,
            "git_commit": self.ctx.git_commit,
            "dataset_hash": self.ctx.dataset_hash,
            "created_at": self.created_at,
            "domain": self.domain,
            "model_name": name,
            "group": group,
            "hyperparameters": hyperparams or {},
            "framework_version": self.ctx.framework_version,
            "package_versions": self.ctx.package_versions,
            "python_version": self.ctx.python_version,
            "random_seed": self.ctx.seed,
            "threshold": float(threshold),
            "threshold_policy": policy,
            "feature_count": feature_count,
            "train_rows": train_rows,
            "test_rows": test_rows,
            "feature_columns": list(feature_columns) if feature_columns is not None else None,
            "artifact_bytes": int(artifact_bytes),
            "artifact_sha256": artifact_sha256,
            "explainability": None,
        }

    def _metrics_rows(self, name, group, eval_val, eval_test, threshold):
        rows = []
        for split, ev in (("validation", eval_val), ("test", eval_test)):
            if not ev:
                continue
            for metric, value in ev.items():
                if metric not in KNOWN_METRICS:
                    continue
                rows.append({
                    "schema_version": SCHEMA_VERSION,
                    "git_commit": self.ctx.git_commit,
                    "dataset_hash": self.ctx.dataset_hash,
                    "created_at": self.created_at,
                    "domain": self.domain,
                    "model": name,
                    "group": group,
                    "split": split,
                    "fold": -1,
                    "metric": metric,
                    "value": float(value) if isinstance(value, (int, float)) else value,
                    "n": ev.get("n"),
                    "n_positive": ev.get("n_positive"),
                    "threshold": float(threshold),
                })
        return rows

    def _curve_rows(self, name, group, curve_block):
        rows = []
        for split, curves in (("validation", curve_block.get("validation", {})),
                              ("test", curve_block.get("test", {}))):
            for ctype in ("roc", "pr", "calibration"):
                data = curves.get(ctype, {})
                xs = data.get("fpr" if ctype in ("roc",) else "prob_pred" if ctype == "calibration" else "precision", [])
                ys = data.get("tpr" if ctype in ("roc",) else "prob_true" if ctype == "calibration" else "recall", [])
                if not xs:
                    xs = data.get("fpr", [])
                    ys = data.get("tpr", [])
                if not xs:
                    continue
                xs = [float(x) for x in xs]
                ys = [float(y) for y in ys]
                for pi, (x, y) in enumerate(zip(xs, ys)):
                    rows.append({
                        "schema_version": SCHEMA_VERSION,
                        "git_commit": self.ctx.git_commit,
                        "dataset_hash": self.ctx.dataset_hash,
                        "created_at": self.created_at,
                        "domain": self.domain,
                        "model": name,
                        "group": group,
                        "split": split,
                        "curve_type": ctype,
                        "x": x,
                        "y": y,
                        "point_index": pi,
                    })
        return rows

    def _threshold_rows(self, name, group, rows, threshold_star, cost_fn, cost_fp):
        out = []
        for r in rows:
            out.append({
                "schema_version": SCHEMA_VERSION,
                "git_commit": self.ctx.git_commit,
                "dataset_hash": self.ctx.dataset_hash,
                "created_at": self.created_at,
                "domain": self.domain,
                "model": name,
                "group": group,
                "split": "validation",
                "threshold": r["threshold"],
                "precision": r["precision"],
                "recall": r["recall"],
                "f1": r["f1"],
                "tpr": r["tpr"],
                "fpr": r["fpr"],
                "tp": r["tp"], "fp": r["fp"], "tn": r["tn"], "fn": r["fn"],
                "cost_fp": r["cost_fp"], "cost_fn": r["cost_fn"],
                "expected_cost": r["expected_cost"],
                "threshold_star": threshold_star,
                "is_at_star": r["threshold"] == threshold_star,
                "expected_cost_cfn_10_cfp_1": r["expected_cost_cfn_10_cfp_1"],
            })
        return out

    def _threshold_at_star_row(self, name, group, threshold_star, tp, fp, fn, tn,
                               expected_cost_at_star, cost_fn, cost_fp):
        return {
            "schema_version": SCHEMA_VERSION,
            "git_commit": self.ctx.git_commit,
            "dataset_hash": self.ctx.dataset_hash,
            "created_at": self.created_at,
            "domain": self.domain,
            "model": name,
            "group": group,
            "threshold_star": threshold_star,
            "tp_at_star": tp,
            "fp_at_star": fp,
            "fn_at_star": fn,
            "tn_at_star": tn,
            "expected_cost_at_star": expected_cost_at_star,
            "cost_fn": cost_fn,
            "cost_fp": cost_fp,
        }

    def _confusion_rows(self, name, group, conf_block, threshold):
        rows = []
        for split, conf in (("validation", conf_block.get("validation", {})),
                            ("test", conf_block.get("test", {}))):
            rows.append({
                "schema_version": SCHEMA_VERSION,
                "git_commit": self.ctx.git_commit,
                "dataset_hash": self.ctx.dataset_hash,
                "created_at": self.created_at,
                "domain": self.domain,
                "model": name,
                "group": group,
                "split": split,
                "threshold": threshold,
                "tp": int(conf.get("tp", 0)),
                "fp": int(conf.get("fp", 0)),
                "tn": int(conf.get("tn", 0)),
                "fn": int(conf.get("fn", 0)),
            })
        return rows

    # ------------------------------------------------------------------ #
    # Skipped / selection / stats / explainability
    # ------------------------------------------------------------------ #

    def write_skipped(self, items: List[dict]) -> None:
        for item in items:
            domain = self.ctx.domain
            self._manifest.append({
                "schema_version": SCHEMA_VERSION,
                "git_commit": self.ctx.git_commit,
                "dataset_hash": self.ctx.dataset_hash,
                "created_at": self.created_at,
                "domain": domain,
                "model": item.get("model"),
                "group": item.get("group"),
                "skipped": True,
                "skip_reason": item.get("reason"),
                "artifact": None,
                "meta": None,
                "eval": None,
                "curve": None,
                "confusion": None,
                "artifact_sha256": None,
                "artifact_bytes": None,
                "split_manifest_hash": "",
                "feature_pipeline_hash": "",
                "quality_gate_hash": "",
                "test_access_count": 0,
                "explainability": None,
            })

    def write_selection(self, ranking: List[dict]) -> None:
        self._selection = ranking

    def write_stats(self, ab: dict, summary_md: str) -> None:
        self._stats = ab
        self._stats_summary_md = summary_md

    def record_explainability(self, name: str, group: str, paths: dict,
                              persona: str = "fraud") -> None:
        """Record an explainability block into the manifest entry for (name, group)."""
        block = dict(paths)
        block["persona"] = persona
        found = False
        for m in self._manifest:
            if m.get("model") == name and m.get("group") == group:
                m["explainability"] = block
                found = True
        # persist to the on-disk meta.json (staging copy)
        domain = self.ctx.domain
        meta_path = self.staging / "models" / domain / "all" / f"{name}__{group}.meta.json"
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["explainability"] = block
            atomic_write_json(meta_path, meta)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            pass
        if not found:
            raise ContractViolation(
                f"model {name}__{group} not in manifest; call write_model before record_explainability"
            )

    def record_explainability_direct(self, name: str, group: str, paths: dict,
                                     persona: str = "fraud",
                                     casefiles: list | None = None) -> None:
        """Persist explainability into an already-finalized bundle's meta.json on disk."""
        import sys as _sys
        print(f"[DEBUG record_explainability_direct] name={name}, group={group}, persona={persona}, casefiles={len(casefiles) if casefiles else 0}", file=_sys.stderr)
        domain = self.ctx.domain
        meta_path = self.out_root / "models" / domain / "all" / f"{name}__{group}.meta.json"
        if not meta_path.exists():
            raise ContractViolation(
                f"no such bundle meta.json: {meta_path}"
            )
        block = dict(paths)
        block["persona"] = persona
        if casefiles:
            block["casefiles"] = casefiles
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["explainability"] = block
        atomic_write_json(meta_path, meta)

    # ------------------------------------------------------------------ #
    # Dashboard tables
    # ------------------------------------------------------------------ #
    # Dashboard tables
    # ------------------------------------------------------------------ #

    def _dim_domain(self):
        return pd.DataFrame([
            {"domain": "spam_detection", "display_name": "Spam Detection", "ordinal": 0},
            {"domain": "fraud_detection", "display_name": "Fraud Detection", "ordinal": 1},
        ])

    def _dim_group(self):
        return pd.DataFrame([
            {"group": "scaled", "display_name": "Scaled", "ordinal": 0},
            {"group": "unscaled", "display_name": "Unscaled", "ordinal": 1},
            {"group": "anomaly", "display_name": "Anomaly", "ordinal": 2},
        ])

    def _dim_split(self):
        return pd.DataFrame([
            {"split": "validation", "display_name": "Validation", "ordinal": 0},
            {"split": "test", "display_name": "Test", "ordinal": 1},
        ])

    def _dim_threshold_policy(self):
        return pd.DataFrame([
            {"policy": "expected_cost", "display_name": "Expected Cost (argmin)", "ordinal": 0},
            {"policy": "fixed", "display_name": "Fixed", "ordinal": 1},
        ])

    def _dim_explainability(self):
        return pd.DataFrame([
            {"persona": "fraud", "display_name": "Fraud Analyst", "ordinal": 0},
            {"persona": "spam", "display_name": "Trust & Safety Reviewer", "ordinal": 1},
            {"persona": "anomaly", "display_name": "Risk Engineer", "ordinal": 2},
        ])

    def _dim_method(self):
        return pd.DataFrame([
            {"method": "mcnemar_chi2_corrected", "display_name": "McNemar chi-square (corrected)", "ordinal": 0},
            {"method": "mcnemar_exact", "display_name": "McNemar exact binomial", "ordinal": 1},
            {"method": "chi2_fallback", "display_name": "Chi-square fallback", "ordinal": 2},
        ])

    def _dim_model(self):
        models = {}
        for m in self._manifest:
            if m.get("skipped"):
                continue
            key = (m["domain"], m["model"])
            if key not in models:
                models[key] = {"model": m["model"], "display_name": m["model"], "ordinal": len(models)}
        return pd.DataFrame(list(models.values()))

    def _write_dashboard_tables(self) -> None:
        from spam_fraud_detector.core.utils import deterministic_parquet

        dash = self.staging / "dashboard"
        dash.mkdir(parents=True, exist_ok=True)
        now = self.created_at

        tables: Dict[str, pd.DataFrame] = {}

        tables["dim_domain"] = self._dim_domain()
        tables["dim_group"] = self._dim_group()
        tables["dim_split"] = self._dim_split()
        tables["dim_threshold_policy"] = self._dim_threshold_policy()
        tables["dim_explainability"] = self._dim_explainability()
        tables["dim_model"] = self._dim_model()
        tables["dim_method"] = self._dim_method()

        base_cols = ["schema_version", "git_commit", "dataset_hash", "created_at"]

        # fact_metrics
        fm_rows = []
        for r in self._metrics:
            row = {c: r.get(c) for c in base_cols}
            row.update({
                "domain": r["domain"], "model": r["model"], "group": r["group"],
                "split": r["split"], "fold": r.get("fold", -1),
                "metric": r["metric"], "value": r["value"],
                "n": r.get("n"), "n_positive": r.get("n_positive"),
                "threshold": r.get("threshold"),
            })
            fm_rows.append(row)
        tables["fact_metrics"] = pd.DataFrame(fm_rows, columns=[
            "schema_version", "git_commit", "dataset_hash", "created_at",
            "domain", "model", "group", "split", "fold", "metric", "value",
            "n", "n_positive", "threshold",
        ]) if fm_rows else pd.DataFrame(columns=[
            "schema_version", "git_commit", "dataset_hash", "created_at",
            "domain", "model", "group", "split", "fold", "metric", "value",
            "n", "n_positive", "threshold",
        ])

        # fact_thresholds
        tables["fact_thresholds"] = pd.DataFrame(self._thresholds, columns=[
            "schema_version", "git_commit", "dataset_hash", "created_at",
            "domain", "model", "group", "split", "threshold", "precision",
            "recall", "f1", "tpr", "fpr", "tp", "fp", "tn", "fn",
            "cost_fp", "cost_fn", "expected_cost", "threshold_star",
            "is_at_star", "expected_cost_cfn_10_cfp_1",
        ]) if self._thresholds else pd.DataFrame(columns=[
            "schema_version", "git_commit", "dataset_hash", "created_at",
            "domain", "model", "group", "split", "threshold", "precision",
            "recall", "f1", "tpr", "fpr", "tp", "fp", "tn", "fn",
            "cost_fp", "cost_fn", "expected_cost", "threshold_star",
            "is_at_star", "expected_cost_cfn_10_cfp_1",
        ])

        # fact_thresholds_at_star
        tables["fact_thresholds_at_star"] = pd.DataFrame(self._thresholds_at_star, columns=[
            "schema_version", "git_commit", "dataset_hash", "created_at",
            "domain", "model", "group", "threshold_star", "tp_at_star",
            "fp_at_star", "fn_at_star", "tn_at_star", "expected_cost_at_star",
            "cost_fn", "cost_fp",
        ]) if self._thresholds_at_star else pd.DataFrame(columns=[
            "schema_version", "git_commit", "dataset_hash", "created_at",
            "domain", "model", "group", "threshold_star", "tp_at_star",
            "fp_at_star", "fn_at_star", "tn_at_star", "expected_cost_at_star",
            "cost_fn", "cost_fp",
        ])

        # fact_curves
        tables["fact_curves"] = pd.DataFrame(self._curves, columns=[
            "schema_version", "git_commit", "dataset_hash", "created_at",
            "domain", "model", "group", "split", "curve_type", "x", "y",
            "point_index",
        ]) if self._curves else pd.DataFrame(columns=[
            "schema_version", "git_commit", "dataset_hash", "created_at",
            "domain", "model", "group", "split", "curve_type", "x", "y",
            "point_index",
        ])

        # fact_confusion
        tables["fact_confusion"] = pd.DataFrame(self._confusion, columns=[
            "schema_version", "git_commit", "dataset_hash", "created_at",
            "domain", "model", "group", "split", "threshold",
            "tp", "fp", "tn", "fn",
        ]) if self._confusion else pd.DataFrame(columns=[
            "schema_version", "git_commit", "dataset_hash", "created_at",
            "domain", "model", "group", "split", "threshold",
            "tp", "fp", "tn", "fn",
        ])

        # fact_stats
        stats_rows = []
        stats_rows_summary = []
        if self._stats is not None:
            comparisons = self._stats.get("comparisons", [])
            for i, c in enumerate(comparisons):
                stats_rows.append({
                    "schema_version": SCHEMA_VERSION,
                    "domain": self.ctx.domain,
                    "created_at": now,
                    "model_a": c.get("model_a"),
                    "model_b": c.get("model_b"),
                    "test": c.get("test"),
                    "statistic": c.get("statistic"),
                    "p_value": c.get("p_value"),
                    "p_holm_adjusted": c.get("p_holm_adjusted"),
                    "effect_size_name": "cohen_h",
                    "effect_size_value": c.get("cohen_h"),
                    "n": c.get("n"),
                    "n_discordant": c.get("n_discordant"),
                    "delta_pr_auc": None,
                    "delta_roc_auc": None,
                    "delta_f1": None,
                    "winner": c.get("winner"),
                    "git_commit": self.ctx.git_commit,
                    "dataset_hash": self.ctx.dataset_hash,
                    "method": c.get("test"),
                    "n01": c.get("n01"),
                    "n10": c.get("n10"),
                    "family_rank": c.get("family_rank"),
                    "significant_holm_0.05": c.get("significant_holm_0.05"),
                    "significant_holm_0.01": c.get("significant_holm_0.01"),
                    "winner_significant": c.get("winner_significant"),
                })
            # Summary row per domain
            n_comparisons = len(comparisons)
            n_significant = sum(1 for c in comparisons if c.get("significant_holm_0.05"))
            winners = [c for c in comparisons if c.get("winner_significant")]
            best_winner = winners[0] if winners else None
            stats_rows_summary.append({
                "domain": self.ctx.domain,
                "n_comparisons": n_comparisons,
                "n_significant_holm_0.05": n_significant,
                "n_significant_holm_0.01": sum(1 for c in comparisons if c.get("significant_holm_0.01")),
                "best_winner_a": best_winner.get("model_a") if best_winner else None,
                "best_winner_b": best_winner.get("model_b") if best_winner else None,
                "best_winner_p_value": best_winner.get("p_value") if best_winner else None,
                "best_winner_p_holm": best_winner.get("p_holm_adjusted") if best_winner else None,
            })
        tables["fact_stats"] = pd.DataFrame(stats_rows, columns=[
            "schema_version", "domain", "created_at", "model_a", "model_b",
            "test", "statistic", "p_value", "p_holm_adjusted",
            "effect_size_name", "effect_size_value", "n", "n_discordant",
            "delta_pr_auc", "delta_roc_auc", "delta_f1", "winner",
            "git_commit", "dataset_hash",
            "method", "n01", "n10", "family_rank",
            "significant_holm_0.05", "significant_holm_0.01", "winner_significant",
        ]) if stats_rows else pd.DataFrame(columns=[
            "schema_version", "domain", "created_at", "model_a", "model_b",
            "test", "statistic", "p_value", "p_holm_adjusted",
            "effect_size_name", "effect_size_value", "n", "n_discordant",
            "delta_pr_auc", "delta_roc_auc", "delta_f1", "winner",
            "git_commit", "dataset_hash",
            "method", "n01", "n10", "family_rank",
            "significant_holm_0.05", "significant_holm_0.01", "winner_significant",
        ])
        tables["fact_stats_summary"] = pd.DataFrame(stats_rows_summary, columns=[
            "domain", "n_comparisons", "n_significant_holm_0.05",
            "n_significant_holm_0.01",
            "best_winner_a", "best_winner_b",
            "best_winner_p_value", "best_winner_p_holm",
        ]) if stats_rows_summary else pd.DataFrame(columns=[
            "domain", "n_comparisons", "n_significant_holm_0.05",
            "n_significant_holm_0.01",
            "best_winner_a", "best_winner_b",
            "best_winner_p_value", "best_winner_p_holm",
        ])

        # manifest
        tables["manifest"] = pd.DataFrame(self._manifest, columns=[
            "schema_version", "git_commit", "dataset_hash", "created_at",
            "domain", "model", "group", "skipped", "skip_reason",
            "artifact", "meta", "eval", "curve", "confusion",
            "artifact_sha256", "artifact_bytes",
            "split_manifest_hash", "feature_pipeline_hash",
            "quality_gate_hash", "test_access_count", "explainability",
        ]) if self._manifest else pd.DataFrame(columns=[
            "schema_version", "git_commit", "dataset_hash", "created_at",
            "domain", "model", "group", "skipped", "skip_reason",
            "artifact", "meta", "eval", "curve", "confusion",
            "artifact_sha256", "artifact_bytes",
            "split_manifest_hash", "feature_pipeline_hash",
            "quality_gate_hash", "test_access_count", "explainability",
        ])

        # selection
        sel_cols = ["schema_version", "domain", "rank", "model", "group",
                    "pr_auc", "f1", "recall_at_prec_0.90", "threshold",
                    "artifact", "git_commit", "dataset_hash", "created_at"]
        sel_rows = []
        ranking = self._selection or []
        for i, r in enumerate(ranking):
            sel_rows.append({
                "schema_version": SCHEMA_VERSION,
                "domain": self.domain,
                "rank": r.get("rank", i + 1),
                "model": r.get("model"),
                "group": r.get("group"),
                "pr_auc": r.get("pr_auc"),
                "f1": r.get("f1"),
                "recall_at_prec_0.90": r.get("recall_at_prec_0.90"),
                "threshold": r.get("threshold"),
                "artifact": r.get("artifact"),
                "git_commit": self.ctx.git_commit,
                "dataset_hash": self.ctx.dataset_hash,
                "created_at": now,
            })
        tables["selection"] = pd.DataFrame(sel_rows, columns=sel_cols)

        # predictions table (from staging predictions dir if present)
        tables["fact_predictions"] = pd.DataFrame(columns=[
            "schema_version", "domain", "model", "group", "row_id",
            "y_true", "y_score", "y_pred", "split", "fold",
        ])

        # Write each table as parquet + csv
        for tname, df in tables.items():
            pq_path = dash / f"{tname}.parquet"
            csv_path = dash / f"{tname}.csv"
            deterministic_parquet(df, pq_path)
            try:
                df.to_csv(csv_path, index=False)
            except Exception:
                pass

        # selection parquet/csv already handled above via tables; write explicitly too
        if tables.get("selection") is not None:
            deterministic_parquet(tables["selection"], dash / "selection.parquet")
            tables["selection"].to_csv(dash / "selection.csv", index=False)

        # index.json
        table_entries = []
        for tname in ["fact_metrics", "fact_confusion", "fact_thresholds",
                      "fact_curves", "fact_stats", "fact_stats_summary",
                      "selection", "manifest"]:
            df = tables[tname]
            table_entries.append({
                "name": tname,
                "path": f"{tname}.parquet",
                "rows": int(len(df)),
            })
        index = {
            "created_at": now,
            "dataset_hash": self.ctx.dataset_hash,
            "domain": self.domain,
            "git_commit": self.ctx.git_commit,
            "recommended_views": ["leaderboard", "group_compare", "threshold_sweep"],
            "schema_version": SCHEMA_VERSION,
            "tables": table_entries,
        }
        atomic_write_json(dash / "index.json", index)

        # filters.json
        filters = {
            "levels": {
                "L1": {"allowed": ["spam_detection", "fraud_detection"], "default": "spam_detection",
                       "dimension": "domain", "label": "Domain", "multi": False, "source": "dim_domain"},
                "L2": {"allowed": ["validation", "test"], "default": "test",
                       "dimension": "split", "label": "Split", "multi": False, "source": "dim_split"},
                "L3": {"allowed": ["scaled", "unscaled", "anomaly"], "default": "scaled",
                       "dimension": "group", "label": "Group", "multi": False, "source": "dim_group"},
                "L4": {"allowed": sorted({m["model"] for m in self._manifest if not m.get("skipped")}),
                       "default": sorted({m["model"] for m in self._manifest if not m.get("skipped")}),
                       "dimension": "model", "label": "Model", "multi": True, "source": "dim_model"},
                "L5": {"allowed": KNOWN_METRICS, "default": "pr_auc",
                       "dimension": "metric", "label": "Metric", "multi": False, "source": "fact_metrics"},
                "L6": {"allowed": [-1], "default": -1,
                       "dimension": "fold", "label": "Fold", "multi": False, "source": "fact_metrics"},
                "L7": {"allowed": [round(i / 100, 2) for i in range(101)], "default": 0.5,
                       "dimension": "threshold", "label": "Threshold", "multi": False, "source": "fact_thresholds"},
                "L8": {"allowed": [1, 2, 3], "default": 3,
                       "dimension": "rank", "label": "Top-K", "multi": False, "source": "selection"},
                "L9": {"allowed": ["fraud", "spam", "anomaly"], "default": "all",
                       "dimension": "persona", "label": "Persona", "multi": False, "source": "dim_explainability"},
                "L10": {"allowed": ["mcnemar_chi2_corrected", "mcnemar_exact", "chi2_fallback"], "default": "mcnemar_chi2_corrected",
                        "dimension": "method", "label": "Test Method", "multi": False, "source": "dim_method"},
            }
        }
        atomic_write_json(dash / "filters.json", filters)

        # _examples.sql
        examples_sql = (
            "-- Example queries (DuckDB)\n"
            "SELECT model, group, metric, value FROM fact_metrics "
            "WHERE domain='fraud_detection' AND split='test' AND metric='pr_auc' ORDER BY value DESC;\n"
            "SELECT model, group, tp, fp, tn, fn FROM fact_confusion WHERE domain='fraud_detection' AND split='test';\n"
            "SELECT threshold, precision, recall, f1 FROM fact_thresholds "
            "WHERE domain='fraud_detection' AND model='XGBoost' AND group='unscaled' AND split='test' ORDER BY threshold;\n"
        )
        atomic_write_text(dash / "_examples.sql", examples_sql)

    # ------------------------------------------------------------------ #
    # Finalize + validate
    # ------------------------------------------------------------------ #

    def validate(self) -> None:
        """Validate persona/explainability invariants on every manifest entry."""
        from .exceptions import ContractViolation as CV
        tree_models = {"XGBClassifier", "LGBMClassifier", "RandomForestClassifier",
                       "GradientBoostingClassifier", "ExtraTreesClassifier"}
        expected_persona_by_domain = {"spam_detection": "spam", "fraud_detection": "fraud"}
        for m in self._manifest:
            if m.get("skipped"):
                continue
            group = m.get("group")
            name = m.get("model", "")
            expl = m.get("explainability")
            if group == "anomaly":
                if expl is not None and expl.get("persona") != "anomaly" \
                        and expl.get("backend") != "off":
                    raise CV(f"anomaly model {name} must have persona=anomaly or backend=off")
            elif name.split("__")[-1] if False else False:
                pass
            cls_name = ""
            try:
                if m.get("artifact"):
                    model_path = self.staging / "models" / self.ctx.domain / "all" / Path(m["artifact"]).name
                    if model_path.exists():
                        loaded = joblib.load(model_path)
                        cls_name = type(loaded).__name__
            except Exception:
                cls_name = ""
            is_tree = cls_name in tree_models
            expected = expected_persona_by_domain.get(self.domain)
            if is_tree and expected:
                if expl is not None and expl.get("persona") not in (expected, None) \
                        and expl.get("backend") != "off":
                    raise CV(f"tree model {name} in {self.domain} must have persona={expected} or backend=off")

    def finalize(self) -> Path:
        """Atomically move staging outputs into out_root and clean up staging."""
        domain = self.ctx.domain

        # Write dashboard tables into staging
        self._write_dashboard_tables()

        # Write stats files
        stats_dir = self.staging / "stats" / domain
        stats_dir.mkdir(parents=True, exist_ok=True)
        if self._stats is not None:
            atomic_write_json(stats_dir / "ab_test.json", self._stats)
            atomic_write_text(stats_dir / "summary.md", self._stats_summary_md)

        # CONTRACT.json
        contract = {
            "data_layout": {
                "hash_algorithm": "sha256",
                "processed": {"fraud": "data/processed/fraud", "spam": "data/processed/spam"},
                "raw": {"fraud": "data/raw/fraud", "spam": "data/raw/spam"},
            },
            "docs_policy": {
                "core_readme": "spam_fraud_detector/core/README.md",
                "core_readme_count_expected": 1,
                "enforced_by": "make docs-check",
                "per_module_readmes_under_core_allowed": False,
            },
            "explainability": {
                "personas": ["fraud", "spam", "anomaly"],
            },
            "filter_levels": ["domain", "split", "group", "model", "metric", "fold", "threshold", "rank", "persona"],
            "layout": {
                "bundles": "models/<domain>/all/<model>__<group>.{joblib,meta.json,eval.json,curve.json,confusion.json}",
                "dashboard": "dashboard/{fact_*,dim_*,selection,manifest,index.json,filters.json,_examples.sql}",
                "predictions": "predictions/<domain>/all/<model>__<group>.parquet",
                "stats": "stats/<domain>/{ab_test.json,summary.md}",
                "top3": "models/<domain>/top3/rank_<k>_<model>.{joblib,meta.json}",
            },
            "schema_version": SCHEMA_VERSION,
            "threshold_policy": {
                "cost_matrix_source": "data/processed/<domain>/05_kpis/kpis.json",
                "dashboard_tables": ["fact_thresholds", "fact_thresholds_at_star", "dim_threshold_policy"],
                "default": "auto",
                "enforced_by": ["make verify-thresholds", "scripts/verify_thresholds.py"],
                "policies_allowed": ["auto", "fixed"],
                "selection_policies_allowed": ["auto", "pr_auc", "f1", "expected_cost"],
            },
        }
        atomic_write_json(self.staging / "CONTRACT.json", contract)

        # README for outputs root
        atomic_write_text(self.staging / "README.md", "# outputs\nGenerated by ResultStore.\n")

        # Merge staging into out_root.
        # - If out_root does not exist yet, do a single atomic os.replace so a
        #   failed move leaves nothing behind (test_atomicity).
        # - If out_root already exists (e.g. a second domain finalized into the
        #   same outputs/), accumulate parquet tables instead of overwriting.
        out_root = self.out_root
        staging = self.staging
        if out_root.exists():
            self._merge_tree(staging, out_root)
        else:
            os.replace(staging, out_root)
        # remove any leftover sibling staging dirs
        for p in self.out_root.parent.glob("outputs.staging.*"):
            if p != self.out_root:
                shutil.rmtree(p, ignore_errors=True)
        return out_root

    def _merge_tree(self, src, dst):
        """Recursively merge src tree into dst, concatenating parquet tables.

        fact_predictions parquet files are overwritten (not merged) because
        different runs may use different test splits, and merging would
        pollute predictions with data from incompatible splits.
        """
        dst.mkdir(parents=True, exist_ok=True)
        for item in sorted(src.iterdir()):
            s = item
            d = dst / item.name
            if s.is_dir():
                self._merge_tree(s, d)
            elif s.suffix == ".parquet":
                if s.parent.name == "fact_predictions":
                    # Overwrite: different runs may use different test splits.
                    d.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(s, d)
                else:
                    self._merge_parquet(s, d)
            else:
                d.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(s, d)

    def _merge_parquet(self, src_path, dst_path):
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        new_df = pd.read_parquet(src_path)
        if dst_path.exists():
            try:
                old_df = pd.read_parquet(dst_path)
                df = pd.concat([old_df, new_df], ignore_index=True)
                df = df.drop_duplicates()
            except Exception:
                df = new_df
        else:
            df = new_df
        from spam_fraud_detector.core.utils import deterministic_parquet
        deterministic_parquet(df, dst_path)
        try:
            df.to_csv(dst_path.with_suffix(".csv"), index=False)
        except Exception:
            pass
