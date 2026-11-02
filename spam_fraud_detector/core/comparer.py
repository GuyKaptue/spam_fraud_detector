# spam_fraud_detector/core/comparer.py
# type: ignore
import os
import pandas as pd
try:
    import mlflow
except Exception:  # pragma: no cover
    mlflow = None
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, roc_curve, auc, precision_recall_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
# Scaling Required
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# Scaling Not Required
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from spam_fraud_detector.core.utils import format_metrics, safe_model_name
from spam_fraud_detector.core.data_loader import KaggleDataLoader

class ClassifierGroupComparer:
    def __init__(
        self,
        task: str,
        df: pd.DataFrame = None,
        data_path: str = "",
        target_col: str = None,
        drop_cols: list = None,
        test_size: float = 0.2,
        random_state: int = 42,
        store=None,
        verbose: bool = False,
    ):
        """
        General-purpose benchmarking class for binary classification tasks.
        Supports both scaled and unscaled model groups.
        """
        self.task = task
        self.df = df
        self.data_path = data_path
        self.target_col = target_col or ("Class" if task == "fraud" else "Prediction")
        self.drop_cols = drop_cols or (["Time"] if task == "fraud" else ["Email No."])
        self.test_size = test_size
        self.random_state = random_state
        self.incompatible_models = []
        self.store = store
        self.strict_anomaly = False
        self.verbose = verbose

        # Load dataset config from utils
        self.target_col = target_col or ("Class" if task == "fraud" else "Prediction")
        self.drop_cols = drop_cols or (["Time"] if task == "fraud" else ["Email No."])

        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

        self.scaling_required = {
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "SVC": SVC(),
            "KNN": KNeighborsClassifier(),
            "SGDClassifier": SGDClassifier(),
            "MLPClassifier": MLPClassifier(max_iter=500)
        }

        self.scaling_not_required = {
            "MultinomialNB": MultinomialNB(),
            "RandomForest": RandomForestClassifier(random_state=self.random_state),
            "GradientBoosting": GradientBoostingClassifier(random_state=self.random_state),
            "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=self.random_state),
            "LightGBM": LGBMClassifier(random_state=self.random_state)
        }

        #if self.target_col == "Class":  # fraud detection
        #   self.scaling_not_required.pop("MultinomialNB", None)

    def load_data(self):
        """Loads data from a DataFrame or file path and drops specified columns."""
        if self.df is not None:
            self.df = self.df.copy()
        elif self.data_path:
            ext = os.path.splitext(self.data_path)[-1]
            if ext == ".csv":
                self.df = pd.read_csv(self.data_path)
            elif ext in [".xlsx", ".xls"]:
                self.df = pd.read_excel(self.data_path)
            else:
                raise ValueError("Unsupported file format. Use .csv or .xlsx")
        else:
            loader = KaggleDataLoader(task=self.task)
            self.df = loader.download_load_and_save()

        if self.drop_cols:
            self.df.drop(columns=self.drop_cols, inplace=True, errors="ignore")

        if self.target_col not in self.df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in dataset.")

    def prepare_data(self):
        """Load persisted train/val splits from SplitManager."""
        from spam_fraud_detector.core.split import SplitManager
        sm = SplitManager(domain=self.task, data_root="data")
        train_df = sm.load_split("train")
        val_df = sm.load_split("val")
        if train_df is None or val_df is None:
            raise ValueError("SplitManager returned None for train/val")
        X = train_df.drop(columns=[self.target_col] + self.drop_cols, errors="ignore")
        y = train_df[self.target_col]
        if (X.select_dtypes(include=["number"]) < 0).any().any():
            self.incompatible_models.append({"model": "MultinomialNB", "group": "unscaled", "reason": "negative inputs"})
            self.scaling_not_required.pop("MultinomialNB", None)
        self.X_train = X
        self.y_train = y
        self.X_val = val_df.drop(columns=[self.target_col] + self.drop_cols, errors="ignore")
        self.y_val = val_df[self.target_col]
        self.X_test = None
        self.y_test = None

    def _load_test_split(self):
        """Load test split once, authorized."""
        from spam_fraud_detector.core.split import SplitManager, authorize_test_access
        sm = SplitManager(domain=self.task, data_root="data")
        with authorize_test_access(run_id="comparer", caller="ClassifierGroupComparer", authorized_by="comparer"):
            test_df = sm.load_split("test")
        self.X_test = test_df.drop(columns=[self.target_col] + self.drop_cols, errors="ignore")
        self.y_test = test_df[self.target_col]

    def evaluate_model(self, model, scale: bool = False):
        model_name = safe_model_name(model)
        if self.verbose:
            print(f"  [DEBUG] Training {model_name} (scaled={scale})...")
        try:
            pipeline = Pipeline([
                ("scaler", StandardScaler()) if scale else ("passthrough", "passthrough"),
                ("classifier", model)
            ])
            import time
            t0 = time.time()
            pipeline.fit(self.X_train, self.y_train)
            fit_time = time.time() - t0
            if self.verbose:
                print(f"  [DEBUG] {model_name} fit complete in {fit_time:.3f}s")
            preds = pipeline.predict(self.X_test)

            if hasattr(pipeline.named_steps["classifier"], "predict_proba"):
                probs = pipeline.predict_proba(self.X_test)[:, 1]
            elif hasattr(pipeline.named_steps["classifier"], "decision_function"):
                probs = pipeline.decision_function(self.X_test)
            else:
                probs = None

            if probs is not None:
                fpr, tpr, _ = roc_curve(self.y_test, probs)
                roc_auc = auc(fpr, tpr)
                precision, recall, _ = precision_recall_curve(self.y_test, probs)
                from sklearn.metrics import average_precision_score
                average_precision = average_precision_score(self.y_test, probs)
            else:
                fpr, tpr, roc_auc = None, None, None
                precision, recall = None, None
                average_precision = None

            acc = accuracy_score(self.y_test, preds)
            prec = precision_score(self.y_test, preds)
            rec = recall_score(self.y_test, preds)
            f1 = f1_score(self.y_test, preds)
            cm = confusion_matrix(self.y_test, preds)
            if self.verbose:
                print(f"  [DEBUG] {model_name} results: acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} f1={f1:.4f}")

            with mlflow.start_run(run_name=f"{safe_model_name(model)}_{self.task}"):
                mlflow.log_param("scaled", scale)
                mlflow.log_metric("accuracy", acc)
                mlflow.log_metric("precision", prec)
                mlflow.log_metric("recall", rec)
                mlflow.log_metric("f1", f1)
                if roc_auc is not None:
                    mlflow.log_metric("roc_auc", roc_auc)
                mlflow.log_dict({"confusion_matrix": cm.tolist()}, "confusion_matrix.json")

            return {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "confusion_matrix": cm,
                "probs": probs,
                "true": self.y_test,
                "row_ids": list(self.X_test.index),
                "fpr": fpr,
                "tpr": tpr,
                "roc_auc": roc_auc,
                "average_precision": average_precision,
                "precision_curve": precision,
                "recall_curve": recall,
                "pipeline": pipeline,
            }

        except ValueError as e:
            if "Negative values in data passed to MultinomialNB" in str(e):
                print("⚠️ Skipping MultinomialNB due to negative values in input data.")
                return None
            else:
                raise

    def _persist(self, name, group, res):
        if self.store is None:
            return
        from sklearn.metrics import confusion_matrix as _cm
        y_true = list(res["true"])
        y_score = list(res["probs"])
        threshold = 0.5
        y_pred = [1 if p >= threshold else 0 for p in y_score]
        cm = _cm(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)
        eval_metrics = {
            "accuracy": res.get("accuracy"),
            "precision": res.get("precision"),
            "recall": res.get("recall"),
            "f1": res.get("f1"),
            "roc_auc": res.get("roc_auc"),
            "pr_auc": res.get("average_precision"),
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
            "n": len(y_true), "n_positive": int(sum(y_true)),
        }
        confusion = {"test": {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}}
        curves = {"test": {
            "roc": {"fpr": [float(x) for x in res.get("fpr", [])],
                    "tpr": [float(x) for x in res.get("tpr", [])]},
            "pr": {"recall": [float(x) for x in res.get("recall_curve", [])],
                   "precision": [float(x) for x in res.get("precision_curve", [])]},
        }}
        hyperparams = {}
        feature_columns = list(self.X_train.columns) if self.X_train is not None else None
        self.store.write_model(
            name, group, res.get("pipeline"),
            hyperparams=hyperparams,
            eval_val=eval_metrics,
            eval_test=eval_metrics,
            curves=curves,
            confusion=confusion,
            threshold=None,
            threshold_policy=getattr(self, "threshold_policy", "auto"),
            y_true_val=y_true,
            y_score_val=y_score,
            y_true_test=y_true,
            y_score_test=y_score,
            row_ids_test=res.get("row_ids", []),
            feature_columns=feature_columns,
        )

    def compare_groups(self):
        """Evaluates all models and returns metrics, confusion matrices, and raw outputs."""
        results_scaled = {}
        results_unscaled = {}

        # Load test split once, authorized
        self._load_test_split()

        for name, model in self.scaling_required.items():
            print(f"Evaluating (scaled): {name}")
            res = self.evaluate_model(model, scale=True)
            results_scaled[name] = res
            if res is not None and self.store is not None:
                self._persist(name, "scaled", res)

        for name, model in self.scaling_not_required.items():
            print(f"Evaluating (unscaled): {name}")
            res = self.evaluate_model(model, scale=False)
            results_unscaled[name] = res
            if res is None:
                self.incompatible_models.append({"model": name, "group": "unscaled", "reason": "incompatible input"})
            if res is not None and self.store is not None:
                self._persist(name, "unscaled", res)

        df_scaled = pd.DataFrame({
            k: format_metrics(v) for k, v in results_scaled.items() if v is not None
        }).T

        df_unscaled = pd.DataFrame({
            k: format_metrics(v) for k, v in results_unscaled.items() if v is not None
        }).T

        group_comparison = pd.DataFrame({
            "Scaled Avg": df_scaled.mean(),
            "Unscaled Avg": df_unscaled.mean()
        })

        scaled_auc = {k: v["roc_auc"] for k, v in results_scaled.items() if v and v["roc_auc"] is not None}
        unscaled_auc = {k: v["roc_auc"] for k, v in results_unscaled.items() if v and v["roc_auc"] is not None}

        df_scaled_auc = pd.DataFrame.from_dict(scaled_auc, orient="index", columns=["AUC"]).sort_values("AUC", ascending=False)
        df_unscaled_auc = pd.DataFrame.from_dict(unscaled_auc, orient="index", columns=["AUC"]).sort_values("AUC", ascending=False)

        scaled_avg_prec = {
            k: v["average_precision"] for k, v in results_scaled.items() if v and v["average_precision"] is not None
        }
        unscaled_avg_prec = {
            k: v["average_precision"] for k, v in results_unscaled.items() if v and v["average_precision"] is not None
        }

        df_scaled_avg_prec = pd.DataFrame.from_dict(scaled_avg_prec, orient="index", columns=["Average Precision"]).sort_values("Average Precision", ascending=False)
        df_unscaled_avg_prec = pd.DataFrame.from_dict(unscaled_avg_prec, orient="index", columns=["Average Precision"]).sort_values("Average Precision", ascending=False)

        if self.store is not None:
            self.store.write_skipped(self.incompatible_models)

        anomaly_results = {}
        anomaly_raw = {}
        if getattr(self, "include_anomaly", True):
            anomaly_results, anomaly_raw = self.compare_anomalies()

        return {
            "scaled_metrics": df_scaled.sort_values("f1", ascending=False),
            "unscaled_metrics": df_unscaled.sort_values("f1", ascending=False),
            "group_comparison": group_comparison,
            "scaled_confusion_matrices": {k: v["confusion_matrix"] for k, v in results_scaled.items() if v},
            "unscaled_confusion_matrices": {k: v["confusion_matrix"] for k, v in results_unscaled.items() if v},
            "scaled_auc": df_scaled_auc,
            "unscaled_auc": df_unscaled_auc,
            "scaled_precision": df_scaled_avg_prec,
            "unscaled_precision": df_unscaled_avg_prec,
            "scaled_raw": results_scaled,
            "unscaled_raw": results_unscaled,
            "anomaly_metrics": anomaly_results,
            "anomaly_raw": anomaly_raw,
        }

    def compare_anomalies(self):
        """Train and evaluate anomaly detectors on the legitimate class only."""
        anomaly_results = {}
        anomaly_raw = {}
        if self.store is None:
            return anomaly_results, anomaly_raw
        try:
            import logging
            logger = logging.getLogger(__name__)
            from spam_fraud_detector.core.anomaly import AnomalyComparer
            # Filter to legitimate class only (caller responsibility)
            import numpy as np
            mask_train_neg = (self.y_train == 0)
            X_train_neg = self.X_train[mask_train_neg] if hasattr(self.X_train, "__getitem__") else self.X_train
            y_train_neg = self.y_train[mask_train_neg]
            if len(X_train_neg) < 100:
                raise ValueError("too few legitimate train rows")
            logger.info("Fitting anomaly detectors on %d legitimate train rows", len(X_train_neg))
            X_train = X_train_neg
            y_train = y_train_neg
            X_val = self.X_test
            y_val = self.y_test
            X_test = self.X_test
            y_test = self.y_test
            ac = AnomalyComparer(X_train, y_train, X_val, y_val, X_test, y_test,
                                random_state=self.random_state)
            results = ac.compare()
            for name, r in results.items():
                anomaly_raw[name] = r
                self.store.write_model(
                    name, "anomaly", r["detector"],
                    hyperparams={"contamination": "auto"},
                    eval_val=r["metrics_val"],
                    eval_test=r["metrics_test"],
                    y_true_val=self.y_test,
                    y_score_val=r["scores_val"],
                    y_true_test=self.y_test,
                    y_score_test=r["scores_test"],
                )
                anomaly_results[name] = r["metrics_test"]
        except Exception as e:
            import logging, traceback
            logging.getLogger(__name__).error("Anomaly comparison failed: %s", e, exc_info=True)
            # record failure
            try:
                import json, os
                from datetime import datetime, timezone
                os.makedirs("outputs/logs", exist_ok=True)
                with open("outputs/logs/anomaly_failures.log", "a") as f:
                    f.write(json.dumps({"domain": f"{self.task}_detection", "reason": str(e),
                                       "caller": "compare_anomalies",
                                       "timestamp": datetime.now(timezone.utc).isoformat(),
                                       "traceback": traceback.format_exc()}) + "\n")
            except Exception:
                pass
            if getattr(self, "strict_anomaly", False):
                raise
        return anomaly_results, anomaly_raw
