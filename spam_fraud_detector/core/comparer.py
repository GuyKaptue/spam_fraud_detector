# spam_fraud_detector/core/comparer.py
# type: ignore
import os
import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.base import ClassifierMixin
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, roc_curve, auc, precision_recall_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.exceptions import NotFittedError
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
        random_state: int = 42
    ):
        """
        General-purpose benchmarking class for binary classification tasks.
        Supports both scaled and unscaled model groups.
        """
        self.task = task
        self.df = df
        self.data_path = data_path
        self.test_size = test_size
        self.random_state = random_state
        self.incompatible_models = []

        # Load dataset config from utils
        loader = KaggleDataLoader(task=task)
        #config = loader.DATASETS[task]
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
        """Splits the data into stratified train/test sets."""
        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]
        if (X.select_dtypes(include=["number"]) < 0).any().any():
            self.incompatible_models.append("MultinomialNB")
            self.scaling_not_required.pop("MultinomialNB", None)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, stratify=y, random_state=self.random_state
        )

    def evaluate_model(self, model, scale: bool = False):
        try:
            pipeline = Pipeline([
                ("scaler", StandardScaler()) if scale else ("passthrough", "passthrough"),
                ("classifier", model)
            ])
            pipeline.fit(self.X_train, self.y_train)
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
            else:
                fpr, tpr, roc_auc = None, None, None
                precision, recall = None, None

            acc = accuracy_score(self.y_test, preds)
            prec = precision_score(self.y_test, preds)
            rec = recall_score(self.y_test, preds)
            f1 = f1_score(self.y_test, preds)
            cm = confusion_matrix(self.y_test, preds)

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
                "fpr": fpr,
                "tpr": tpr,
                "roc_auc": roc_auc,
                "precision_curve": precision,
                "recall_curve": recall
            }

        except ValueError as e:
            if "Negative values in data passed to MultinomialNB" in str(e):
                print("⚠️ Skipping MultinomialNB due to negative values in input data.")
                return None
            else:
                raise

    def compare_groups(self):
        """Evaluates all models and returns metrics, confusion matrices, and raw outputs."""
        results_scaled = {}
        results_unscaled = {}

        for name, model in self.scaling_required.items():
            print(f"Evaluating (scaled): {name}")
            results_scaled[name] = self.evaluate_model(model, scale=True)

        for name, model in self.scaling_not_required.items():
            print(f"Evaluating (unscaled): {name}")
            results_unscaled[name] = self.evaluate_model(model, scale=False)

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
            "unscaled_raw": results_unscaled
        }
