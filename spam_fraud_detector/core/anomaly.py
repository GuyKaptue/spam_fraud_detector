"""Anomaly detectors as a first-class model group."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from .exceptions import DataSchemaError


class AnomalyComparer:
    """Trains unsupervised detectors on the majority (negative) class only."""

    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test,
                 random_state=42, **kwargs):
        y_train = np.asarray(y_train).ravel()
        int(np.sum(y_train))
        if np.any(y_train == 1):
            raise DataSchemaError(
                "AnomalyComparer requires X_train to contain only the "
                "legitimate (negative) class; positives leaked."
            )
        self.X_train_neg = X_train
        self.X_val = X_val
        self.y_val = np.asarray(y_val).ravel()
        self.X_test = X_test
        self.y_test = np.asarray(y_test).ravel()
        self.random_state = random_state
        self.results_ = {}

    def _as_score(self, decision_output, invert=True):
        s = np.asarray(decision_output, dtype=float).ravel()
        if invert:
            s = -s
        return s

    def _normalize(self, scores, train_scores):
        s = np.asarray(scores, dtype=float).ravel()
        t = np.asarray(train_scores, dtype=float).ravel()
        lo, hi = float(t.min()), float(t.max())
        if hi - lo < 1e-12:
            return np.clip(s, 0.0, 1.0)
        return np.clip((s - lo) / (hi - lo), 0.0, 1.0)
    def fit(self):
        detectors = {
            "IsolationForest": IsolationForest(
                n_estimators=300, contamination="auto",
                random_state=self.random_state,
            ),
            "LOF": LocalOutlierFactor(novelty=True, contamination="auto"),
            "OneClassSVM": OneClassSVM(nu=0.01, kernel="rbf", gamma="scale"),
        }
        for name, det in detectors.items():
            det.fit(self.X_train_neg)
            train_raw = det.decision_function(self.X_train_neg)
            val_raw = det.decision_function(self.X_val)
            test_raw = det.decision_function(self.X_test)
            train_score = self._as_score(train_raw)
            val_score = self._normalize(self._as_score(val_raw), train_score)
            test_score = self._normalize(self._as_score(test_raw), train_score)
            self.results_[name] = {
                "detector": det,
                "scores_val": val_score,
                "scores_test": test_score,
            }
        return self

    def score(self, X=None):
        if not self.results_:
            self.fit()
        return {name: r["scores_test"] for name, r in self.results_.items()}

    def compare(self):
        if not self.results_:
            self.fit()
        from sklearn.metrics import average_precision_score, roc_auc_score
        out = {}
        for name, r in self.results_.items():
            val_scores = r["scores_val"]
            test_scores = r["scores_test"]
            val_metrics = {
                "roc_auc": float(roc_auc_score(self.y_val, val_scores)),
                "pr_auc": float(average_precision_score(self.y_val, val_scores)),
            }
            test_metrics = {
                "roc_auc": float(roc_auc_score(self.y_test, test_scores)),
                "pr_auc": float(average_precision_score(self.y_test, test_scores)),
            }
            out[name] = {
                "scores_val": val_scores,
                "scores_test": test_scores,
                "metrics_val": val_metrics,
                "metrics_test": test_metrics,
                "detector": self.results_[name]["detector"],
            }
        return out

    def get_results(self):
        return self.results_
