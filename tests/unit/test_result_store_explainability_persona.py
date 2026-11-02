"""Tests for ResultStore explainability persona support."""
from __future__ import annotations

import numpy as np
import pytest

from spam_fraud_detector.core import ResultStore, RunContext


@pytest.fixture()
def store(tmp_path):
    out = tmp_path / "outputs"
    ctx = RunContext(domain="spam_detection", seed=1)
    return ResultStore(out, ctx), out


def test_record_explainability_persona(store):
    rs, out = store
    y = np.array([0, 1, 0, 1])
    s = np.array([0.1, 0.9, 0.2, 0.8])
    rs.write_model("M", "scaled", object(), hyperparams={},
                   eval_val={"accuracy": 0.9, "tp": 1, "fp": 0, "tn": 2, "fn": 1, "n": 4, "n_positive": 2},
                   eval_test={"accuracy": 0.9, "tp": 1, "fp": 0, "tn": 2, "fn": 1, "n": 4, "n_positive": 2},
                   y_true_val=y, y_score_val=s, y_true_test=y, y_score_test=s, threshold=0.5,
                   confusion={"validation": {"tp": 1, "fp": 0, "tn": 2, "fn": 1}, "test": {"tp": 1, "fp": 0, "tn": 2, "fn": 1}})
    rs.record_explainability("M", "scaled", {
        "summary_path": "reports/spam_detection/docs/explainability/summary.png",
        "waterfall_paths": [],
        "background_hash": "sha256:" + "a" * 64,
        "feature_names_hash": "sha256:" + "b" * 64,
        "backend": "shap",
    }, persona="spam")
    assert any(m.get("explainability", {}).get("persona") == "spam" for m in rs._manifest)


def test_record_explainability_before_write_model_raises(store):
    rs, out = store
    with pytest.raises(Exception):
        rs.record_explainability("NonExistent", "scaled", {"backend": "shap"}, persona="fraud")


def test_idempotent(store):
    rs, out = store
    y = np.array([0, 1, 0, 1])
    s = np.array([0.1, 0.9, 0.2, 0.8])
    rs.write_model("M", "scaled", object(), hyperparams={},
                   eval_val={"accuracy": 0.9, "tp": 1, "fp": 0, "tn": 2, "fn": 1, "n": 4, "n_positive": 2},
                   eval_test={"accuracy": 0.9, "tp": 1, "fp": 0, "tn": 2, "fn": 1, "n": 4, "n_positive": 2},
                   y_true_val=y, y_score_val=s, y_true_test=y, y_score_test=s, threshold=0.5,
                   confusion={"validation": {"tp": 1, "fp": 0, "tn": 2, "fn": 1}, "test": {"tp": 1, "fp": 0, "tn": 2, "fn": 1}})
    rs.record_explainability("M", "scaled", {"backend": "shap"}, persona="fraud")
    rs.record_explainability("M", "scaled", {"backend": "shap"}, persona="fraud")
    # should not raise, idempotent
    assert True