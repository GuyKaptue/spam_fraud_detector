"""Tests for hypothetical case-file building (v4.4.6)."""
import numpy as np
import pandas as pd

from spam_fraud_detector.core.explainer import (
    ModelExplainer,
    build_hypothetical_casefile,
    render_hypothetical_case_waterfall,
)
from spam_fraud_detector.core.fake_subjects import load_scenarios, subject_generator
from spam_fraud_detector.core.recommendations import recommend


class _LinearModel:
    """Tiny deterministic linear model with predict_proba and feature_names_in_."""

    def __init__(self, coef, intercept=0.0, feature_names=None):
        self.coef_ = np.asarray(coef, dtype=float)
        self.intercept_ = float(intercept)
        self.feature_names_in_ = list(feature_names or [f"f{i}" for i in range(len(coef))])

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        z = X @ self.coef_ + self.intercept_
        p = 1.0 / (1.0 + np.exp(-z))
        return np.column_stack([1 - p, p])


def _make_explainer(persona, model, feature_names):
    X_bg = pd.DataFrame(np.zeros((10, len(feature_names))), columns=feature_names)
    if persona == "spam":
        return ModelExplainer(model, X_bg, feature_names=feature_names,
                              vocabulary=feature_names, backend="shap")
    return ModelExplainer(model, X_bg, feature_names=feature_names, backend="shap")


def test_hypothetical_fraud_casefile_structure():
    feature_names = ["Amount", "V14", "V12"]
    model = _LinearModel([0.5, -0.4, -0.2], intercept=0.0, feature_names=feature_names)
    explainer = _make_explainer("fraud", model, feature_names)
    scenario = load_scenarios("fraud")[0]
    training_means = {"Amount": 100.0, "V14": 0.0, "V12": 0.0}
    provenance = {"model": "M", "group": "unscaled", "dataset_hash": "sha256:x",
                  "background_hash": "sha256:y", "feature_names_hash": "sha256:z",
                  "split": "val", "recorded_at": "2026-01-01T00:00:00Z"}
    casefile, row = build_hypothetical_casefile(
        "fraud", model, explainer, scenario, feature_names, training_means, None,
        provenance, lambda sid, seed=42, persona=None: subject_generator(sid, seed=seed, persona="fraud"),
        lambda cf: recommend("fraud", cf), threshold=0.5,
    )
    assert casefile["case_type"] == "hypothetical"
    assert casefile["persona"] == "fraud"
    assert casefile["scenario_id"] == scenario["scenario_id"]
    assert casefile["y_true"] is None
    assert 0.0 <= casefile["score"] <= 1.0
    assert casefile["y_pred"] in (0, 1)
    assert casefile["provenance"].get("row_id") is None
    # fabricated features applied
    assert row.at[0, "Amount"] == scenario["features"]["Amount"]
    assert row.at[0, "V14"] == scenario["features"]["V14"]
    assert row.at[0, "V12"] == scenario["features"]["V12"]
    # top features present
    tf = casefile["top_features"]
    assert len(tf) == 3
    for f in tf:
        assert set(f) == {"name", "value", "shap_value", "direction", "explanation"}
    # recommendation present and valid
    rec = casefile["recommendation"]
    assert rec["action"] in {"block", "escalate", "review", "monitor", "clear"}
    assert rec["priority"] in {"high", "medium", "low"}
    assert rec["confidence"] in {"high", "medium", "low"}
    assert rec["suggested_next_steps"]
    assert "[NAME]" in rec["reviewer_notes_template"]
    assert "[DATE]" in rec["reviewer_notes_template"]
    # narrative mentions hypothetical and subject name
    assert "hypothetical" in casefile["narrative"].lower()
    assert casefile["subject"]["full_name"] in casefile["narrative"]


def test_hypothetical_spam_casefile_structure():
    feature_names = ["click", "prize", "free", "win"]
    model = _LinearModel([1.0, 0.8, 0.6, 0.4], intercept=-1.0, feature_names=feature_names)
    explainer = _make_explainer("spam", model, feature_names)
    scenario = load_scenarios("spam")[0]
    provenance = {"model": "RF", "group": "unscaled", "dataset_hash": "sha256:x",
                  "background_hash": "sha256:y", "feature_names_hash": "sha256:z",
                  "split": "val", "recorded_at": "2026-01-01T00:00:00Z"}
    casefile, _row = build_hypothetical_casefile(
        "spam", model, explainer, scenario, feature_names, {}, None,
        provenance, lambda sid, seed=42, persona=None: subject_generator(sid, seed=seed, persona="spam"),
        lambda cf: recommend("spam", cf), threshold=0.5,
    )
    assert casefile["case_type"] == "hypothetical"
    assert casefile["persona"] == "spam"
    assert casefile["y_true"] is None
    assert 0.0 <= casefile["score"] <= 1.0
    tf = casefile["top_tokens"]
    assert len(tf) == 4
    for t in tf:
        assert set(t) == {"token", "count", "shap_value", "direction", "explanation"}
    rec = casefile["recommendation"]
    assert rec["action"] in {"block", "escalate", "review", "monitor", "clear"}
    assert rec["estimated_cost_if_missed"] == "n/a — hypothetical message"


def test_hypothetical_anomaly_casefile_structure():
    feature_names = ["Amount", "Time", "V14"]
    model = _LinearModel([0.6, -0.2, -0.3], intercept=0.1, feature_names=feature_names)
    explainer = _make_explainer("anomaly", model, feature_names)
    scenario = load_scenarios("anomaly")[0]
    provenance = {"model": "IF", "group": "anomaly", "dataset_hash": "sha256:x",
                  "background_hash": "sha256:y", "feature_names_hash": "sha256:z",
                  "split": "val", "recorded_at": "2026-01-01T00:00:00Z"}
    casefile, _row = build_hypothetical_casefile(
        "anomaly", model, explainer, scenario, feature_names, {}, None,
        provenance, lambda sid, seed=42, persona=None: subject_generator(sid, seed=seed, persona="anomaly"),
        lambda cf: recommend("anomaly", cf), threshold=0.5,
    )
    assert casefile["case_type"] == "hypothetical"
    assert casefile["persona"] == "anomaly"
    assert casefile["y_true"] is None
    assert 0.0 <= casefile["score"] <= 1.0
    assert casefile["disagreement_type"] in ("novel", "sophisticated", "agree_high", "agree_low")
    rec = casefile["recommendation"]
    assert rec["action"] in {"block", "escalate", "review", "monitor", "clear"}


def test_deterministic_subjects():
    s1 = subject_generator("H-FRAUD-001", seed=42, persona="fraud")
    s2 = subject_generator("H-FRAUD-001", seed=42, persona="fraud")
    assert s1.to_dict() == s2.to_dict()
    assert s1.full_name == "Priya Nakamura"
    assert s1.city == "Porto"


def test_render_hypothetical_waterfall(tmp_path):
    feature_names = ["Amount", "V14", "V12"]
    model = _LinearModel([0.5, -0.4, -0.2], intercept=0.0, feature_names=feature_names)
    explainer = _make_explainer("fraud", model, feature_names)
    scenario = load_scenarios("fraud")[0]
    training_means = {"Amount": 100.0, "V14": 0.0, "V12": 0.0}
    provenance = {"model": "M", "group": "unscaled", "dataset_hash": "sha256:x",
                  "background_hash": "sha256:y", "feature_names_hash": "sha256:z",
                  "split": "val", "recorded_at": "2026-01-01T00:00:00Z"}
    casefile, row = build_hypothetical_casefile(
        "fraud", model, explainer, scenario, feature_names, training_means, None,
        provenance, lambda sid, seed=42, persona=None: subject_generator(sid, seed=seed, persona="fraud"),
        lambda cf: recommend("fraud", cf), threshold=0.5,
    )
    out = tmp_path / "hypo_waterfall.png"
    render_hypothetical_case_waterfall(explainer, row, casefile, out, top_k=3)
    assert out.exists()
    assert out.stat().st_size > 5000
