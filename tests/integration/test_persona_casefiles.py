"""Integration tests for persona case files (empirical + hypothetical) with recommendations."""
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from spam_fraud_detector.core.explainer import (
    ModelExplainer,
    build_casefile,
    build_hypothetical_casefile,
    render_case_waterfall,
    render_hypothetical_case_waterfall,
)
from spam_fraud_detector.core.fake_subjects import load_scenarios, subject_generator
from spam_fraud_detector.core.recommendations import recommend
from spam_fraud_detector.core.result_store import ResultStore, RunContext


@pytest.fixture()
def tiny_store(tmp_path):
    """A tiny ResultStore with one fraud model bundle persisted."""
    out_root = tmp_path / "outputs"
    ctx = RunContext(domain="fraud_detection", seed=42, git_commit="abc", package_versions={})
    store = ResultStore(out_root, ctx)
    X = pd.DataFrame({"Amount": [1.0, 5000.0, 12.0, 7200.0, 3.0],
                      "V14": [0.0, -3.0, 0.5, -3.1, 0.1],
                      "V12": [0.0, -1.0, 0.2, -1.0, 0.0]})
    y = np.array([0, 1, 0, 1, 0])
    model = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=1000))])
    model.fit(X, y)
    store.write_model(
        "LogisticRegression", "scaled", model,
        eval_val={"pr_auc": 0.9, "n": 5, "n_positive": 2},
        eval_test={"pr_auc": 0.85, "n": 5, "n_positive": 2},
        y_true_val=y, y_score_val=model.predict_proba(X)[:, 1],
        y_true_test=y, y_score_test=model.predict_proba(X)[:, 1],
        threshold=0.5, feature_columns=list(X.columns),
    )
    store.finalize()
    return out_root


def _load_meta(store_out, name, group):
    import joblib
    meta = json.loads((store_out / "models" / "fraud_detection" / "all" / f"{name}__{group}.meta.json").read_text())
    m = joblib.load(store_out / "models" / "fraud_detection" / "all" / f"{name}__{group}.joblib")
    return meta, m


def test_empirical_case_file_with_recommendation(tmp_path, tiny_store):
    meta, model = _load_meta(tiny_store, "LogisticRegression", "scaled")
    feature_cols = meta["feature_columns"]
    X = pd.DataFrame({"Amount": [7200.0], "V14": [-3.1], "V12": [-1.0]}, columns=feature_cols)
    score = float(model.predict_proba(X)[:, 1][0])
    explainer = ModelExplainer(model, X, feature_names=feature_cols, backend="shap")
    subject = {"subject_id": "F-0001", "full_name": "Priya Nakamura", "city": "Porto",
               "account_last4": "7361", "amount_eur": 7200.0}
    provenance = {"model": "LogisticRegression", "group": "scaled",
                  "dataset_hash": "sha256:x", "background_hash": "sha256:y",
                  "feature_names_hash": "sha256:z", "split": "val",
                  "row_id": 0, "recorded_at": "2026-01-01T00:00:00Z"}
    top = [{"name": "Amount", "value": 7200.0, "shap_value": 0.5, "direction": "increases_fraud_risk",
            "explanation": "high amount"}]
    casefile = build_casefile("fraud", model, X, 1, 1, score, float(meta["threshold"]),
                              "scaled", subject, top, provenance,
                              lambda cf: recommend("fraud", cf))
    assert casefile["case_type"] == "empirical"
    assert casefile["error_type"] == "true_positive"
    assert casefile["recommendation"]["action"] in {"block", "escalate", "review", "monitor", "clear"}
    assert casefile["recommendation"]["reviewer_notes_template"].count("[NAME]") >= 1
    assert casefile["recommendation"]["reviewer_notes_template"].count("[DATE]") >= 1

    out = tmp_path / "emp_waterfall.png"
    render_case_waterfall(explainer, X, casefile, out, top_k=2)
    assert out.exists() and out.stat().st_size > 5000


def test_hypothetical_case_file_with_recommendation(tmp_path, tiny_store):
    meta, model = _load_meta(tiny_store, "LogisticRegression", "scaled")
    feature_cols = meta["feature_columns"]
    scenario = load_scenarios("fraud")[0]
    training_means = {c: float(np.mean([1.0, 5000.0, 12.0, 7200.0, 3.0]
                                       if c == "Amount" else [0.0, -3.0, 0.5, -3.1, 0.1]
                                       if c == "V14" else [0.0, -1.0, 0.2, -1.0, 0.0]))
                      for c in feature_cols}
    provenance = {"model": "LogisticRegression", "group": "scaled",
                  "dataset_hash": "sha256:x", "background_hash": "sha256:y",
                  "feature_names_hash": "sha256:z", "split": "val",
                  "recorded_at": "2026-01-01T00:00:00Z"}
    casefile, row = build_hypothetical_casefile(
        "fraud", model, ModelExplainer(model, pd.DataFrame(
            [[0.0] * len(feature_cols)], columns=feature_cols),
            feature_names=feature_cols, backend="shap"),
        scenario, feature_cols, training_means, None, provenance,
        lambda sid, seed=42, persona=None: subject_generator(sid, seed=seed, persona="fraud"),
        lambda cf: recommend("fraud", cf), threshold=float(meta["threshold"]),
    )
    assert casefile["case_type"] == "hypothetical"
    assert casefile["scenario_id"] == scenario["scenario_id"]
    assert casefile["y_true"] is None
    assert casefile["recommendation"]["action"] == "review"
    assert casefile["recommendation"]["estimated_cost_if_missed"].startswith("n/a")

    out = tmp_path / "hypo_waterfall.png"
    render_hypothetical_case_waterfall(ModelExplainer(
        model, pd.DataFrame([[0.0] * len(feature_cols)], columns=feature_cols),
        feature_names=feature_cols, backend="shap"), row, casefile, out, top_k=3)
    assert out.exists() and out.stat().st_size > 5000


def test_load_scenarios_returns_all_personas():
    for persona in ("fraud", "spam", "anomaly"):
        scen = load_scenarios(persona)
        assert isinstance(scen, list) and len(scen) >= 1
        for s in scen:
            assert "scenario_id" in s and "description" in s
