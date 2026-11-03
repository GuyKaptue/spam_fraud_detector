import numpy as np

from spam_fraud_detector.core import ResultStore, RunContext, TopKSelector
from spam_fraud_detector.core.utils import get_git_commit, get_package_versions


def test_selector_uses_current_staging(tmp_path):
    out = tmp_path / "outputs"
    ctx = RunContext(domain="spam_detection", seed=1, git_commit=get_git_commit(),
                     package_versions=get_package_versions())
    store = ResultStore(out, ctx)
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    y = np.array([0, 0, 1, 1])
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit(X, y)
    store.write_model("FRESH", "scaled", model, eval_val={"pr_auc": 0.9, "n": 4, "n_positive": 2},
                      eval_test={"pr_auc": 0.85, "n": 4, "n_positive": 2},
                      y_true_val=y, y_score_val=model.predict_proba(X)[:, 1],
                      y_true_test=y, y_score_test=model.predict_proba(X)[:, 1])
    store._write_dashboard_tables()

    # stale staging dir that must NOT be picked
    stale = tmp_path / "outputs.staging.stale"
    (stale / "dashboard").mkdir(parents=True, exist_ok=True)
    import pandas as pd

    from spam_fraud_detector.core.utils import deterministic_parquet
    stale_manifest = pd.DataFrame([{"domain": "spam_detection", "model": "STALE",
                                    "group": "scaled", "skipped": False}])
    deterministic_parquet(stale_manifest, stale / "dashboard" / "manifest.parquet")
    stale_fact = pd.DataFrame([{"domain": "spam_detection", "model": "STALE",
                                "group": "scaled", "split": "test", "metric": "pr_auc",
                                "value": 0.1}])
    deterministic_parquet(stale_fact, stale / "dashboard" / "fact_metrics.parquet")

    sel = TopKSelector(out, "spam_detection", k=1)
    ranking = sel.select(store=store)
    assert any(r["model"] == "FRESH" for r in ranking)