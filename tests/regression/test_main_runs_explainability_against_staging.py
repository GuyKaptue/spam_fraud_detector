import numpy as np

from spam_fraud_detector.core import ResultStore, RunContext
from spam_fraud_detector.core.utils import get_git_commit, get_package_versions
from spam_fraud_detector.main import SpamFraudDetector


def test_main_runs_explainability_against_staging(tmp_path):
    out = tmp_path / "outputs"
    ctx = RunContext(domain="spam_detection", seed=1, git_commit=get_git_commit(),
                     package_versions=get_package_versions())
    store = ResultStore(out, ctx)
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    y = np.array([0, 0, 1, 1])
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit(X, y)
    store.write_model("M", "scaled", model, eval_val={"pr_auc": 0.9, "n": 4, "n_positive": 2},
                      y_true_val=y, y_score_val=model.predict_proba(X)[:, 1])

    detector = SpamFraudDetector(task="spam", explainer="shap")
    detector.store = store
    detector._run_explainability()

    meta_path = store.staging / "models" / "spam_detection" / "all" / "M__scaled.meta.json"
    meta = __import__("json").loads(meta_path.read_text())
    assert meta.get("explainability", {}).get("persona") in ("spam", "fraud", "anomaly")