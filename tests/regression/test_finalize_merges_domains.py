import numpy as np
import pandas as pd

from spam_fraud_detector.core import ResultStore, RunContext
from spam_fraud_detector.core.utils import get_git_commit, get_package_versions


def _write_one(out, domain):
    ctx = RunContext(domain=domain, seed=1, git_commit=get_git_commit(),
                     package_versions=get_package_versions())
    store = ResultStore(out, ctx)
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    y = np.array([0, 0, 1, 1])
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit(X, y)
    store.write_model("M", "scaled", model, eval_val={}, y_true_val=y,
                      y_score_val=model.predict_proba(X)[:, 1])
    store.finalize()
    return store


def test_finalize_merges_domains(tmp_path):
    out = tmp_path / "outputs"
    _write_one(out, "spam_detection")
    _write_one(out, "fraud_detection")
    dim = pd.read_parquet(out / "dashboard" / "dim_explainability.parquet") if (out / "dashboard" / "dim_explainability.parquet").exists() else None
    # manifest should contain rows for both domains
    manifest = pd.read_parquet(out / "dashboard" / "manifest.parquet")
    assert set(manifest["domain"].unique()) >= {"spam_detection", "fraud_detection"}