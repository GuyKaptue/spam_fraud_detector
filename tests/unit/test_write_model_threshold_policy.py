import json
import numpy as np
from spam_fraud_detector.core import ResultStore, RunContext
from spam_fraud_detector.core.utils import get_git_commit, get_package_versions


def test_write_model_threshold_policy(tmp_path):
    out = tmp_path / "outputs"
    ctx = RunContext(domain="spam_detection", seed=1, git_commit=get_git_commit(), package_versions=get_package_versions())
    store = ResultStore(out, ctx)
    X = np.array([[1.,2.],[3.,4.],[5.,6.],[7.,8.]])
    y = np.array([0,0,1,1])
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(); m.fit(X,y)
    probs = m.predict_proba(X)[:,1]
    store.write_model("M","scaled",m, eval_val={"pr_auc":0.9,"n":4,"n_positive":2}, eval_test={"pr_auc":0.85,"n":4,"n_positive":2},
                      y_true_val=y, y_score_val=probs, y_true_test=y, y_score_test=probs,
                      threshold_policy="fixed:0.5")
    meta = json.loads((store.staging/"models"/"spam_detection"/"all"/"M__scaled.meta.json").read_text())
    assert meta["threshold"] == 0.5
    assert meta["threshold_policy"] == "fixed"