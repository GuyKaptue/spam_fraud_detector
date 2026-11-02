import numpy as np
from spam_fraud_detector.core import ResultStore, RunContext
from spam_fraud_detector.core.utils import get_git_commit, get_package_versions


def test_threshold_sweep_cost(tmp_path):
    out = tmp_path / "outputs"
    ctx = RunContext(domain="spam_detection", seed=1, git_commit=get_git_commit(), package_versions=get_package_versions())
    store = ResultStore(out, ctx)
    y = np.array([0,0,0,1,1])
    s = np.array([0.1,0.2,0.3,0.8,0.9])
    rows, star = store._threshold_sweep(y, s, cost_fn=1, cost_fp=1)
    assert "expected_cost" in rows[0]
    # symmetric (1,1) -> expected_cost = (fn+fp)/n
    for r in rows:
        assert abs(r["expected_cost"] - (r["fn"]+r["fp"])/len(y)) < 1e-9
    # star is argmin
    assert star == min(rows, key=lambda r: r["expected_cost"])["threshold"]