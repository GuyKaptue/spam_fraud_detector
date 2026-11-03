from pathlib import Path

import numpy as np
import pytest

from spam_fraud_detector.core import ResultStore, RunContext


@pytest.fixture
def store(tmp_path):
    out = tmp_path / "outputs"
    ctx = RunContext(domain="spam_detection", seed=1)
    return ResultStore(out, ctx), out


def test_write_model_creates_bundle_and_buffers(store):
    rs, _out = store
    y = np.array([0, 1, 0, 1])
    s = np.array([0.1, 0.9, 0.2, 0.8])
    rs.write_model("M", "scaled", object(), hyperparams={}, eval_val={"accuracy": 0.9, "tp": 1, "fp": 0, "tn": 2, "fn": 1, "n": 4, "n_positive": 2},
                   eval_test={"accuracy": 0.9, "tp": 1, "fp": 0, "tn": 2, "fn": 1, "n": 4, "n_positive": 2},
                   y_true_val=y, y_score_val=s, y_true_test=y, y_score_test=s, threshold=0.5,
                   confusion={"validation": {"tp": 1, "fp": 0, "tn": 2, "fn": 1}, "test": {"tp": 1, "fp": 0, "tn": 2, "fn": 1}})
    assert len(rs._metrics) == 2  # accuracy * 2 splits
    rs.write_skipped([{"model": "Bad", "group": "scaled", "reason": "negative inputs"}])
    sel = [{"rank": i + 1, "model": "M", "group": "scaled", "pr_auc": 0.9, "f1": 0.8, "recall_at_prec_0.90": 1.0, "threshold": 0.5, "artifact": "x"} for i in range(3)]
    rs.write_selection(sel)
    rs.write_stats({"comparisons": [{"model_a": "M", "model_b": "N", "test": "mcnemar", "statistic": 1.0, "p_value": 0.1, "p_holm_adjusted": 0.2, "n": 4, "winner": "M"}]}, "# summary")
    p = rs.finalize()
    assert (p / "dashboard" / "index.json").exists()
    assert (p / "CONTRACT.json").exists()
    # bundle files
    base = p / "models" / "spam_detection" / "all" / "M__scaled"
    for ext in [".joblib", ".meta.json", ".eval.json", ".curve.json", ".confusion.json"]:
        assert (Path(str(base) + ext)).exists(), ext
    # parquet/csv parity
    for pq in (p / "dashboard").glob("*.parquet"):
        assert pq.with_suffix(".csv").exists()


def test_atomicity(store, monkeypatch):
    rs, out = store
    y = np.array([0, 1, 0, 1])
    s = np.array([0.1, 0.9, 0.2, 0.8])
    rs.write_model("M", "scaled", object(), hyperparams={}, eval_val={"accuracy": 0.9, "tp": 1, "fp": 0, "tn": 2, "fn": 1, "n": 4, "n_positive": 2},
                   eval_test={"accuracy": 0.9, "tp": 1, "fp": 0, "tn": 2, "fn": 1, "n": 4, "n_positive": 2},
                   y_true_val=y, y_score_val=s, y_true_test=y, y_score_test=s, threshold=0.5,
                   confusion={"validation": {"tp": 1, "fp": 0, "tn": 2, "fn": 1}, "test": {"tp": 1, "fp": 0, "tn": 2, "fn": 1}})
    # monkeypatch os.replace to fail
    import os
    orig = os.replace
    def boom(*a, **k):
        raise RuntimeError("boom")
    monkeypatch.setattr(os, "replace", boom)
    try:
        rs.finalize()
    except Exception:  # noqa: BLE001, S110
        pass
    monkeypatch.setattr(os, "replace", orig)
    # outputs dir should not exist (staging cleaned)
    assert not out.exists() or not (out / "dashboard").exists()
