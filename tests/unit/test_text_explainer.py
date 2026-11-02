"""Tests for TextExplainer."""
from __future__ import annotations

import numpy as np
import pytest

from spam_fraud_detector.core.explainer import TextExplainer, ExplainerError


def test_vocabulary_required():
    import numpy as np
    X = np.array([[1, 2], [3, 4]])
    with pytest.raises(ExplainerError, match="vocabulary"):
        TextExplainer(model=None, X_background=X, vocabulary=None)


def test_linear_explainer_selected():
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=float)
    y = np.array([0, 0, 1, 1])
    model.fit(X, y)
    vocab = ["feat0", "feat1"]
    explainer = TextExplainer(model=model, X_background=X, vocabulary=vocab)
    assert explainer.vocabulary == vocab
    assert explainer.feature_names == vocab


def test_top_k_mistakes_inherited():
    from spam_fraud_detector.core.explainer import top_k_mistakes
    y_true = np.array([0, 1, 0, 1, 0])
    y_score = np.array([0.9, 0.1, 0.8, 0.2, 0.7])
    fps, fns = top_k_mistakes(y_true, y_score, threshold=0.5, k=2)
    assert isinstance(fps, list)
    assert isinstance(fns, list)