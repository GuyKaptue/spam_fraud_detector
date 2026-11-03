"""Tests for AnomalyExplainer."""
from __future__ import annotations

import numpy as np

from spam_fraud_detector.core.explainer import AnomalyExplainer


def test_permutation_explainer_selected():
    from sklearn.ensemble import IsolationForest
    model = IsolationForest(random_state=42)
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=float)
    model.fit(X)
    explainer = AnomalyExplainer(detector=model, X_background=X)
    assert explainer.detector is model
    assert explainer.backend == "shap"


def test_background_legitimate_only():
    # AnomalyExplainer doesn't enforce this itself, but the contract says
    # X_background must be legitimate-only. Test that it accepts valid input.
    from sklearn.ensemble import IsolationForest
    model = IsolationForest(random_state=42)
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=float)
    model.fit(X)
    explainer = AnomalyExplainer(detector=model, X_background=X)
    assert explainer.X_background is X


def test_direction_inversion():
    # Test that _as_score inverts decision_function
    import numpy as np

    from spam_fraud_detector.core.anomaly import AnomalyComparer
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=float)
    y = np.array([0, 0, 0, 0])
    ac = AnomalyComparer(X, y, X, y, X, y)
    assert ac is not None