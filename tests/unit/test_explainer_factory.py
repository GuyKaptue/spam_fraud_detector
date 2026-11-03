"""Tests for make_explainer factory."""
from __future__ import annotations

import numpy as np
import pytest

from spam_fraud_detector.core.explainer import (
    AnomalyExplainer,
    ExplainerError,
    ModelExplainer,
    TextExplainer,
    make_explainer,
)


def test_factory_routes_fraud():
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(random_state=42)
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=float)
    y = np.array([0, 0, 1, 1])
    model.fit(X, y)
    explainer = make_explainer(persona="fraud", model=model, X_background=X)
    assert isinstance(explainer, ModelExplainer)
    assert not isinstance(explainer, TextExplainer)
    assert not isinstance(explainer, AnomalyExplainer)


def test_factory_routes_spam():
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=float)
    y = np.array([0, 0, 1, 1])
    model.fit(X, y)
    vocab = ["a", "b"]
    explainer = make_explainer(persona="spam", model=model, X_background=X, vocabulary=vocab)
    assert isinstance(explainer, TextExplainer)


def test_factory_routes_anomaly():
    from sklearn.ensemble import IsolationForest
    model = IsolationForest(random_state=42)
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=float)
    model.fit(X)
    explainer = make_explainer(persona="anomaly", model=model, X_background=X)
    assert isinstance(explainer, AnomalyExplainer)


def test_factory_unknown_persona():
    with pytest.raises(ExplainerError, match="Unknown persona"):
        make_explainer(persona="unknown", model=None, X_background=None)


def test_factory_spam_requires_vocabulary():
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=float)
    y = np.array([0, 0, 1, 1])
    model.fit(X, y)
    with pytest.raises(ExplainerError, match="vocabulary"):
        make_explainer(persona="spam", model=model, X_background=X, vocabulary=None)