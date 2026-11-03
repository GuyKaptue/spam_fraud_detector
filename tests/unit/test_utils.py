"""Unit tests for utils."""
from spam_fraud_detector.core.utils import format_metrics, set_seed


def test_set_seed_returns_generator():
    g = set_seed(42)
    assert g is not None


def test_format_metrics():
    m = format_metrics({"a": 1.23456, "b": "x"})
    assert "a" in m
    assert "b" not in m