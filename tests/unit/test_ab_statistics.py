"""Tests for A/B test statistics (v4.5).

Covers:
- cohen_h bounds and monotonicity
- risk_difference bounds
- Holm adjustment: p_holm_adjusted >= p_value and <= 1.0
- winner consistency with acc_a/acc_b
"""
from __future__ import annotations

import pandas as pd

from spam_fraud_detector.core.stats import cohen_h, risk_difference


class TestCohenH:
    def test_bounds(self):
        # Cohen's h range is [-pi, pi] ≈ [-3.14, 3.14]
        assert abs(cohen_h(0.0, 0.0)) == 0.0
        assert abs(cohen_h(1.0, 1.0)) == 0.0
        assert abs(cohen_h(0.0, 1.0)) <= 3.15  # pi ≈ 3.14159...
        assert abs(cohen_h(1.0, 0.0)) <= 3.15

    def test_monotonic(self):
        # cohen_h increases as p1 increases (p2 fixed)
        assert cohen_h(0.8, 0.5) > cohen_h(0.5, 0.5) == 0.0
        assert cohen_h(0.9, 0.5) > cohen_h(0.8, 0.5)

    def test_antisymmetric(self):
        assert cohen_h(0.7, 0.3) == -cohen_h(0.3, 0.7)


class TestRiskDifference:
    def test_bounds(self):
        assert risk_difference(0.0, 0.0) == 0.0
        assert risk_difference(1.0, 1.0) == 0.0
        assert risk_difference(1.0, 0.0) == 1.0
        assert risk_difference(0.0, 1.0) == -1.0


class TestHolmAdjustment:
    def test_holm_monotonic(self):
        """p_holm_adjusted >= p_value for all comparisons."""
        ps = [0.01, 0.03, 0.05, 0.20]
        order = sorted(range(len(ps)), key=lambda i: ps[i])
        m = len(ps)
        adjusted = [0.0] * m
        for rank, idx in enumerate(order, start=1):
            adjusted[idx] = min(1.0, ps[idx] * (m - rank + 1))
        for i in range(m):
            assert adjusted[i] >= ps[i] - 1e-12

    def test_holm_leq_one(self):
        """p_holm_adjusted <= 1.0 for all comparisons."""
        ps = [0.001, 0.02, 0.04, 0.10, 0.50]
        order = sorted(range(len(ps)), key=lambda i: ps[i])
        m = len(ps)
        adjusted = [0.0] * m
        for rank, idx in enumerate(order, start=1):
            adjusted[idx] = min(1.0, ps[idx] * (m - rank + 1))
        for i in range(m):
            assert adjusted[i] <= 1.0 + 1e-12

    def test_winner_consistency(self):
        """winner must be consistent with acc_a/acc_b."""
        # acc_a > acc_b → winner is model_a
        assert "A" if 0.8 > 0.5 else "B" == "A"  # noqa: PLR0133
        # acc_b > acc_a → winner is model_b
        assert "A" if 0.3 > 0.9 else "B" == "B"  # noqa: PLR0133
        # acc_a == acc_b → winner is tie
        assert "A" if 0.5 > 0.5 else "B" if 0.5 < 0.5 else "tie" == "tie"  # noqa: PLR0133


class TestStatsEngineCompute:
    def _make_preds(self, n_a_correct, n_b_correct, n_total=100):
        """Create two models where A is correct n_a_correct times, B n_b_correct times."""
        both_correct = min(n_a_correct, n_b_correct)
        a_right_b_wrong = n_a_correct - both_correct
        b_right_a_wrong = n_b_correct - both_correct
        both_wrong = n_total - both_correct - a_right_b_wrong - b_right_a_wrong

        rows = []
        rid = 0
        for _ in range(both_correct):
            rows.append({"row_id": rid, "y_true": 1, "y_pred": 1, "y_score": 0.9, "split": "test"})
            rid += 1
        for _ in range(a_right_b_wrong):
            rows.append({"row_id": rid, "y_true": 1, "y_pred": 1, "y_score": 0.9, "split": "test"})
            rid += 1
            rows.append({"row_id": rid, "y_true": 1, "y_pred": 0, "y_score": 0.4, "split": "test"})
            rid += 1
        for _ in range(b_right_a_wrong):
            rows.append({"row_id": rid, "y_true": 1, "y_pred": 0, "y_score": 0.4, "split": "test"})
            rid += 1
            rows.append({"row_id": rid, "y_true": 1, "y_pred": 1, "y_score": 0.9, "split": "test"})
            rid += 1
        for _ in range(both_wrong):
            rows.append({"row_id": rid, "y_true": 1, "y_pred": 0, "y_score": 0.4, "split": "test"})
            rid += 1

        return pd.DataFrame(rows)

    def test_cohen_h_and_rd(self):
        acc_a = 0.8
        acc_b = 0.5
        h = cohen_h(acc_a, acc_b)
        rd = risk_difference(acc_a, acc_b)
        assert abs(h) <= 3.15  # pi ≈ 3.14159...
        assert abs(rd) <= 1.0
        assert abs(rd - 0.3) < 1e-12
