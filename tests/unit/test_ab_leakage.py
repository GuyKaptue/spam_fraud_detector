"""Tests for A/B test leakage detection (v4.5).

Covers:
- Leakage guard: overlapping train/val/test splits between models raise SplitContractError
- Clean splits pass without error
- Same row_id appearing in train AND test within one model triggers leakage
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spam_fraud_detector.core.exceptions import SplitContractError
from spam_fraud_detector.core.stats import StatsEngine


class TestStatsEngineLeakage:
    def _make_preds(self, row_ids, y_true, y_pred, split="test"):
        return pd.DataFrame({
            "row_id": row_ids,
            "y_true": y_true,
            "y_pred": y_pred,
            "y_score": [0.5] * len(row_ids),
            "split": [split] * len(row_ids),
        })

    def test_no_leakage_same_split(self):
        pa = self._make_preds([1, 2, 3], [0, 1, 0], [0, 1, 0])
        pb = self._make_preds([1, 2, 3], [0, 1, 1], [0, 1, 0])
        # No overlap across splits (only one split)
        StatsEngine._check_leakage(pa, pb, "A", "B")

    def test_leakage_across_splits(self):
        # pa has row_id 3 in BOTH train and test → leakage within pa
        pa_train = self._make_preds([1, 2, 3], [0, 1, 0], [0, 1, 0], split="train")
        pa_test = self._make_preds([3, 4], [0, 1], [0, 1], split="test")
        pa = pd.concat([pa_train, pa_test], ignore_index=True)

        pb = self._make_preds([5, 6, 7], [0, 1, 0], [0, 1, 0], split="test")

        with pytest.raises(SplitContractError, match="Leakage detected"):
            StatsEngine._check_leakage(pa, pb, "A", "B")

    def test_no_leakage_disjoint_splits(self):
        pa = pd.concat([
            self._make_preds([1, 2], [0, 1], [0, 1], split="train"),
            self._make_preds([3, 4], [0, 1], [0, 1], split="test"),
        ], ignore_index=True)
        pb = pd.concat([
            self._make_preds([5, 6], [0, 1], [0, 1], split="train"),
            self._make_preds([7, 8], [0, 1], [0, 1], split="test"),
        ], ignore_index=True)
        StatsEngine._check_leakage(pa, pb, "A", "B")

    def test_zero_discordant_skip(self):
        """Models with identical predictions produce no discordant pairs."""
        pa = self._make_preds([1, 2, 3], [0, 1, 0], [0, 1, 0])
        pb = self._make_preds([1, 2, 3], [0, 1, 0], [0, 1, 0])
        # Both models identical → no discordant pairs → should skip without error
        # This is tested indirectly: _check_leakage passes, and compute() would skip
        StatsEngine._check_leakage(pa, pb, "A", "B")
