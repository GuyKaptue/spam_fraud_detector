"""Tests for deterministic parquet writing."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from spam_fraud_detector.core.utils import deterministic_parquet, deterministic_parquet_bytes


def test_byte_identical_writes(tmp_path: Path) -> None:
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    p1 = tmp_path / "a.parquet"
    p2 = tmp_path / "b.parquet"
    deterministic_parquet(df, p1)
    deterministic_parquet(df, p2)
    assert p1.read_bytes() == p2.read_bytes()


def test_bytes_function(tmp_path: Path) -> None:
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    b1 = deterministic_parquet_bytes(df)
    b2 = deterministic_parquet_bytes(df)
    assert b1 == b2


def test_atomicity(tmp_path: Path, monkeypatch) -> None:
    df = pd.DataFrame({"a": [1, 2, 3]})
    path = tmp_path / "out.parquet"
    original_replace = os.replace
    def boom(*a, **k):
        raise RuntimeError("boom")
    monkeypatch.setattr(os, "replace", boom)
    try:
        deterministic_parquet(df, path)
    except RuntimeError:
        pass
    monkeypatch.setattr(os, "replace", original_replace)
    assert not path.exists() or path.stat().st_size == 0


def test_csv_roundtrip(tmp_path: Path) -> None:
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    path = tmp_path / "rt.parquet"
    deterministic_parquet(df, path)
    loaded = pd.read_parquet(path)
    pd.testing.assert_frame_equal(df, loaded)