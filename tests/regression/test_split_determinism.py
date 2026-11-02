"""Regression test: split determinism across column reorders."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from spam_fraud_detector.core.split import SplitManager


@pytest.fixture()
def tmp_df(tmp_path: Path) -> Path:
    df = pd.DataFrame({"a": list(range(200)), "b": list(range(200, 400)), "y": [0, 1] * 100})
    p = tmp_path / "data.parquet"
    df.to_parquet(p, index=False)
    return p


def test_split_determinism(tmp_path: Path) -> None:
    df1 = pd.DataFrame({"a": list(range(200)), "b": list(range(200, 400)), "y": [0, 1] * 100})
    df2 = pd.DataFrame({"b": list(range(200, 400)), "a": list(range(200)), "y": [0, 1] * 100})

    dir1 = tmp_path / "run1"
    dir2 = tmp_path / "run2"
    dir1.mkdir()
    dir2.mkdir()

    sm1 = SplitManager(domain="spam", data_root=dir1, seed=42)
    sm2 = SplitManager(domain="spam", data_root=dir2, seed=42)

    sm1.split(df1)
    sm2.split(df2)

    m1 = json.loads((dir1 / "processed" / "spam" / "01_split" / "split.json").read_text())
    m2 = json.loads((dir2 / "processed" / "spam" / "01_split" / "split.json").read_text())

    # normalize timestamps
    m1.pop("created_at", None)
    m2.pop("created_at", None)

    assert m1 == m2, f"split manifests differ:\n{m1}\n{m2}"