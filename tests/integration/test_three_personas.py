"""Integration test: three personas produce artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def test_three_personas():
    # This test requires outputs to exist; skip if not
    outputs = REPO / "outputs"
    if not outputs.exists():
        pytest.skip("outputs/ not present; run make pipeline first")
    # Check dim_explainability exists
    dim_path = outputs / "dashboard" / "dim_explainability.parquet"
    if dim_path.exists():
        import pandas as pd
        df = pd.read_parquet(dim_path)
        assert "fraud" in df["persona"].values
        assert "spam" in df["persona"].values
        assert "anomaly" in df["persona"].values
    # Check at least one meta.json has explainability with persona
    meta_files = list((outputs / "models").rglob("*.meta.json"))
    if meta_files:
        found_persona = False
        for mf in meta_files:
            meta = json.loads(mf.read_text())
            expl = meta.get("explainability")
            if expl and expl.get("persona"):
                found_persona = True
                break
        assert found_persona, "no meta.json with persona found"