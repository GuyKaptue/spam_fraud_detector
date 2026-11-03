# type: ignore
"""Data quality gate."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .exceptions import DataQualityError


class QualityGate:
    def __init__(self, domain: str, data_root: Any = "data", thresholds: Any = None, **kwargs: Any) -> None:
        self.domain = domain
        self.data_root = Path(data_root)
        self.processed_dir = self.data_root / "processed" / domain
        self.quality_dir = self.processed_dir / "05_quality"
        self.quality_dir.mkdir(parents=True, exist_ok=True)
        self.thresholds = thresholds or {"max_null_pct": 0.5, "max_dupe_pct": 0.5}

    def evaluate(self, df: pd.DataFrame) -> dict[str, Any]:
        null_pct = float(df.isna().mean().mean()) if len(df) else 0.0
        dupe_pct = float(df.duplicated().mean()) if len(df) else 0.0
        report = {
            "null_pct": null_pct,
            "dupe_pct": dupe_pct,
            "n_rows": len(df),
            "n_cols": int(df.shape[1]),
        }
        passed = null_pct <= self.thresholds["max_null_pct"] and dupe_pct <= self.thresholds["max_dupe_pct"]
        gate = {"pass": bool(passed), "blocking": True}
        self.quality_dir.mkdir(parents=True, exist_ok=True)
        (self.quality_dir / "quality_report.json").write_text(__import__("json").dumps(report, sort_keys=True, indent=2) + "\n")
        (self.quality_dir / "gate.json").write_text(__import__("json").dumps(gate, sort_keys=True, indent=2) + "\n")
        if not passed:
            raise DataQualityError("quality gate failed")
        return {"report": report, "gate": gate}

    def run(self, df: pd.DataFrame) -> dict[str, Any]:
        return self.evaluate(df)
