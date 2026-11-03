# type: ignore
"""Feature engineering."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler


class FeatureEngineer:
    def __init__(self, domain: str, data_root: Any = "data", **kwargs: Any) -> None:
        self.domain = domain
        self.data_root = Path(data_root)
        self.processed_dir = self.data_root / "processed" / domain
        self.feature_dir = self.processed_dir / "06_features"
        self.feature_dir.mkdir(parents=True, exist_ok=True)
        self.encoder_path = self.feature_dir / "feature_encoder.joblib"
        self.scaler = StandardScaler()

    def fit_transform(self, train_df: pd.DataFrame, val_df: pd.DataFrame = None) -> Any:
        numeric = train_df.select_dtypes(include="number")
        self.scaler.fit(numeric)
        joblib.dump(self.scaler, self.encoder_path)
        tr = self.scaler.transform(numeric)
        out = pd.DataFrame(tr, columns=numeric.columns, index=train_df.index)
        return out

    def transform(self, df: pd.DataFrame) -> Any:
        if self.encoder_path.exists():
            sc = joblib.load(self.encoder_path)
        else:
            sc = self.scaler
        numeric = df.select_dtypes(include="number")
        tr = sc.transform(numeric)
        return pd.DataFrame(tr, columns=numeric.columns, index=df.index)

    def run(self, train_df: pd.DataFrame, val_df: pd.DataFrame = None, test_df: pd.DataFrame = None) -> Any:
        self.fit_transform(train_df, val_df)
        return {"train": train_df, "val": val_df, "test": test_df}
