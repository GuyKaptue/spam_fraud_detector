# type: ignore
"""Split management with test-access guard."""
from __future__ import annotations

import contextlib
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

from .exceptions import TestLeakError, SplitContractError
from .utils import deterministic_parquet


class _AuthState:
    allowed: bool = False
    run_id: Optional[str] = None
    caller: Optional[str] = None
    authorized_by: Optional[str] = None


_AUTH = _AuthState()


def _hash_index(idx) -> str:
    h = hashlib.sha256()
    for v in list(idx):
        h.update(str(v).encode("utf-8"))
    return "sha256:" + h.hexdigest()


def authorize_test_access(run_id: str = "", caller: str = "", authorized_by: str = ""):
    @contextlib.contextmanager
    def ctx() -> Iterator[None]:
        prev = (_AUTH.allowed, _AUTH.run_id, _AUTH.caller, _AUTH.authorized_by)
        _AUTH.allowed = True
        _AUTH.run_id = run_id
        _AUTH.caller = caller
        _AUTH.authorized_by = authorized_by
        try:
            yield
        finally:
            _AUTH.allowed, _AUTH.run_id, _AUTH.caller, _AUTH.authorized_by = prev
    return ctx()


class SplitManager:
    def __init__(self, domain: str, data_root: Any = "data", seed: int = 42,
                 split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15), **kwargs: Any) -> None:
        self.domain = domain
        self.data_root = Path(data_root)
        self.seed = seed
        self.split_ratios = tuple(split_ratios)
        self.processed_dir = self.data_root / "processed" / domain
        self.split_dir = self.processed_dir / "01_split"
        self.split_dir.mkdir(parents=True, exist_ok=True)
        self.split_path = self.split_dir / "split.json"
    def _stratified_split(self, df: pd.DataFrame) -> Dict[str, List[int]]:
        label_col = None
        for c in ("target", "label", "Class", "Prediction", "y"):
            if c in df.columns:
                label_col = c
                break
        if label_col is None:
            label_col = df.columns[-1]
        y = df[label_col].to_numpy()
        indices = np.arange(len(df))
        rng = np.random.default_rng(self.seed)
        train_idx: List[int] = []
        val_idx: List[int] = []
        test_idx: List[int] = []
        for cls in np.unique(y):
            cls_idx = indices[y == cls]
            rng.shuffle(cls_idx)
            n = len(cls_idx)
            n_train = int(round(n * self.split_ratios[0]))
            n_val = int(round(n * self.split_ratios[1]))
            train_idx.extend(cls_idx[:n_train].tolist())
            val_idx.extend(cls_idx[n_train:n_train + n_val].tolist())
            test_idx.extend(cls_idx[n_train + n_val:].tolist())
        return {"train": sorted(train_idx), "val": sorted(val_idx), "test": sorted(test_idx)}

    def split(self, df: pd.DataFrame) -> Dict[str, Any]:
        from datetime import datetime, timezone

        splits = self._stratified_split(df)
        label_col = None
        for c in ("target", "label", "Class", "Prediction", "y"):
            if c in df.columns:
                label_col = c
                break
        if label_col is None:
            label_col = df.columns[-1]
        y = df[label_col]

        def dist(idx):
            return {str(k): int(v) for k, v in y.iloc[idx].value_counts().to_dict().items()}

        manifest = {
            "seed": self.seed,
            "ratios": {"train": self.split_ratios[0], "val": self.split_ratios[1], "test": self.split_ratios[2]},
            "n_train": len(splits["train"]),
            "n_val": len(splits["val"]),
            "n_test": len(splits["test"]),
            "row_index_hashes": {
                "train": _hash_index(splits["train"]),
                "val": _hash_index(splits["val"]),
                "test": _hash_index(splits["test"]),
            },
            "label_distribution": {
                "train": dist(splits["train"]),
                "val": dist(splits["val"]),
                "test": dist(splits["test"]),
            },
            "created_at": datetime.now(timezone.utc).isoformat(),
            "data_layout_version": "v3.2",
        }
        self.split_dir.mkdir(parents=True, exist_ok=True)
        for name, idx in splits.items():
            out = self.processed_dir / "01_split" / f"{name}.parquet"
            out.parent.mkdir(parents=True, exist_ok=True)
            deterministic_parquet(df.iloc[idx], out)
        import json as _json

        with open(self.split_path, "w", encoding="utf-8") as f:
            _json.dump(manifest, f, sort_keys=True, indent=2)
            f.write("\n")
        return manifest

    create_split = split

    def load_manifest(self) -> Dict[str, Any]:
        if not self.split_path.exists():
            raise SplitContractError(f"missing split manifest: {self.split_path}")
        return json.loads(self.split_path.read_text())

    def load(self, split: str, caller: str = "", run_id: str = "") -> pd.DataFrame:
        return self.load_split(split, caller=caller, run_id=run_id)

    def load_split(self, split: str, caller: str = "", run_id: str = "", authorized_by: str = "") -> pd.DataFrame:
        if split not in ("train", "val", "test"):
            raise SplitContractError(f"unknown split: {split}")
        if split == "test" and not _AUTH.allowed:
            raise TestLeakError("unauthorized test split access")
        path = self.processed_dir / "01_split" / f"{split}.parquet"
        if not path.exists():
            raise SplitContractError(f"missing split artifact: {path}")
        if split == "test":
            import json as _json
            from datetime import datetime, timezone

            log_dir = Path("outputs") / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / "test_access.log"
            rec = {
                "run_id": run_id or _AUTH.run_id,
                "caller": caller or _AUTH.caller,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "authorized_by": authorized_by or _AUTH.authorized_by,
            }
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(_json.dumps(rec) + "\n")
            if not rec.get("authorized_by"):
                raise SplitContractError("missing authorized_by for test access")
        return pd.read_parquet(path)


def load_split(domain: str, split: str, data_root: Any = "data") -> pd.DataFrame:
    sm = SplitManager(domain=domain, data_root=data_root)
    return sm.load_split(split)

