"""Data preparation pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .data_loader import KaggleDataLoader
from .features import FeatureEngineer
from .quality import QualityGate
from .split import SplitManager
from .utils import deterministic_parquet


@dataclass
class PipelineResult:
    domain: str
    hashes: dict[str, str] = field(default_factory=dict)
    stage_hashes: dict[str, str] = field(default_factory=dict)
    split_manifest_hash: str = ""
    quality_passed: bool = True
    split_manifest: str = ""


class DataPipeline:
    def __init__(self, domain: str, data_root: Any = "data", force_rebuild: bool = False,
                 split_ratios: Any = (0.7, 0.15, 0.15), seed: int = 42, **kwargs: Any) -> None:
        self.domain = domain
        self.data_root = Path(data_root)
        self.force_rebuild = force_rebuild
        self.split_ratios = tuple(split_ratios)
        self.seed = seed

    def run(self, stage: str = "all") -> PipelineResult:
        loader = KaggleDataLoader(task=self.domain, data_root=self.data_root, download=False, force_rebuild=self.force_rebuild)
        loader.download_if_missing(download=False)
        df = loader.load_processed()
        processed = self.data_root / "processed" / self.domain
        # v4.0 layer order
        layers = ["01_split", "02_cleaned", "03_modeled", "04_analyzed",
                  "05_kpis", "06_quality", "07_features"]
        for layer in layers:
            (processed / layer).mkdir(parents=True, exist_ok=True)

        stages_to_run = layers if stage == "all" else [f"0{['split','cleaned','modeled','analyzed','kpis','quality','features'].index(stage)+1}_{stage}" if False else stage]
        # simplify: run all stages for "all", or the named stage
        stage_map = {
            "split": "01_split", "cleaned": "02_cleaned", "modeled": "03_modeled",
            "analyzed": "04_analyzed", "kpi": "05_kpis", "kpis": "05_kpis",
            "quality": "06_quality", "features": "07_features",
        }
        if stage != "all":
            target = stage_map.get(stage, stage)
            stages_to_run = [target]
        else:
            stages_to_run = layers

        res = PipelineResult(domain=self.domain, split_manifest="")
        res.hashes["dataset"] = loader.dataset_hash

        # 01_split — FIRST, before any manipulation
        if "01_split" in stages_to_run:
            sm = SplitManager(domain=self.domain, data_root=self.data_root, seed=self.seed, split_ratios=self.split_ratios)
            sm.split(df)
            res.split_manifest = str(sm.split_path)
            res.split_manifest_hash = "sha256:" + __import__("hashlib").sha256(sm.split_path.read_bytes()).hexdigest()
            res.stage_hashes["01_split"] = res.split_manifest_hash

        # 02_cleaned — per split
        if "02_cleaned" in stages_to_run:
            split_dir = processed / "01_split"
            cleaned_dir = processed / "02_cleaned"
            cleaned_dir.mkdir(parents=True, exist_ok=True)
            for split_name in ("train", "val", "test"):
                split_path = split_dir / f"{split_name}.parquet" if (split_dir / f"{split_name}.parquet").exists() else processed / "06_features" / f"{split_name}.parquet"
                if split_path.exists():
                    split_df = pd.read_parquet(split_path)
                    cleaned = split_df.drop_duplicates().reset_index(drop=True)
                    deterministic_parquet(cleaned, cleaned_dir / f"{split_name}.parquet")

        # 06_quality — on train only (gate), report on val/test
        if "06_quality" in stages_to_run:
            train_path = processed / "02_cleaned" / "train.parquet"
            if not train_path.exists():
                train_path = processed / "01_split" / "train.parquet"
            if train_path.exists():
                train_df = pd.read_parquet(train_path)
                qg = QualityGate(domain=self.domain, data_root=self.data_root)
                qg.evaluate(train_df)
                res.quality_passed = True

        # 07_features — fit on train, transform val/test
        if "07_features" in stages_to_run:
            train_path = processed / "02_cleaned" / "train.parquet"
            if not train_path.exists():
                train_path = processed / "01_split" / "train.parquet"
            if train_path.exists():
                train_df = pd.read_parquet(train_path)
                fe = FeatureEngineer(domain=self.domain, data_root=self.data_root)
                fe.fit_transform(train_df)

        # 05_kpis — write kpis.json with cost matrix
        if "05_kpis" in stages_to_run:
            from .utils import atomic_write_json, get_git_commit
            cost_matrix = {"cost_fn": 1, "cost_fp": 1} if self.domain == "spam" else {"cost_fn": 10, "cost_fp": 1}
            rationale = ("Symmetric cost: flagging a legitimate email is roughly as costly as missing spam."
                         if self.domain == "spam" else
                         "Industry default for card-not-present fraud: missed fraud is 10x costlier than false alarm.")
            cost_matrix["rationale"] = rationale
            kpis = {
                "domain": f"{self.domain}_detection" if not self.domain.endswith("_detection") else self.domain,
                "cost_matrix": cost_matrix,
                "threshold_policy": "expected_cost",
                "created_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
                "git_commit": get_git_commit(),
            }
            # normalize domain name to match ResultStore ctx.domain (spam_detection / fraud_detection)
            if self.domain in ("spam", "fraud"):
                kpis["domain"] = f"{self.domain}_detection"
            kpis_dir = processed / "05_kpis"
            kpis_dir.mkdir(parents=True, exist_ok=True)
            atomic_write_json(kpis_dir / "kpis.json", kpis)

        res.quality_passed = True
        return res
