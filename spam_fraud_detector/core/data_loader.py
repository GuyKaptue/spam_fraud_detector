# type: ignore
# spam_fraud_detector/core/data_loader.py
from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import kagglehub
except Exception as e:  # pragma: no cover  # noqa: BLE001, F841
    kagglehub = None

from .exceptions import DatasetNotFoundError
from .utils import deterministic_parquet, get_dataset_config


class KaggleDataLoader:
    def __init__(
        self,
        task: str = "spam",
        adapter: Any = None,
        save_dir: str = "datasets",
        data_root: str | Path = "data",
        download: bool = False,
        force_rebuild: bool = False,
        **kwargs: Any,
    ) -> None:
        config = get_dataset_config(task)
        self.task = task
        self.dataset_name = config["dataset_name"]
        self.file_path = config["file_path"]
        if adapter is None:
            try:
                from kagglehub import KaggleDatasetAdapter

                adapter = getattr(KaggleDatasetAdapter, "PANDAS", None)
            except Exception as e:  # pragma: no cover  # noqa: BLE001
                    print(f"Download failed: {e}")
        self.adapter = adapter
        self.save_dir = save_dir
        self.data_root = Path(data_root)
        self.raw_dir = self.data_root / "raw" / task
        self.processed_dir = self.data_root / "processed" / task
        self.download = bool(download)
        self.force_rebuild = bool(force_rebuild)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def has_credentials(self) -> bool:
        if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
            return True
        return (Path.home() / ".kaggle" / "kaggle.json").exists()

    def raw_files(self) -> list[Path]:
        if not self.raw_dir.exists():
            return []
        return sorted(p for p in self.raw_dir.rglob("*") if p.is_file())

    def has_raw(self) -> bool:
        return len(self.raw_files()) > 0

    def expected_raw_paths(self) -> list[str]:
        return [str(self.raw_dir / self.file_path), str(self.raw_dir)]

    def download_dataset(self) -> str:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        if kagglehub is not None:
            try:
                path = kagglehub.dataset_download(self.dataset_name, path=str(self.raw_dir))
                print(f"Dataset downloaded to: {path}")
                return str(path) if path else str(self.raw_dir)
            except TypeError:
                try:
                    path = kagglehub.dataset_download(self.dataset_name)
                    print(f"Dataset downloaded to: {path}")
                    return str(path) if path else str(self.raw_dir)
                except Exception as exc:  # noqa: BLE001
                    print(f"Download failed: {exc}")
            try:
                subprocess.run(
                    ["kaggle", "datasets", "download", "-d", self.dataset_name, "-p", str(self.raw_dir), "--unzip"],
                    check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
            except Exception as exc:  # pragma: no cover  # noqa: BLE001
                print(f"Download failed: {exc}")
        return str(self.raw_dir)

    def download_if_missing(self, download: bool | None = None) -> Path:
        if download is None:
            download = self.download
        if self.has_raw():
            return self.raw_dir
        if download:
            self.download_dataset()
            if self.has_raw():
                return self.raw_dir
        raise DatasetNotFoundError(
            "Missing raw dataset files. Expected one of: " + ", ".join(self.expected_raw_paths())
        )

    def ensure_raw(self, download: bool | None = None) -> Path:
        return self.download_if_missing(download=download)
    def _extract_archives(self) -> None:
        import zipfile

        for archive in sorted(self.raw_dir.rglob("*.zip")):
            try:
                with zipfile.ZipFile(archive) as zf:
                    zf.extractall(archive.parent)
            except Exception:  # noqa: BLE001,S112
                continue

    def _parse_raw(self) -> None:
        self._extract_archives()
        raws = self.raw_files()
        if not raws:
            raise DatasetNotFoundError(
                "Missing raw dataset files. Expected one of: " + ", ".join(self.expected_raw_paths())
            )
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        for src in raws:
            if src.suffix.lower() not in {".csv", ".parquet", ".xlsx", ".json"}:
                continue
            try:
                if src.suffix.lower() == ".csv":
                    df = pd.read_csv(src)
                elif src.suffix.lower() == ".parquet":
                    df = pd.read_parquet(src)
                elif src.suffix.lower() == ".xlsx":
                    df = pd.read_excel(src)
                else:
                    df = pd.read_json(src)
            except Exception:  # noqa: BLE001,S112
                continue
            try:
                df = df.drop_duplicates().reset_index(drop=True)
            except Exception:  # noqa: BLE001,S110
                pass
            stem = src.stem
            out_parquet = self.processed_dir / f"{stem}.parquet"
            out_csv = self.processed_dir / f"{stem}.csv"
            deterministic_parquet(df, out_parquet)
            df.to_csv(out_csv, index=False)
        processed_files = sorted(self.processed_dir.glob("*.parquet"))
        if processed_files:
            frames = [pd.read_parquet(p) for p in processed_files]
            combined = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
            deterministic_parquet(combined, self.processed_dir / "data.parquet")
            combined.to_csv(self.processed_dir / "data.csv", index=False)

    parse = _parse_raw
    parse_raw = _parse_raw
    _parse = _parse_raw

    def build_processed(self, force: bool = False) -> Path:
        processed_data_files = [p for p in self.processed_dir.iterdir() if p.is_file() and p.suffix.lower() in {".csv", ".parquet", ".xlsx", ".json"}]
        if processed_data_files and not (force or self.force_rebuild):
            return self.processed_dir
        self._parse_raw()
        return self.processed_dir

    def process(self, force: bool = False) -> Path:
        return self.build_processed(force=force)

    def prepare(self, force: bool = False) -> Path:
        return self.build_processed(force=force)

    def build(self, force: bool = False) -> Path:
        return self.build_processed(force=force)

    @property
    def dataset_hash(self) -> str:
        files = sorted(p for p in self.processed_dir.rglob("*") if p.is_file())
        h = hashlib.sha256()
        for p in files:
            h.update(str(p.relative_to(self.processed_dir)).encode("utf-8"))
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    h.update(chunk)
        return "sha256:" + h.hexdigest()

    def compute_dataset_hash(self) -> str:
        return self.dataset_hash

    def get_dataset_hash(self) -> str:
        return self.dataset_hash

    def layer_path(self, domain: str, layer: str) -> Path:
        return self.data_root / "processed" / domain / layer

    def load_raw(self, domain: str | None = None) -> pd.DataFrame:
        d = domain or self.task
        loader = KaggleDataLoader(task=d, data_root=self.data_root, download=self.download, force_rebuild=self.force_rebuild)
        files = loader.raw_files()
        if not files:
            raise DatasetNotFoundError("missing raw files")
        # read first tabular
        for f in files:
            if f.suffix.lower() == ".csv":
                return pd.read_csv(f)
            if f.suffix.lower() == ".parquet":
                return pd.read_parquet(f)
        raise DatasetNotFoundError("no readable raw files")

    def load_layered(self, domain: str, layer: str) -> pd.DataFrame:
        d = domain or self.task
        base = self.data_root / "processed" / d / layer
        # find parquet/csv
        parquets = sorted(base.glob("*.parquet"))
        if parquets:
            return pd.read_parquet(parquets[0])
        csvs = sorted(base.glob("*.csv"))
        if csvs:
            return pd.read_csv(csvs[0])
        raise DatasetNotFoundError(f"missing layered artifact: {base}")

    def load_split(self, domain: str, split: str) -> pd.DataFrame:
        import warnings
        warnings.warn(
            "KaggleDataLoader.load_split is deprecated; use SplitManager(domain).load_split(...)",
            DeprecationWarning,
            stacklevel=2,
        )
        from .split import SplitManager

        d = domain or self.task
        sm = SplitManager(domain=d, data_root=self.data_root)
        return sm.load_split(split)

    def load_processed(self) -> pd.DataFrame:
        self.download_if_missing(download=self.download)
        self.build_processed(force=self.force_rebuild)
        preferred = self.processed_dir / "data.parquet"
        if preferred.exists():
            return pd.read_parquet(preferred)
        parquets = sorted(self.processed_dir.glob("*.parquet"))
        if parquets:
            frames = [pd.read_parquet(p) for p in parquets]
            return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
        preferred_csv = self.processed_dir / "data.csv"
        if preferred_csv.exists():
            return pd.read_csv(preferred_csv)
        csvs = sorted(self.processed_dir.glob("*.csv"))
        if csvs:
            frames = [pd.read_csv(p) for p in csvs]
            return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
        raise DatasetNotFoundError(
            "No processed dataset files. Expected one of: " + ", ".join(self.expected_raw_paths())
        )

    def load(self) -> pd.DataFrame:
        return self.load_processed()

    def load_dataset(self) -> pd.DataFrame:
        return self.load_processed()

    def load_data(self) -> pd.DataFrame:
        return self.load_processed()

    def download_load_and_save(self, save_as: str = "csv") -> pd.DataFrame:
        self.download_if_missing(download=True)
        data = self.load_processed()
        return data


def load_dataset(task: str = "fraud") -> pd.DataFrame:
    loader = KaggleDataLoader(task=task)
    return loader.load_dataset()
