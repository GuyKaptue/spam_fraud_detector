# type: ignore
# spam_fraud_detector/core/utils.py
import os
import numpy as np  
import pandas as pd  

# Predefined dataset configurations
DATASETS = {
    "spam": {
        "dataset_name": "balaka18/email-spam-classification-dataset-csv",
        "file_path": "emails.csv"
    },
    "fraud": {
        "dataset_name": "mlg-ulb/creditcardfraud",
        "file_path": "creditcard.csv"
    }
}

def get_dataset_config(task: str) -> dict:
    """Return dataset configuration for a given task name ('spam' or 'fraud')."""
    if task not in DATASETS:
        raise ValueError(f"Unsupported task: {task}. Choose from {list(DATASETS.keys())}")
    return DATASETS[task]

def ensure_dir(path: str):
    """Ensure a directory exists. Handles both file paths and directory paths."""
    dir_path = path if os.path.isdir(path) or path.endswith("/") else os.path.dirname(path)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path)


def format_metrics(metrics: dict, round_digits: int = 4):
    """Round and format only scalar metrics."""
    return {
        k: round(v, round_digits)
        for k, v in metrics.items()
        if isinstance(v, (int, float))
    }

def summarize_class_distribution(y: pd.Series):
    """Return a summary of class counts and imbalance ratio."""
    counts = y.value_counts()
    ratio = counts.min() / counts.max()
    return {
        "class_counts": counts.to_dict(),
        "imbalance_ratio": round(ratio, 4)
    }

def safe_model_name(model) -> str:
    """Return a clean model name for logging or filenames."""
    return model.__class__.__name__.replace("Classifier", "")

def save_dataframe(data: pd.DataFrame, path: str, format: str = "csv") -> str:
    """Save a DataFrame to CSV or Excel format."""
    try:
        ensure_dir(path)
        if format == "csv":
            data.to_csv(path, index=False)
        elif format in ["xlsx", "sheet", "excel"]:
            data.to_excel(path, index=False)
        else:
            raise ValueError("Unsupported format. Use 'csv' or 'xlsx'.")
        print(f"Data saved to: {path}")
        return path
    except Exception as e:
        print(f"Saving failed: {e}")
        return ""
    
def load_dataset(task: str="fraud") -> pd.DataFrame:
    """Load the dataset for the given task ('spam' or 'fraud') from Kaggle."""
    config = get_dataset_config(task)
    dataset_dir = os.path.join("datasets", task)
    file_path = os.path.join(dataset_dir, config["file_path"])

    # Ensure the dataset file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Dataset file not found: {file_path}\n"
            f"Make sure you've downloaded it from Kaggle using:\n"
            f"!kaggle datasets download -d {config['dataset_name']} -p {dataset_dir} --unzip"
        )

    # Load the dataset
    df = pd.read_csv(file_path)
    print(f"{task.capitalize()} dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df
import json
import hashlib
import subprocess
import sys
from pathlib import Path
from typing import Any

def set_seed(seed: int = 42) -> np.random.Generator:
    """Set global seeds and return a numpy Generator."""
    np.random.seed(seed)
    import random
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:
        pass
    return np.random.default_rng(seed)

def get_git_commit() -> str:
    """Return current git commit hash or 'unknown'."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        return out or "unknown"
    except Exception:
        return "unknown"

def get_package_versions() -> dict:
    """Return versions of key packages."""
    import importlib
    pkgs = ["numpy", "pandas", "scikit-learn", "joblib", "pyarrow", "scipy"]
    versions = {}
    for p in pkgs:
        try:
            mod = importlib.import_module(p.replace("-", "_"))
            versions[p] = getattr(mod, "__version__", "unknown")
        except Exception:
            versions[p] = "not_installed"
    versions["python"] = sys.version.split()[0]
    return versions

def hash_file(path) -> str:
    """Return sha256 hexdigest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()

def atomic_write_json(path, obj: Any) -> None:
    """Write JSON atomically via temp file + os.replace."""
    path = Path(path)
    ensure_dir(str(path))
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, sort_keys=True, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)

def atomic_write_text(path, text: str) -> None:
    path = Path(path)
    ensure_dir(str(path))
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Sort columns by name so column order cannot cause byte divergence."""
    if len(df.columns) > 1:
        df = df.reindex(sorted(df.columns), axis=1)
    return df


def deterministic_parquet_bytes(df: pd.DataFrame) -> bytes:
    """Return deterministic parquet bytes for a DataFrame.

    Columns are sorted by name and a fixed compression/codec is used so the
    same logical frame always yields byte-identical output.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    df = _normalize_columns(df).reset_index(drop=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    buf = pa.BufferOutputStream()
    pq.write_table(
        table,
        buf,
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
        version="2.6",
    )
    return buf.getvalue().to_pybytes()


def deterministic_parquet(df: pd.DataFrame, path) -> None:
    """Atomically write a DataFrame to parquet with deterministic bytes."""
    path = Path(path)
    ensure_dir(str(path))
    data = deterministic_parquet_bytes(df)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)

