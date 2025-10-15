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
    """Create directory if it doesn't exist."""
    dir_path = os.path.dirname(path)
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

