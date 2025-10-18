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
