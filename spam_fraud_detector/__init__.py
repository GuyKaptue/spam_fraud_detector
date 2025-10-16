# spam_fraud_detector/spam_fraud_detector/__init__.py

from spam_fraud_detector.core import (
 ClassifierGroupComparer,
 ModelVisualizer,
 KaggleDataLoader,
 ensure_dir,
 format_metrics,
 summarize_class_distribution,
 safe_model_name,
 save_dataframe,
 get_dataset_config,
 DATASETS
)

from .main import SpamFraudDetector
__all__ = [
    "SpamFraudDetector",
    "ClassifierGroupComparer",
    "ModelVisualizer",
    "KaggleDataLoader",
    "ensure_dir",
    "format_metrics",
    "summarize_class_distribution",
    "safe_model_name",
    "save_dataframe",
    "get_dataset_config",
    "DATASETS"
]
