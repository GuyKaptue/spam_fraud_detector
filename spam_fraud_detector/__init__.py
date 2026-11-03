__version__ = "0.1.0"

# spam_fraud_detector/spam_fraud_detector/__init__.py

from spam_fraud_detector.core import (
 DATASETS,
 ClassifierGroupComparer,
 KaggleDataLoader,
 ModelVisualizer,
 ensure_dir,
 format_metrics,
 get_dataset_config,
 load_dataset,
 safe_model_name,
 save_dataframe,
 summarize_class_distribution,
)

from .main import SpamFraudDetector

__all__ = [
 "DATASETS",
 "ClassifierGroupComparer",
 "KaggleDataLoader",
 "ModelVisualizer",
 "SpamFraudDetector",
 "ensure_dir",
 "format_metrics",
 "get_dataset_config",
 "load_dataset",
 "safe_model_name",
 "save_dataframe",
 "summarize_class_distribution"
]
