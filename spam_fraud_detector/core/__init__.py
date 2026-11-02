# spam_fraud_detector/core/__init__.py
#
from .comparer import ClassifierGroupComparer
from .visualizer import ModelVisualizer
from .data_loader import KaggleDataLoader
from .exceptions import (
    DataSchemaError, ArtifactError, StatsError, ContractViolation, DatasetNotFoundError,
    TestLeakError, DataQualityError, SplitContractError, ExplainerError,
)
from .result_store import ResultStore, RunContext
from .selector import TopKSelector
from .stats import StatsEngine, cohen_h, risk_difference
from .split import SplitManager, authorize_test_access
from .anomaly import AnomalyComparer
from .explainer import ModelExplainer
from .data_pipeline import DataPipeline
from .utils import (
    ensure_dir,
    format_metrics,
    summarize_class_distribution,
    safe_model_name,
    save_dataframe,
    get_dataset_config,
    load_dataset,
    DATASETS,
    set_seed,
    get_git_commit,
    get_package_versions,
    hash_file,
    atomic_write_json,
    atomic_write_text,
    deterministic_parquet,
    deterministic_parquet_bytes,
)

__all__ = [
    "ClassifierGroupComparer",
    "ModelVisualizer",
    "KaggleDataLoader",
    "ResultStore",
    "RunContext",
    "DataSchemaError", "ArtifactError", "StatsError", "ContractViolation",
    "ensure_dir",
    "format_metrics",
    "summarize_class_distribution",
    "safe_model_name",
    "save_dataframe",
    "get_dataset_config",
    "load_dataset",
    "DATASETS",
    "set_seed",
    "get_git_commit",
    "get_package_versions",
    "hash_file",
    "atomic_write_json",
    "atomic_write_text",
    "deterministic_parquet",
    "deterministic_parquet_bytes",
    "TopKSelector",
    "StatsEngine",
    "cohen_h",
    "risk_difference",
    "DatasetNotFoundError",
    "TestLeakError", "DataQualityError", "SplitContractError", "ExplainerError",
    "AnomalyComparer",
    "ModelExplainer",
    "DataPipeline",
    "SplitManager", "authorize_test_access",
]
