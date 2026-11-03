# spam_fraud_detector/core/__init__.py
#
from .anomaly import AnomalyComparer
from .comparer import ClassifierGroupComparer
from .data_loader import KaggleDataLoader
from .data_pipeline import DataPipeline
from .exceptions import (
    ArtifactError,
    ContractViolation,
    DataQualityError,
    DataSchemaError,
    DatasetNotFoundError,
    ExplainerError,
    SplitContractError,
    StatsError,
    TestLeakError,
)
from .explainer import ModelExplainer
from .result_store import ResultStore, RunContext
from .selector import TopKSelector
from .split import SplitManager, authorize_test_access
from .stats import StatsEngine, cohen_h, risk_difference
from .utils import (
    DATASETS,
    atomic_write_json,
    atomic_write_text,
    deterministic_parquet,
    deterministic_parquet_bytes,
    ensure_dir,
    format_metrics,
    get_dataset_config,
    get_git_commit,
    get_package_versions,
    hash_file,
    load_dataset,
    safe_model_name,
    save_dataframe,
    set_seed,
    summarize_class_distribution,
)
from .visualizer import ModelVisualizer

__all__ = [
    "DATASETS",
    "AnomalyComparer",
    "ArtifactError",
    "ClassifierGroupComparer",
    "ContractViolation",
    "DataPipeline",
    "DataQualityError",
    "DataSchemaError",
    "DatasetNotFoundError",
    "ExplainerError",
    "KaggleDataLoader",
    "ModelExplainer",
    "ModelVisualizer",
    "ResultStore",
    "RunContext",
    "SplitContractError",
    "SplitManager",
    "StatsEngine",
    "StatsError",
    "TestLeakError",
    "TopKSelector",
    "atomic_write_json",
    "atomic_write_text",
    "authorize_test_access",
    "cohen_h",
    "deterministic_parquet",
    "deterministic_parquet_bytes",
    "ensure_dir",
    "format_metrics",
    "get_dataset_config",
    "get_git_commit",
    "get_package_versions",
    "hash_file",
    "load_dataset",
    "risk_difference",
    "safe_model_name",
    "save_dataframe",
    "set_seed",
    "summarize_class_distribution",
]
