"""Typed exceptions for the benchmarking framework."""


class DataSchemaError(Exception):
    """Raised when input data does not match the expected schema."""


class ArtifactError(Exception):
    """Raised when an artifact cannot be written or read."""


class StatsError(Exception):
    """Raised when statistical computation fails."""


class ContractViolation(Exception):
    """Raised when an outputs/ invariant is violated."""


class DatasetNotFoundError(Exception):
    """Raised when a required dataset, bundle, or split artifact is missing."""


class TestLeakError(Exception):
    """Raised when the test split is accessed without authorization."""


class DataQualityError(Exception):
    """Raised when the data quality gate fails."""


class SplitContractError(Exception):
    """Raised when a split contract is violated (missing manifest, unknown split, ...)."""


class ExplainerError(Exception):
    """Raised when an explainer cannot be constructed for a persona/model."""


__all__ = [
    "DataSchemaError",
    "ArtifactError",
    "StatsError",
    "ContractViolation",
    "DatasetNotFoundError",
    "TestLeakError",
    "DataQualityError",
    "SplitContractError",
    "ExplainerError",
]