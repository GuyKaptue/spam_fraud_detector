# Data Pipeline

Six-stage data preparation ladder (L1–L6) that transforms raw Kaggle datasets into
production-ready feature matrices for model training. Each stage is idempotent,
cache-aware, and produces deterministic Parquet/CSV artifacts.

## Pipeline Stages

| Layer | Name | Component | Input | Output | Contract |
|---|---|---|---|---|---|
| L1 | Raw Data | `KaggleDataLoader` | Kaggle dataset | `data/raw/<domain>/` | Downloaded, cached, schema-validated |
| L2 | Cleaned | `DataPipeline.clean()` | Raw CSVs | `data/processed/<domain>/01_cleaned/` | Deduplicated, typed, missing-value handled |
| L3 | Modeled | `DataPipeline.model()` | Cleaned data | `data/processed/<domain>/02_model/` | Train/val/test split, stratified |
| L4 | Analyzed | `DataPipeline.analyze()` | Modeled data | `data/processed/<domain>/03_analyzed/` | Statistical summaries, class distributions |
| L5 | KPIs | `DataPipeline.kpis()` | Analyzed data | `data/processed/<domain>/04_kpis/` | Threshold costs, expected cost matrices |
| L6 | Features | `FeatureEngineer` | Analyzed data | `data/processed/<domain>/05_features/` | Scaled features, encoders, split manifests |

## Stage Details

### L1 — Raw Data (`KaggleDataLoader`)

Downloads Kaggle datasets via `kagglehub` or the `kaggle` CLI, caches them under
`data/raw/<domain>/`, and validates column schemas. Idempotent — re-running does
not re-download if files already exist and credentials are available.

### L2 — Cleaned (`DataPipeline.clean()`)

Reads raw CSVs, removes duplicates, handles missing values, infers column types,
and writes cleaned Parquet files under `data/processed/<domain>/01_cleaned/`.

### L3 — Modeled (`DataPipeline.model()`)

Performs stratified train/validation/test split (`random_state=42`), writes
stratified splits under `data/processed/<domain>/02_model/`, and records the split
manifest with row-id hashes at `data/processed/<domain>/splits/split.json`.

### L4 — Analyzed (`DataPipeline.analyze()`)

Computes class distributions, feature statistics, and data quality metrics.
Writes analysis artifacts under `data/processed/<domain>/03_analyzed/`.

### L5 — KPIs (`DataPipeline.kpis()`)

Computes expected cost matrices, threshold sweep data, and cost-model parameters.
Writes `kpis.json` and threshold sweep CSVs under `data/processed/<domain>/04_kpis/`.

### L6 — Features (`FeatureEngineer`)

Fits `StandardScaler` and token encoders on training data only (no leakage),
persists fitted encoders as `feature_encoder.joblib`, and writes scaled feature
matrices under `data/processed/<domain>/05_features/`.

## Pipeline Execution

```bash
# Run the complete pipeline (data + train + stats + explain + verify)
make pipeline

# Run only data preparation
make data

# Verify split determinism
make deterministic-parquet-check
```

## Split Manager

`SplitManager(domain, seed=42)` performs stratified splits and writes:

| Artifact | Path | Contents |
|---|---|---|
| Split manifest | `data/processed/<domain>/splits/split.json` | Seed, ratios, row counts, row-id hashes, label distributions |
| Train split | `data/processed/<domain>/06_features/train.parquet` | Training subset |
| Val split | `data/processed/<domain>/06_features/val.parquet` | Validation subset |
| Test split | `data/processed/<domain>/06_features/test.parquet` | Test subset |

Test-split access is guarded: `StatsEngine._check_leakage()` verifies no
preprocessing information from train/val leaks into test feature engineering.

## Feature Engineering

`FeatureEngineer` fits `StandardScaler` on training data only and transforms
all splits. Fitted encoders are persisted as
`data/processed/<domain>/06_features/feature_encoder.joblib` and referenced by
every model bundle's `meta.json` to prevent preprocessing leakage.

## Idempotency

All pipeline stages are idempotent: re-running produces byte-identical artifacts
given the same inputs. Use `make deterministic-parquet-check` to verify.

---

**Source:** `spam_fraud_detector/core/data_pipeline.py` (`DataPipeline`),
`spam_fraud_detector/core/split.py` (`SplitManager`),
`spam_fraud_detector/core/features.py` (`FeatureEngineer`),
`spam_fraud_detector/core/data_loader.py` (`KaggleDataLoader`)
