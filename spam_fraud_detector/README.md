# spam-fraud-detector Package

[![CI](https://github.com/GuyKaptue/spam-fraud-detector/actions/workflows/ci.yml/badge.svg)](https://github.com/GuyKaptue/spam-fraud-detector/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/GuyKaptue/spam-fraud-detector/branch/main/graph/badge.svg)](https://codecov.io/gh/GuyKaptue/spam-fraud-detector)
[![PyPI](https://img.shields.io/pypi/v/spam-fraud-detector)](https://pypi.org/project/spam-fraud-detector/)
[![Python](https://img.shields.io/pypi/pyversions/spam-fraud-detector)](https://pypi.org/project/spam-fraud-detector/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../../LICENSE)

**Author:** Guy M. Kaptue T. — [GitHub](https://github.com/GuyKaptue) — guykaptue24@gmail.com  
**License:** MIT · **Version:** 5.0.0 · **Python:** ≥3.9  
**Repository:** https://github.com/GuyKaptue/spam-fraud-detector

---

## 1. Purpose

The `spam_fraud_detector` package is the core Python module of a production-grade machine learning experimentation, explainability, statistical validation, and model governance framework for binary classification under class imbalance — specifically spam detection and credit-card fraud detection.

This package implements the entire experimental pipeline in-process: data loading and cleaning, stratified splitting, feature engineering, model training and comparison, statistical validation (McNemar + Holm correction), SHAP-based explainability for three personas, artifact persistence via `ResultStore`, and dashboard-table generation.

It is designed for **reproducibility, traceability, and auditability**: every model run is logged to MLflow, every artifact is materialized once and stored under `outputs/`, every documentation figure is deterministic, and every documentation claim is grounded in `docs/EVIDENCE_INVENTORY.json`.

---

## 2. Repository Map

```
spam-fraud-detector/
├── README.md                    # Executive overview (root)
├── Makefile                     # Build, train, explain, verify, audit targets
├── pyproject.toml               # Package configuration
├── requirements.txt             # Runtime dependencies
├── CITATION.cff                 # Academic citation metadata
├── CHANGELOG.md                 # Release history
├── LICENSE                      # MIT license
├── .github/workflows/ci.yml     # GitHub Actions CI pipeline
├── scripts/                     # Helper scripts (evidence, figures, verification, audits)
├── spam_fraud_detector/         # Core Python package (this directory)
│   ├── README.md                # This file
│   ├── core/                    # Core framework modules
│   │   ├── README.md            # Core module documentation
│   │   ├── __init__.py
│   │   ├── comparer.py          # ClassifierGroupComparer
│   │   ├── data_loader.py       # KaggleDataLoader
│   │   ├── data_pipeline.py     # DataPipeline (L1–L6)
│   │   ├── exceptions.py        # Typed exceptions
│   │   ├── features.py          # FeatureEngineer
│   │   ├── fake_subjects.py     # Synthetic subject generation
│   │   ├── hypothetical_scenarios.json
│   │   ├── quality.py           # QualityGate
│   │   ├── recommendations.py   # Persona-aware recommendations
│   │   ├── result_store.py      # ResultStore (atomic artifact writer)
│   │   ├── selector.py          # TopKSelector
│   │   ├── split.py             # SplitManager
│   │   ├── stats.py             # StatsEngine (McNemar + Holm)
│   │   ├── utils.py             # Shared utilities
│   │   ├── visualizer.py        # ModelVisualizer
│   │   ├── anomaly.py           # AnomalyComparer
│   │   ├── case_renderer.py     # Case file renderer
│   │   └── explainer.py         # ModelExplainer / TextExplainer / AnomalyExplainer
│   └── main.py                  # CLI entry point
├── tests/                       # Unit, integration, regression tests
│   ├── README.md                # Test suite documentation
│   ├── unit/
│   ├── integration/
│   ├── regression/
│   └── conftest.py
├── docs/                        # Generated documentation figures
│   └── figures/
├── reports/                     # Generated domain reports
│   ├── README.md
│   ├── docs/                    # Authored documentation (FRAMEWORK.md, PAPER.md, etc.)
│   ├── spam_detection/          # Spam domain reports
│   └── fraud_detection/         # Fraud domain reports
├── outputs/                     # Generated experimental artifacts (gitignored)
│   ├── models/                  # Trained model bundles
│   ├── predictions/             # Prediction parquet files
│   ├── stats/                   # A/B test results
│   ├── dashboard/               # Fact/dimension tables
│   └── logs/                    # Execution logs
├── mlruns/                      # MLflow experiment tracking (gitignored)
├── data/                        # Raw and processed datasets (gitignored)
└── .gitignore                   # Git ignore rules
```

---

## 3. Installation

```bash
git clone https://github.com/GuyKaptue/spam-fraud-detector.git
cd spam-fraud-detector

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or: .venv\Scripts\activate  # Windows

# Install package with dev dependencies
pip install -e ".[dev]"
```

---

## 4. Quickstart

```bash
# 1. Download and prepare data
make data

# 2. Train all models for both domains
make train

# 3. Regenerate documentation figures
make figures

# 4. Run explainability for all personas
make explain

# 5. Verify documentation against evidence inventory
make docs-verify

# 6. Run full test suite
make test
```

---

## 5. Package Modules

| Module | Purpose | Key Classes |
|---|---|---|
| `comparer` | Train and evaluate classifiers in scaled/unscaled regimes | `ClassifierGroupComparer` |
| `data_loader` | Download, cache, and prepare Kaggle datasets | `KaggleDataLoader` |
| `data_pipeline` | Orchestrate six-stage data preparation (L1–L6) | `DataPipeline` |
| `exceptions` | Typed exceptions for all failure classes | `ExplainerError`, `TestLeakError`, etc. |
| `features` | Fit scalers/encoders on train only, transform val/test | `FeatureEngineer` |
| `fake_subjects` | Generate deterministic synthetic subjects for hypothetical cases | `SyntheticSubject`, `load_scenarios` |
| `case_renderer` | Render persona case files to reviewer-facing text | `render_case_summary` |
| `quality` | Data quality gate evaluation | `QualityGate` |
| `recommendations` | Persona-aware recommendation blocks | `recommend()` |
| `result_store` | Atomic artifact persistence | `ResultStore`, `RunContext` |
| `selector` | Select top-K models from persisted outputs | `TopKSelector` |
| `split` | Stratified train/val/test split with test-access guard | `SplitManager` |
| `stats` | McNemar + Holm statistical validation | `StatsEngine`, `cohen_h`, `risk_difference` |
| `visualizer` | Publication-ready figures (ROC, PR, confusion, F1) | `ModelVisualizer` |
| `anomaly` | Unsupervised anomaly detection baseline | `AnomalyComparer` |
| `explainer` | SHAP-based explainability (3 personas) | `ModelExplainer`, `TextExplainer`, `AnomalyExplainer` |
| `utils` | Shared helpers (seeding, hashing, atomic writes) | `set_seed`, `hash_file`, `atomic_write_json` |

---

## 5. Experimental Protocol

All experiments follow a **split-once, write-once** protocol:

1. **Split:** The canonical dataset is stratified into `train / val / test` exactly once with `random_state=42`. The test split is never exposed outside the pipeline except through `ResultStore.write_model()`.
2. **Train:** Each estimator is trained on the train split (scaled or unscaled group).
3. **Evaluate:** Predictions are generated on the held-out test set; metrics (PR-AUC, ROC-AUC, F1, accuracy, precision, recall) and curves (ROC, PR) are computed.
4. **Persist:** Each model bundle (`.joblib`, `.meta.json`, `.eval.json`, `.curve.json`, `.confusion.json`, `.parquet` predictions) is written via `ResultStore.write_model()`.
5. **Validate:** `ResultStore.finalize()` atomically replaces `outputs/` with the staging directory after validating invariants.
6. **Explain:** SHAP explanations are generated on the validation split only — never on test.
7. **Validate statistically:** McNemar tests with Holm correction compare pairwise model performance on identical test samples.

---

## 6. Explainability Personas

| Persona | Explainer | Use Case |
|---|---|---|
| **Fraud Analyst** | `ModelExplainer` (SHAP TreeExplainer/KernelExplainer) | Tree/linear model explanations, per-instance waterfalls |
| **Trust & Safety** | `TextExplainer` (SHAP mapped to TF-IDF tokens) | Token-level spam explanations |
| **Risk Engineer** | `AnomalyExplainer` (PermutationExplainer) | Anomaly detector explanations |

Waterfalls are computed on the **validation split only** — never on test — to prevent leakage.

---

## 7. CLI Entry Point

The package exposes a CLI via `spam_fraud_detector.main`:

```bash
# Train all models
python -m spam_fraud_detector.main --dataset spam
python -m spam_fraud_detector.main --dataset fraud

# Statistics only
python -m spam_fraud_detector.main --dataset spam --stats-only
python -m spam_fraud_detector.main --dataset fraud --stats-only

# Explainability
python -m spam_fraud_detector.main --dataset spam --explainer shap --explain-only --persona spam
python -m spam_fraud_detector.main --dataset fraud --explainer shap --explain-only --persona fraud
python -m spam_fraud_detector.main --dataset fraud --explainer shap --explain-only --persona anomaly

# Stats tournament (sequential pairwise A/B)
python -m spam_fraud_detector.main --dataset spam --stats-tournament
python -m spam_fraud_detector.main --dataset fraud --stats-tournament
```

Run `python -m spam_fraud_detector.main --help` for all options.

---

## 8. Reproducibility

| Mechanism | Guarantee | Verification |
|---|---|---|
| Fixed seed (`SEED=42`) | All split and stochastic operations deterministic | `make reproduce` |
| Hashed row identifiers | Row identity is content-based, not positional | `tests/regression/test_split_determinism.py` |
| Deterministic parquet | Two writes of same DataFrame produce byte-identical files | `make deterministic-parquet-check` |
| MLflow tracking | Every run has params, metrics, tags, artifacts | `mlflow ui` |
| Evidence inventory | Every doc claim bound to artifact path + value | `make docs-verify` |

---

## 9. Related Documentation

| Document | Location | Contents |
|---|---|---|
| Executive Overview | `README.md` (root) | Problem, architecture, results, reproducibility |
| Core Module Docs | `spam_fraud_detector/core/README.md` | Per-module `What it does / How it does it / What it produces / How to use it` |
| Framework Whitepaper | `spam_fraud_detector/reports/docs/FRAMEWORK.md` | Architecture, methodology, statistical validation |
| Paper-Ready Results | `spam_fraud_detector/reports/docs/PAPER.md` | Publication-ready results, significance, threats to validity |
| Spam Case Study | `spam_fraud_detector/reports/docs/SPAM_DETECTION.md` | Spam domain metrics, statistical validation |
| Fraud Case Study | `spam_fraud_detector/reports/docs/FRAUD_DETECTION.md` | Fraud domain cost analysis, threshold sensitivity |
| Model Validation Report | `spam_fraud_detector/reports/docs/MODEL_VALIDATION_REPORT.md` | Industry-grade validation for banking/insurance/healthcare |
| Explainability Framework | `spam_fraud_detector/reports/docs/EXPLAINABILITY.md` | Three personas, SHAP analysis, case investigations |
| Evidence Inventory | `spam_fraud_detector/reports/docs/EVIDENCE_INVENTORY.json` | Machine-readable evidence (25 models, 2 domains) |
| Test Suite | `tests/README.md` | Unit, integration, regression test documentation |

---

## 10. License

MIT © Guy M. Kaptue T. See [../../LICENSE](../../LICENSE).
