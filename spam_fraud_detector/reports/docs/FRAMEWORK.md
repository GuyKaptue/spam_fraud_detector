# Framework & Methodology
### Binary Classification Benchmarking — Design, Metrics, and Validity (v5.0)

## 1. Purpose

A production-grade machine learning experimentation, explainability, statistical validation, and model governance framework for spam and credit-card fraud detection. The platform combines reproducible benchmarking, MLflow experiment tracking, statistical significance testing (McNemar test with Holm correction, effect-size estimation), SHAP-based explainability, anomaly detection, and automated evidence-based reporting. All documentation is generated from actual experimental outputs.

## 2. Architecture

```
Data Ingestion ──▶ Split Manager ──▶ Feature Engineering ──▶ Model Training
                                                      │
                                      ┌───────────────┴───────────────┐
                                      ▼                               ▼
                              Statistical Validation ─────────▶ Explainability
                                      │                               │
                                      ▼                               ▼
                              Model Selection ─────────────▶ Reporting
```

| Layer | Component | Contract |
|---|---|---|
| Data | `KaggleDataLoader` | Download, cache, prepare Kaggle datasets; idempotent; column schema validation |
| Experiment | `ClassifierGroupComparer` | Split, train scaled & unscaled groups, collect metrics |
| Reporting | `ModelVisualizer` | Confusion matrices, F1 bars, group comparison, ROC & PR curves, Excel/PDF |
| Tracking | MLflow | params, metrics, tags per run; experiment store in `mlruns/` |
| Evidence | `ResultStore` + `render_doc_figures.py` | Artifact persistence; deterministic figure rendering from `docs/EVIDENCE_INVENTORY.json` |
| Explainability | `ModelExplainer`, `TextExplainer`, `AnomalyExplainer` | SHAP explanations, waterfall plots, case files |

## 3. Experimental Protocol

### Data Splits

- **Split:** Stratified train/test split, fixed `random_state=42`.
- **Spam:** 5,172 total samples → 773 test samples (214 positives, 27.7%).
- **Fraud:** 284,807 total samples → 34,857 test samples (60 positives, 0.17%).
- All models evaluated on identical holdout test sets.

### Model Groups

| Group | Preprocessing | Estimators |
|---|---|---|
| Scaled | StandardScaler → estimator | LogisticRegression, KNN, SVC, SGDClassifier, MLPClassifier |
| Unscaled | Estimator only | RandomForest, GradientBoosting, XGBoost, LightGBM, MultinomialNB |
| Anomaly | Unsupervised | IsolationForest, LOF, OneClassSVM |

### Skip Policy

Estimators that raise on input (e.g., MultinomialNB on negative features) are recorded as skipped and excluded from aggregate figures.

### Logging

Every estimator = one MLflow run. Confusion matrix saved as artifact JSON. Git commit hash recorded in run tags when available.

## 4. Metric Definitions

| Metric | Formula / Meaning | Use when |
|---|---|---|
| Accuracy | (TP+TN)/N | Balanced classes only |
| Precision | TP/(TP+FP) | Cost of false alarms |
| Recall (TPR) | TP/(TP+FN) | Cost of misses |
| F1 | 2PR/(P+R) | Single balanced score |
| ROC-AUC | Area under TPR–FPR | Ranking, less imbalance-sensitive |
| **PR-AUC / AUPRC** | Area under precision–recall | **Primary for rare positives** |

**Why PR-AUC for fraud:** With 0.17% positives, ROC-AUC is inflated by the huge TN mass. PR-AUC focuses on the positive class and penalizes missed fraud.

## 5. Actual Results Summary

### Spam Detection (n=773, 214 positives)

| Model | Group | PR-AUC | ROC-AUC | F1 | Precision | Recall | Threshold |
|---|---|---|---|---|---|---|---|
| RandomForest | unscaled | **0.9885** | **0.9955** | **0.9447** | 0.9318 | 0.9579 | 0.46 |
| XGBoost | unscaled | 0.9857 | 0.9945 | 0.9385 | 0.9156 | 0.9626 | 0.35 |
| LightGBM | unscaled | 0.9824 | 0.9938 | 0.9438 | 0.9091 | **0.9813** | 0.51 |
| GradientBoosting | unscaled | 0.9846 | 0.9943 | 0.9340 | 0.9429 | 0.9252 | 0.46 |
| MLPClassifier | scaled | 0.9592 | 0.9910 | 0.9399 | 0.8979 | 0.9860 | 0.85 |

### Fraud Detection (n=34,857, 60 positives)

| Model | Group | PR-AUC | ROC-AUC | F1 | Precision | Recall | Threshold |
|---|---|---|---|---|---|---|---|
| MLPClassifier | scaled | **0.8518** | 0.9723 | 0.8468 | 0.9216 | 0.7833 | 0.37 |
| RandomForest | unscaled | 0.8422 | 0.9486 | 0.8496 | 0.9057 | 0.8000 | 0.56 |
| XGBoost | unscaled | 0.8041 | 0.9620 | **0.8673** | **0.9245** | **0.8167** | 0.34 |
| SVC | scaled | 0.8044 | 0.9389 | 0.7850 | 0.8936 | 0.7000 | 0.00 |

### Statistical Validation

**Spam:** 3 McNemar comparisons, 0 significant at Holm 0.05. No significant winner among RandomForest, XGBoost, GradientBoosting.

**Fraud:** 3 McNemar comparisons, 2 significant at Holm 0.05:
- RandomForest vs SVC: p=0.008151, Holm p=0.008151, significant (RandomForest wins)
- MLPClassifier vs SVC: p=0.0114, Holm p=0.0114, significant (MLPClassifier wins)  
- RandomForest vs MLPClassifier: p=0.7055, Holm p=0.7055, not significant (tie)

Cohen's h = 0.008528 (small effect), Risk difference = 0.000201.

## 6. Evidence Inventory

`docs/EVIDENCE_INVENTORY.json` is the single source of truth for documentation claims:

- **25 models** across 2 domains (spam_detection, fraud_detection)
- **15 dashboard tables** (fact_metrics, fact_stats, dim_method, etc.)
- **Per-model data**: PR-AUC, F1, threshold, confusion matrix, ROC/PR curves, Holm comparisons
- **Generated at:** 2026-11-01T09:33:12.797694+00:00
- **Git commit:** b7a8b4fb282489f8c2dadda5e3e5f5e046c41a3d

All figure render functions read exclusively from this inventory via `scripts/render_doc_figures.py`.

## 7. Reproducibility

- Fixed seeds for split and stochastic estimators (`random_state=42`).
- MLflow captures params, metrics, tags, and `confusion_matrix.json` per run.
- Git commit hash recorded in run tags when available.
- All documentation figures are deterministic: `make figures` regenerates byte-identical PNGs from `docs/EVIDENCE_INVENTORY.json`.

## 8. Threats to Validity

- **Single split** (no CV) → metric variance unquantified. Mitigation: add k-fold + CI (roadmap).
- **No threshold tuning** → expected-cost-optimal threshold per model; PR curves expose the full trade-off.
- **Preprocessing leakage risk** → scaling is fit within the pipeline (verified by `StatsEngine._check_leakage()`).
- **Dataset drift** → Kaggle snapshots; record versions in `data/`.
- **Anomaly detector limitations** → High ROC-AUC but low PR-AUC on fraud; suitable only as supplementary signals.

## 9. Reproducibility Contract

This project treats reproducibility as a **testable contract**, not a promise. Six independent mechanisms enforce it:

| Mechanism | What it guarantees | How to verify |
|---|---|---|
| `SEED=42` threaded through `SplitManager` | All split and stochastic operations deterministic | `make reproduce` |
| `SplitManager.row_id_hashes` in `split.json` | Row identity is content-based, not positional | `tests/regression/test_split_determinism.py` |
| `deterministic_parquet` | Two writes of same DataFrame produce byte-identical files | `make deterministic-parquet-check` |
| `reproduce_diff.py` | Two full pipeline runs produce byte-identical `outputs/` | `make reproduce` |
| `docs/EVIDENCE_INVENTORY.json` | Every documented number bound to artifact path + value | `make docs-verify` |
| MLflow runs in `mlruns/` | Every model run has params, metrics, tags, artifacts | `mlflow ui` |

**Not reproducible:** PNG bytes across matplotlib versions, timestamps in `.meta.json`, order of unordered `groupby` rows (always sorted before writing).

## 10. Evidence & Verification

`docs/EVIDENCE_INVENTORY.json` is the **single source of truth for every documented claim**. Regenerated by `make evidence`, it lists:

- **25 models** across 2 domains with `meta.json`, `eval.json`, `curve.json`, `confusion.json`, `thresholds.json` paths
- **6 statistical comparison files** (3 per domain)
- **15 dashboard tables** (fact_metrics, fact_stats, etc.)
- **Every figure** in `docs/figures/`

The verification chain:

```
make evidence        # regenerates EVIDENCE_INVENTORY.json from outputs/
make figures         # regenerates docs/figures/*.png from the inventory
make docs-verify     # verifies every claim in every doc against the inventory
make reproduce       # confirms two runs produce identical outputs
```

If any step fails, **the doc is wrong, not the artifact.** Update the doc to match the artifact.
