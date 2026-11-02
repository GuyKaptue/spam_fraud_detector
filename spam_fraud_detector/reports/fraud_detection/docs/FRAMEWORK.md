# Framework & Methodology
### Binary Classification Benchmarking — Design, Metrics, and Validity

## 1. Purpose
Provide a **reusable, dataset-agnostic** harness that (a) benchmarks many classifiers under
identical splits, (b) contrasts **scaled vs. unscaled** pipelines, (c) generates reproducible
figures/tables, and (d) records every run in MLflow.

## 2. Architecture

| Layer | Component | Contract |
|---|---|---|
| Data | `KaggleDataLoader` | `load()` → DataFrame; idempotent download/cache; column schema validation |
| Experiment | `ClassifierGroupComparer` | `load_data()`, `prepare_data()`, `compare_groups()` → results dict |
| Reporting | `ModelVisualizer` | `save_*()` methods; one figure/table per call; writes to `reports/<name>/` |
| Tracking | MLflow | params (`scaled`), metrics (acc/f1/prec/recall), tags (runName, source, git commit) |

## 3. Experimental Protocol
1. **Split:** stratified train/test (default 80/20), fixed `random_state`.
2. **Groups:** *scaled* (StandardScaler → estimator) and *unscaled* (estimator only).
3. **Estimators:** LogisticRegression, KNN, SVC, DecisionTree, RandomForest, GradientBoosting,
   MLP, MultinomialNB (where valid), XGBoost, LightGBM.
4. **Skip policy:** estimators that raise on input (e.g., MultinomialNB with negatives) are
   recorded in the skipped list and excluded from aggregate figures.
5. **Logging:** every estimator = one MLflow run; confusion matrix saved as artifact JSON.

## 4. Metric Definitions

| Metric | Formula / Meaning | Use when |
|---|---|---|
| Accuracy | (TP+TN)/N | balanced classes only |
| Precision | TP/(TP+FP) | cost of false alarms |
| Recall (TPR) | TP/(TP+FN) | cost of misses |
| F1 | 2PR/(P+R) | single balanced score |
| ROC-AUC | area under TPR–FPR | ranking, less imbalance-sensitive |
| **PR-AUC / AUPRC** | area under precision–recall | **primary for rare positives** |

**Why AUPRC for fraud:** with 0.172% positives, ROC-AUC is inflated by the huge TN mass; AUPRC
focuses on the positive class and penalizes missed fraud.

## 5. Reproducibility
- Fixed seeds for split and stochastic estimators.
- MLflow captures params, metrics, tags, and `confusion_matrix.json` per run.
- Git commit hash recorded in run tags when available.

## 6. Threats to Validity
- **Single split** (no CV) → metric variance unquantified. *Mitigation:* add k-fold + CI (roadmap).
- **No threshold tuning** → default 0.5 threshold; PR curves expose the full trade-off.
- **Preprocessing leakage risk** → scaling is fit within the pipeline (verify in `comparer.py`).
- **Dataset drift** → Kaggle snapshots; record versions in `data/`.

## 7. Anomaly-Detection Connection
Rare-positive classification = **supervised anomaly detection**. This framework is the supervised
baseline against which unsupervised detectors (IsolationForest, LOF, One-Class SVM) should be
compared. Both sub-project reports include this analysis.

## 8. Figures & Provenance
All report figures are produced by `ModelVisualizer` and stored under
`spam_fraud_detector/reports/<dataset>/figures/`. The two animated GIFs in the root README
(`spam_fraud_detector/reports/<dataset>/figures/*_demo.gif`) are generated from those PNGs by `scripts/make_gifs.py`.
Do not edit figures by hand — regenerate them.
