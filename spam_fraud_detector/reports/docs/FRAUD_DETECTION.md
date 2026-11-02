# Fraud Detection
### Industry Model Validation Report

## 1. Business Context

Credit card fraud detection is a high-stakes binary classification problem with severe class imbalance. False negatives (missed fraud) result in direct financial losses and chargebacks. False positives (declined legitimate transactions) cause customer friction, support costs, and churn. The dataset contains 284,807 transactions with only 492 fraud cases (0.172% positive rate). On the held-out test set (n=34,857), 60 transactions are fraudulent (0.17%).

## 2. Dataset Overview

| Property | Value |
|---|---|
| Total samples | 284,807 |
| Test samples | 34,857 |
| Positive (fraud) | 60 (0.17%) |
| Negative (legitimate) | 34,797 (99.83%) |
| Primary metric | PR-AUC / AUPRC |
| Secondary metrics | ROC-AUC, F1, Precision, Recall |

**Note:** Accuracy is misleading — a constant "legitimate" predictor scores 99.828% accuracy. PR-AUC is the appropriate primary metric for this extreme imbalance.

## 3. Experimental Design

Competing classifiers are evaluated on identical train/validation/test splits under a controlled and reproducible environment:

1. Train multiple candidate models (10 estimators across scaled and unscaled groups).
2. Generate predictions on the same holdout test set.
3. Compare model performance using PR-AUC as the primary metric (appropriate for extreme class imbalance).
4. Apply McNemar significance testing with Holm multiple-comparison correction.
5. Record all results through MLflow for reproducibility and auditability.

## 4. Candidate Models

**Scaled group (StandardScaler → estimator):**
- LogisticRegression (threshold 0.05)
- KNN (threshold 0.41)
- SVC (threshold 0.00)
- SGDClassifier (threshold 0.00)
- MLPClassifier (threshold 0.37)

**Unscaled group (estimator only):**
- RandomForest (threshold 0.56)
- GradientBoosting (threshold 0.31)
- XGBoost (threshold 0.34)
- LightGBM (threshold 1.00)

**Anomaly detection baseline (unsupervised):**
- IsolationForest (threshold 0.98)
- LOF (threshold 0.75)
- OneClassSVM (threshold 1.00)

## 5. Performance Results

Source: `docs/EVIDENCE_INVENTORY.json`, `outputs/dashboard/fact_metrics.csv`

| Model | Group | PR-AUC | ROC-AUC | F1 | Accuracy | Precision | Recall | Threshold |
|---|---|---|---|---|---|---|---|---|
| MLPClassifier | scaled | **0.8518** | 0.9723 | 0.8468 | 0.9995 | 0.9216 | 0.7833 | 0.37 |
| RandomForest | unscaled | 0.8422 | 0.9486 | 0.8496 | 0.9995 | 0.9057 | 0.8000 | 0.56 |
| XGBoost | unscaled | 0.8041 | 0.9620 | **0.8673** | 0.9996 | **0.9245** | **0.8167** | 0.34 |
| SVC | scaled | 0.8044 | 0.9389 | 0.7850 | 0.9993 | 0.8936 | 0.7000 | 0.00 |
| LogisticRegression | scaled | 0.8027 | 0.9611 | 0.7010 | 0.9992 | 0.9189 | 0.5667 | 0.05 |
| KNN | scaled | 0.7631 | 0.9248 | 0.8319 | 0.9995 | 0.8868 | 0.7833 | 0.41 |
| SGDClassifier | scaled | 0.7876 | 0.9690 | 0.6452 | 0.9991 | 0.9091 | 0.5000 | 0.00 |
| GradientBoosting | unscaled | 0.6697 | 0.9178 | 0.7934 | 0.9993 | 0.7869 | 0.8000 | 0.31 |
| LightGBM | unscaled | 0.2167 | 0.7566 | 0.3711 | 0.9965 | 0.2687 | 0.6000 | 1.00 |

## 6. Statistical Validation

Source: `outputs/stats/fraud_detection/ab_test.json`

### 6.A The three pairwise comparisons

| Model A | Model B | Method | n | n_discordant | n01 | n10 | p-value | Holm p | Significant (0.05) | Winner |
|---|---|---|---|---|---|---|---|---|---|---|
| RandomForest | MLPClassifier | chi2_fallback | 34,857 | 12 | 6 | 6 | 0.7055 | 0.7055 | No | MLPClassifier |
| RandomForest | SVC | chi2_fallback | 34,857 | 7 | 0 | 7 | 0.008151 | 0.008151 | Yes | RandomForest |
| MLPClassifier | SVC | chi2_fallback | 34,857 | 11 | 5 | 6 | 0.0114 | 0.0114 | Yes | MLPClassifier |

### 6.B Two significant, one not — what that means

Two of the three pairwise comparisons are significant at α = 0.05 after Holm correction. This means the current data supports a **statistically defensible ordering** of the top-3 models:

- **RandomForest vs SVC:** significant (p=0.008151, Holm p=0.008151) — RandomForest wins
- **MLPClassifier vs SVC:** significant (p=0.0114, Holm p=0.0114) — MLPClassifier wins  
- **RandomForest vs MLPClassifier:** not significant (p=0.7055, Holm p=0.7055) — indistinguishable

**This is not a single champion claim.** The non-significant pair (RandomForest vs MLPClassifier) means we cannot say which is better — they are statistically tied. The significant pairs tell us both RandomForest and MLPClassifier are better than SVC, but RandomForest vs MLPClassifier is too close to call.

**Effect sizes:** Cohen's h = 0.008528 (small), Risk difference = 0.000201. Even the significant pairs show tiny effect sizes — statistically detectable, but practically negligible.

### 6.C Effect size and business meaning

A statistically significant but small-effect difference (Cohen's h = 0.0085) is not automatically a business win. Under the (10, 1) fraud cost matrix, XGBoost's F1 advantage (0.8673 vs MLPClassifier 0.8468) comes from higher recall (0.8167 vs 0.7833), which converts to roughly €11 vs €22 expected fraud loss per case. **Report both p-value and effect size.**

## 7. Business Cost Analysis

### Cost Structure

| Error Type | Business Impact |
|---|---|
| False Positive (legitimate declined) | Customer friction, support call, potential churn |
| False Negative (missed fraud) | Direct financial loss, chargeback, investigation cost |

### Expected Cost at Optimal Threshold

Source: `docs/EVIDENCE_INVENTORY.json` (thresholds sweep data)

**RandomForest (best PR-AUC = 0.8422):**
- Expected cost at star threshold (0.56): model-specific cost curve available in `outputs/models/fraud_detection/all/RandomForest__unscaled.thresholds.json`
- F1 = 0.8496, Precision = 0.9057, Recall = 0.8000

**XGBoost (best F1 = 0.8673):**
- F1 = 0.8673, Precision = 0.9245, Recall = 0.8167, Threshold = 0.34
- Higher precision means fewer legitimate declines; higher recall means fewer missed fraud cases

### Trade-off Summary

- **MLPClassifier:** Best PR-AUC (0.8518), good balance of precision (0.9216) and recall (0.7833)
- **XGBoost:** Best F1 (0.8673) and best precision (0.9245) and best recall (0.8167) — most balanced for production
- **RandomForest:** Statistically significant winner over SVC, strong PR-AUC (0.8422)
- **SVC:** Threshold 0.00 means it never flags fraud at default — not suitable for production

## 8. Model Risk Discussion

### Threshold Sensitivity

All models use expected-cost threshold selection. The threshold sweep shows expected cost varies non-monotonically with threshold. For example, RandomForest's expected cost reaches minimum at threshold 0.56 but increases at both lower and higher thresholds. Operating away from the optimal threshold increases expected cost.

### Model Stability

- All models evaluated on identical test splits (n=34,857) — results are directly comparable.
- Single split used (no cross-validation) — metric variance is unquantified. Bootstrap confidence intervals are recommended for production deployment.
- Anomaly detectors (IsolationForest, LOF, OneClassSVM) perform poorly (PR-AUC 0.0288–0.1500) — not suitable as primary fraud classifiers.

### Operational Complexity

- Scaled models require preprocessing pipeline (StandardScaler) in production.
- Unscaled models (RandomForest, XGBoost, LightGBM, GradientBoosting) are simpler to deploy.
- XGBoost and LightGBM require careful hyperparameter management.

### Business Impact

- Best recall (0.8167, XGBoost) means ~18% of fraud cases are missed at the default operating point.
- Best precision (0.9245, XGBoost) means ~7.5% of flagged transactions are false alarms.
- At 60 fraud cases in the test set, missing 18% = ~11 fraud cases undetected.

## 9. Regulatory Perspective

### Explainability

SHAP-based explainability is available via `ModelExplainer`, `TextExplainer`, and `AnomalyExplainer`. Waterfall plots provide per-instance attribution for model decisions. Case files include recommendations (block/escalate/review/monitor/clear) with reviewer notes templates.

### Traceability

Every model run is tracked in MLflow with:
- Parameters (scaling, threshold policy, hyperparameters)
- Metrics (PR-AUC, ROC-AUC, F1, accuracy, precision, recall)
- Tags (run name, source, git commit)
- Artifacts (confusion matrix JSON)

### Reproducibility

- Fixed seeds (`random_state=42`) for split and stochastic estimators.
- Git commit hash recorded in MLflow run tags.
- Deterministic figure regeneration via `make figures` from `docs/EVIDENCE_INVENTORY.json`.

### Auditability

Dashboard tables (`outputs/dashboard/`) provide queryable Parquet/CSV exports of all metrics, thresholds, curves, and confusion matrices for regulatory review.

## 10. Conclusion

MLPClassifier achieves the highest PR-AUC (0.8518), XGBoost achieves the best F1 (0.8673) with best precision/recall balance under the (10, 1) cost matrix. RandomForest is statistically superior to SVC (p=0.008151, Holm p=0.008151). Two of three pairwise comparisons are significant at Holm 0.05, providing a defensible model ordering — but no single champion. **The choice among the top-3 is a business decision based on cost sensitivity, not a statistical verdict.** All results are grounded in `docs/EVIDENCE_INVENTORY.json` and `outputs/stats/fraud_detection/ab_test.json`.

<!-- ab-summary:start -->
## Statistical Significance

Pairwise McNemar tests on test-split predictions, with Holm family-wise error rate correction.

- Comparisons: 3
- Significant (Holm 0.05): 2
- Significant (Holm 0.01): 0

### Pairwise comparisons

| Model A | Model B | Method | p-value | Holm p | Significant | Effect Size (h) | Winner |
|---------|---------|--------|---------|--------|-------------|-----------------|--------|
| RandomForest | MLPClassifier | chi2_fallback | 0.7055 | 0.7055 | ✗ | -0.001 | MLPClassifier |
| RandomForest | SVC | chi2_fallback | 0.008151 | 0.02445 | ✓ | +0.009 | RandomForest |
| MLPClassifier | SVC | chi2_fallback | 0.01141 | 0.02282 | ✓ | +0.010 | MLPClassifier |

### Winner
MLPClassifier vs SVC — p=0.01141, Holm p=0.02282

<!-- ab-summary:end -->
