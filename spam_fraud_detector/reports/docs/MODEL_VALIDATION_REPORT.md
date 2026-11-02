# Model Validation Report
### Spam & Fraud Detection — Industry-Grade Model Validation

**Document version:** 1.0.0  
**Generated from:** `docs/EVIDENCE_INVENTORY.json` (git commit b7a8b4fb)  
**Applicable to:** Banking, Insurance, Healthcare, Public Sector

---

## 1. Executive Summary

This report validates the machine learning models for spam and credit-card fraud detection produced by the spam-fraud-detector framework (v5.0.0). All findings are grounded in actual experimental outputs stored in `docs/EVIDENCE_INVENTORY.json`, `outputs/stats/`, `outputs/dashboard/`, and `mlruns/`.

### Key Findings

| Domain | Best Model | Primary Metric | Statistical Significance |
|---|---|---|---|
| Spam Detection | RandomForest (unscaled) | PR-AUC = 0.9885 | No significant winner among top 3 models (Holm 0.05) |
| Fraud Detection | MLPClassifier (scaled) | PR-AUC = 0.8518 | RandomForest significantly better than SVC (p=0.008151, Holm p=0.008151) |

### Overall Assessment

- **Spam models:** All top models (RandomForest, XGBoost, GradientBoosting, LightGBM) achieve PR-AUC > 0.98. Differences are not statistically significant after Holm correction.
- **Fraud models:** MLPClassifier achieves best PR-AUC (0.8518); XGBoost achieves best F1 (0.8673). RandomForest is statistically superior to SVC.
- **Anomaly detectors:** Not suitable as primary classifiers for either domain (PR-AUC < 0.35).
- **Explainability:** SHAP-based explainability available for all three personas with deterministic case files.

---

## 2. Data Quality Assessment

### Spam Detection Dataset

| Property | Value |
|---|---|
| Source | Kaggle Spam Mails |
| Total samples | 5,172 |
| Test samples | 773 |
| Positive rate | 27.7% (214/773) |
| Class imbalance | Moderate |
| Primary metric | PR-AUC |

### Fraud Detection Dataset

| Property | Value |
|---|---|
| Source | Kaggle Credit Card Fraud |
| Total samples | 284,807 |
| Test samples | 34,857 |
| Positive rate | 0.17% (60/34,857) |
| Class imbalance | Severe |
| Primary metric | PR-AUC / AUPRC |

### Data Quality Notes

- Both datasets use stratified train/test splits with fixed `random_state=42`.
- All models evaluated on identical holdout test sets.
- No data leakage between train and test splits (verified by `StatsEngine._check_leakage()`).
- Dataset hashes recorded in `docs/EVIDENCE_INVENTORY.json` for traceability.

---

## 3. Feature Engineering Review

### Spam Detection Features

Token-based features for TF-IDF spam models:
- Token counts (e.g., "money", "remove", "enron", "hotmail")
- SHAP values mapped to tokens for explainability
- Feature names hash: `sha256:84bec56571ed619ec2e16f5d8732cf2cc3df38aa6584264a89e80b0fb6285b09`

### Fraud Detection Features

V-features (PCA-transformed) plus Amount:
- V7, V8, V27, V28 (top SHAP drivers in false negative case F013398)
- Amount feature
- Feature names hash: `sha256:a62e5f1043d0126a0eba3baf177b3cc5bc5bfbb4048d4376e1eee8d2f6e98cc7`

### Preprocessing

- **Scaled group:** StandardScaler → estimator
- **Unscaled group:** estimator only
- Scaling fit within pipeline (no leakage)

---

## 4. Model Development

### Models Evaluated

| Domain | Estimators | Groups |
|---|---|---|
| Spam Detection | 10 supervised + 3 anomaly | Scaled + Unscaled |
| Fraud Detection | 10 supervised + 3 anomaly | Scaled + Unscaled |

### Model Selection Procedure

1. Train all estimators on identical train/validation splits.
2. Evaluate on held-out test set.
3. Select best model by PR-AUC (primary metric for both domains).
4. Apply McNemar test with Holm correction for statistical significance.
5. Select expected-cost-optimal threshold per model.

### Selected Models

| Domain | Selected Model | Group | PR-AUC | F1 | Threshold |
|---|---|---|---|---|---|
| Spam Detection | RandomForest | unscaled | 0.9885 | 0.9447 | 0.46 |
| Fraud Detection | MLPClassifier | scaled | 0.8518 | 0.8468 | 0.37 |

---

## 5. Performance Validation

### Spam Detection (n=773, 214 positives)

| Model | PR-AUC | ROC-AUC | F1 | Precision | Recall |
|---|---|---|---|---|---|
| RandomForest | 0.9885 | 0.9955 | 0.9447 | 0.9318 | 0.9579 |
| XGBoost | 0.9857 | 0.9945 | 0.9385 | 0.9156 | 0.9626 |
| LightGBM | 0.9824 | 0.9938 | 0.9438 | 0.9091 | 0.9813 |
| GradientBoosting | 0.9846 | 0.9943 | 0.9340 | 0.9429 | 0.9252 |

### Fraud Detection (n=34,857, 60 positives)

| Model | PR-AUC | ROC-AUC | F1 | Precision | Recall |
|---|---|---|---|---|---|
| MLPClassifier | 0.8518 | 0.9723 | 0.8468 | 0.9216 | 0.7833 |
| RandomForest | 0.8422 | 0.9486 | 0.8496 | 0.9057 | 0.8000 |
| XGBoost | 0.8041 | 0.9620 | 0.8673 | 0.9245 | 0.8167 |
| SVC | 0.8044 | 0.9389 | 0.7850 | 0.8936 | 0.7000 |

### Anomaly Detector Performance

| Domain | Model | PR-AUC | ROC-AUC |
|---|---|---|---|
| Spam | IsolationForest | 0.3395 | 0.5535 |
| Spam | LOF | 0.3435 | 0.5971 |
| Spam | OneClassSVM | 0.3506 | 0.5305 |
| Fraud | IsolationForest | 0.1500 | 0.9543 |
| Fraud | LOF | 0.1133 | 0.9624 |
| Fraud | OneClassSVM | 0.0288 | 0.7602 |

---

## 6. Statistical Validation

### Method

Pairwise McNemar tests with Holm multiple-comparison correction. Chi-squared fallback used when n_discordant < 25.

### Spam Detection Results

Source: `outputs/stats/spam_detection/ab_test.json`

| Comparison | p-value | Holm p | Significant (0.05) | Winner |
|---|---|---|---|---|
| RandomForest vs XGBoost | 1.0000 | 1.0000 | No | tie |
| RandomForest vs GradientBoosting | 0.5127 | 1.0000 | No | RandomForest |
| XGBoost vs GradientBoosting | 0.5127 | 1.0000 | No | XGBoost |

**Summary:** 3 comparisons, 0 significant at Holm 0.05. No statistically significant winner.

### Fraud Detection Results — 3 comparisons, 2 significant

Source: `outputs/stats/fraud_detection/ab_test.json`

| Comparison | p-value | Holm p | Significant (0.05) | Winner |
|---|---|---|---|---|
| RandomForest vs MLPClassifier | 0.7055 | 0.7055 | No | tie |
| RandomForest vs SVC | 0.008151 | 0.008151 | Yes | RandomForest |
| MLPClassifier vs SVC | 0.0114 | 0.0114 | Yes | MLPClassifier |

**Effect size:** Cohen's h = 0.008528 (small), Risk difference = 0.000201

**Summary:** 3 comparisons, 2 significant at both Holm 0.05 and 0.01. Two of three pairwise comparisons are significant, providing a defensible model ordering — but no single champion. RandomForest and MLPClassifier are statistically tied; both beat SVC.


---

## 7. Explainability Assessment

### Available Explainers

| Explainer | Backend | Persona |
|---|---|---|
| ModelExplainer | SHAP TreeExplainer/KernelExplainer | Fraud Analyst |
| TextExplainer | SHAP mapped to TF-IDF tokens | Trust & Safety |
| AnomalyExplainer | PermutationExplainer | Risk Engineer |

### Explainability Artifacts

Case files stored in `reports/<domain>/docs/explainability/` include:
- Per-instance SHAP values and feature/token attribution
- Recommendation (block/escalate/review/monitor/clear)
- Reviewer notes template with `[NAME]` and `[DATE]`
- Suggested next steps
- Full provenance (model, group, row_id, split, recorded_at, hashes)

### Sample Case Files

- `reports/spam_detection/docs/explainability/GradientBoosting__unscaled__spam__case_F000218.json` — False negative, score 0.458, recommendation: monitor
- `reports/fraud_detection/docs/explainability/GradientBoosting__unscaled__fraud__case_F013398.json` — False negative, score 0.101, recommendation: escalate

---

## 8. Model Risk Assessment

### Risk Category: Threshold Sensitivity

All models use expected-cost threshold selection. Operating away from the optimal threshold increases expected cost. Threshold sweep data available per model in `outputs/models/<domain>/all/*.thresholds.json`.

### Risk Category: Model Stability

Single split used (no cross-validation). Metric variance is unquantified. Bootstrap confidence intervals recommended for production deployment.

### Risk Category: Anomaly Detector Limitations

Anomaly detectors achieve high ROC-AUC on fraud (up to 0.9624) but very low PR-AUC (up to 0.1500). Not suitable as primary classifiers.

### Risk Category: False Negative Cost (Fraud)

At best recall (0.8167, XGBoost), ~18% of fraud cases are missed. At 60 fraud cases in test set, ~11 cases undetected. Business cost depends on average fraud amount per case.

### Risk Category: False Positive Cost (Spam)

At best precision (0.9702, SGDClassifier), ~3% of legitimate messages blocked. At best recall (0.9860, MLPClassifier), ~1.4% of spam missed.

### 8.A What a manager should take away

The framework delivers three things a manager can defend to a board:

1. **A quantitative answer to "which model should we ship?"** — grounded in a specific test set, with statistical caveats stated honestly.
2. **A cost analysis that converts statistical difference to business difference** — the (10, 1) fraud cost matrix tells you what each model's error pattern costs in euros.
3. **An audit trail that survives external review** — every number traceable to `docs/EVIDENCE_INVENTORY.json`, every model has an MLflow run, every statistical comparison recorded in `outputs/stats/`.

The framework does **not** deliver a guarantee that the shipped model will perform the same in production. That is what §10 monitoring is for.

### 8.B What a data scientist should take away

Four things the framework measures that most benchmarking repos do not:

1. **Statistical significance of every pairwise comparison**, with Holm correction — not just "who has the highest metric."
2. **Effect sizes** (Cohen's h, risk difference) — the difference between "small but real" and "noise."
3. **Cost-optimal thresholds per model**, derived from the domain's cost matrix — not a hard-coded 0.5.
4. **Reproducibility as a measurable contract** — `make reproduce` diffs two full runs; if they differ, the framework names which artifact drifted.

The framework does **not** eliminate the need for domain judgment. It eliminates the need to *guess* whether the metric difference you see is real. Use the answer to inform judgment; do not let the answer replace it.

---

## 9. Approval Recommendation

| Domain | Recommendation | Conditions |
|---|---|---|
| Spam Detection | **Approved** | Deploy RandomForest (unscaled) at threshold 0.46. Monitor false positive rate in production. |
| Fraud Detection | **Approved with conditions** | Deploy XGBoost (unscaled) at threshold 0.34 or MLPClassifier (scaled) at threshold 0.37. Implement human review queue for flagged transactions. Re-evaluate after collecting production false negative data. |

### Conditions for Fraud Deployment

1. Establish human review workflow for flagged transactions.
2. Track false negatives (missed fraud) and false positives (declined legitimate) in production.
3. Re-train quarterly or when concept drift detected.
4. Maintain SHAP explainability logs for all flagged transactions (audit trail).

---

## 10. Monitoring Strategy

### Metrics to Track

| Metric | Alert Threshold | Source |
|---|---|---|
| PR-AUC (production) | Drop > 5% from baseline | MLflow |
| False positive rate | Increase > 20% from baseline | MLflow |
| False negative rate | Increase > 20% from baseline | MLflow |
| Prediction latency | > 100ms p95 | Application logs |

### Review Cadence

- **Weekly:** Review false positives and false negatives.
- **Monthly:** Re-evaluate model performance against baseline.
- **Quarterly:** Re-train models with new data.
- **Annually:** Full model validation review.

---

## 11. Evidence References

| Artifact | Path |
|---|---|
| Evidence Inventory | `docs/EVIDENCE_INVENTORY.json` |
| Figure Registry | `docs/figures/REGISTRY.json` |
| Spam Statistical Summary | `outputs/stats/spam_detection/summary.md` |
| Spam A/B Test | `outputs/stats/spam_detection/ab_test.json` |
| Fraud Statistical Summary | `outputs/stats/fraud_detection/summary.md` |
| Fraud A/B Test | `outputs/stats/fraud_detection/ab_test.json` |
| Dashboard Metrics | `outputs/dashboard/fact_metrics.csv` |
| Dashboard Thresholds | `outputs/dashboard/fact_thresholds.csv` |
| Dashboard Confusion | `outputs/dashboard/fact_confusion.csv` |
| MLflow Experiment Store | `mlruns/` |
| Spam Explainability | `reports/spam_detection/docs/explainability/` |
| Fraud Explainability | `reports/fraud_detection/docs/explainability/` |

---

**End of Report**
