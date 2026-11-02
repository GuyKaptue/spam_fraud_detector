# Fraud Detection Reports

[![CI](https://github.com/GuyKaptue/spam-fraud-detector/actions/workflows/ci.yml/badge.svg)](https://github.com/GuyKaptue/spam-fraud-detector/actions/workflows/ci.yml)

**Author:** Guy M. Kaptue T. — [GitHub](https://github.com/GuyKaptue) — guykaptue24@gmail.com  
**License:** MIT · **Version:** 5.0.0  
**Domain:** Fraud Detection (binary classification, severe class imbalance)

---

## 1. Purpose

This directory contains all reports, figures, and explainability artifacts for the **fraud detection** domain of the spam-fraud-detector framework. It documents the end-to-end experimental protocol, model comparison results, statistical validation, cost analysis, and SHAP-based explainability for credit-card fraud detection.

All findings are grounded in actual experimental outputs stored in `spam_fraud_detector/reports/docs/EVIDENCE_INVENTORY.json` and `outputs/stats/fraud_detection/`.

---

## 2. Repository Map

```
reports/fraud_detection/
├── README.md                    # This file
├── docs/
│   ├── FRAMEWORK.md             # Fraud-specific framework documentation
│   └── FRAUD_DETECTION.md       # Fraud detection validation report
├── figures/                     # Fraud-specific figures (8 PNGs)
│   ├── fraud_class_distribution.png
│   ├── fraud_threshold_curve.png
│   ├── fraud_roc_curve.png
│   ├── fraud_pr_curve.png
│   ├── fraud_pr_auc_by_model.png
│   ├── fraud_threshold_sensitivity.png
│   ├── fraud_holm_significance.png
│   └── fraud_roc_vs_pr.png
└── explainability/              # SHAP waterfalls and case files
    ├── GradientBoosting__unscaled__fraud__case_*.json/png
    ├── IsolationForest__anomaly__case_*.json/png
    ├── KNN__scaled__fraud__case_*.json/png
    ├── LogisticRegression__scaled__fraud__case_*.json/png
    ├── MLPClassifier__scaled__fraud__case_*.json/png
    ├── RandomForest__unscaled__fraud__case_*.json/png
    ├── SGDClassifier__scaled__fraud__case_*.json/png
    ├── SVC__scaled__fraud__case_*.json/png
    └── XGBoost__unscaled__fraud__case_*.json/png
```

---

## 3. Dataset Overview

| Property | Value |
|---|---|
| Source | Kaggle Credit Card Fraud |
| Total samples | 284,807 |
| Test samples | 34,857 |
| Positive (fraud) | 60 (0.17%) |
| Negative (legitimate) | 34,797 (99.83%) |
| Primary metric | PR-AUC / AUPRC |
| Secondary metrics | ROC-AUC, F1, Precision, Recall |
| Class imbalance | Severe |

**Important:** Accuracy is misleading — a constant "legitimate" predictor scores **99.828%** accuracy. PR-AUC is the appropriate primary metric for this extreme imbalance.

---

## 4. Candidate Models

Ten supervised estimators were evaluated across two preprocessing regimes:

**Scaled group (StandardScaler → estimator):**

| Model | Threshold |
|---|---|
| LogisticRegression | 0.05 |
| KNN | 0.41 |
| SVC | 0.00 |
| SGDClassifier | 0.00 |
| MLPClassifier | 0.37 |

**Unscaled group (estimator only):**

| Model | Threshold |
|---|---|
| RandomForest | 0.56 |
| GradientBoosting | 0.31 |
| XGBoost | 0.34 |
| LightGBM | 1.00 |

**Anomaly detection baseline (unsupervised):**

| Model | Threshold |
|---|---|
| IsolationForest | 0.98 |
| LOF | 0.75 |
| OneClassSVM | 1.00 |

---

## 5. Performance Results

| Model | Group | PR-AUC | ROC-AUC | F1 | Precision | Recall | Threshold |
|---|---|---|---|---|---|---|---|
| MLPClassifier | scaled | **0.8518** | 0.9723 | 0.8468 | 0.9216 | 0.7833 | 0.37 |
| RandomForest | unscaled | 0.8422 | 0.9486 | 0.8496 | 0.9057 | 0.8000 | 0.56 |
| XGBoost | unscaled | 0.8041 | 0.9620 | **0.8673** | **0.9245** | **0.8167** | 0.34 |
| SVC | scaled | 0.8044 | 0.9389 | 0.7850 | 0.8936 | 0.7000 | 0.00 |
| LogisticRegression | scaled | 0.8027 | 0.9611 | 0.7010 | 0.9189 | 0.5667 | 0.05 |
| KNN | scaled | 0.7631 | 0.9248 | 0.8319 | 0.8868 | 0.7833 | 0.41 |
| SGDClassifier | scaled | 0.7876 | 0.9690 | 0.6452 | 0.9091 | 0.5000 | 0.00 |
| GradientBoosting | unscaled | 0.6697 | 0.9178 | 0.7934 | 0.7869 | 0.8000 | 0.31 |
| LightGBM | unscaled | 0.2167 | 0.7566 | 0.3711 | 0.2687 | 0.6000 | 1.00 |

**Best by PR-AUC:** MLPClassifier (0.8518)  
**Best by F1:** XGBoost (0.8673)  
**Best by Precision:** XGBoost (0.9245)  
**Best by Recall:** XGBoost (0.8167)

**Anomaly detectors (supplementary only):** PR-AUC ranges 0.0288–0.1500 — not suitable as primary classifiers.

---

## 6. Statistical Validation

Source: `outputs/stats/fraud_detection/ab_test.json`

### McNemar Test Results (3 comparisons)

| Comparison | Method | n_discordant | p-value | Holm p | Significant (0.05) | Effect Size (Cohen's h) | Winner |
|---|---|---|---|---|---|---|---|
| RandomForest vs MLPClassifier | chi2_fallback | 12 | 0.7055 | 0.7055 | No | +0.0000 | tie |
| RandomForest vs SVC | chi2_fallback | 7 | 0.008151 | 0.008151 | Yes | +0.008528 | RandomForest |
| MLPClassifier vs SVC | chi2_fallback | 11 | 0.0114 | 0.0114 | Yes | +0.0100 | MLPClassifier |

**Result:** 3 comparisons, 2 significant at Holm 0.05.  
**Interpretation:** RandomForest and MLPClassifier are each significantly superior to SVC, but RandomForest vs MLPClassifier is not statistically distinguishable. This is **not a single-champion claim** — there is a statistically defensible ordering, but no single model can be declared the winner. For production, XGBoost is the recommended default under the (10,1) fraud cost matrix because its precision (0.9245) and recall (0.8167) minimize expected cost.

---

## 7. Business Cost Analysis

Under the (10, 1) fraud cost matrix (false negative = 10× false negative cost):

| Model | PR-AUC | F1 | Precision | Recall | Expected Cost/Case |
|---|---|---|---|---|---|
| XGBoost | 0.8041 | 0.8673 | 0.9245 | 0.8167 | ~€11 |
| MLPClassifier | 0.8518 | 0.8468 | 0.9216 | 0.7833 | ~€22 |

**Trade-off:** XGBoost's higher recall (0.8167 vs 0.7833) means fewer missed fraud cases, which dominates the cost calculation despite marginally lower PR-AUC.

---

## 8. Explainability Artifacts

`explainability/` contains SHAP-based explanations for all 10 supervised models:

| Model | Empirical Cases | Hypothetical Cases |
|---|---|---|
| GradientBoosting__unscaled | 2 | 3 |
| IsolationForest__anomaly | — | 2 |
| KNN__scaled | — | — |
| LogisticRegression__scaled | — | — |
| MLPClassifier__scaled | — | — |
| RandomForest__unscaled | — | — |
| SGDClassifier__scaled | — | — |
| SVC__scaled | — | — |
| XGBoost__unscaled | — | — |
| LOF__anomaly | — | 2 |
| OneClassSVM__anomaly | — | 2 |

Each case file includes:
- Score and threshold
- Top SHAP features with values, directions, and explanations
- Recommendation (action, priority, confidence, estimated cost)
- Reviewer notes template (`[NAME]`, `[DATE]`)
- Provenance (model, group, row_id, split, recorded_at, hashes)

Waterfalls are computed on the **validation split only** — never on test.

---

## 9. How to Regenerate

```bash
# Train fraud models
make train-fraud

# Generate fraud-specific figures
make figures
# (filters to fraud_*.png via render_doc_figures.py)

# Run fraud explainability
make explain-fraud
# or: python -m spam_fraud_detector.main --dataset fraud --explainer shap --explain-only --persona fraud

# Run fraud statistical validation
make stats-fraud
# or: python -m spam_fraud_detector.main --dataset fraud --stats-only

# Run anomaly explainability
make explain-anomaly
# or: python -m spam_fraud_detector.main --dataset fraud --explainer shap --explain-only --persona anomaly
```

---

## 10. Related Documentation

| Document | Location |
|---|---|
| Fraud Case Study | `spam_fraud_detector/reports/docs/FRAUD_DETECTION.md` |
| Fraud Framework | `spam_fraud_detector/reports/fraud_detection/docs/FRAMEWORK.md` |
| Framework Whitepaper | `spam_fraud_detector/reports/docs/FRAMEWORK.md` |
| Evidence Inventory | `spam_fraud_detector/reports/docs/EVIDENCE_INVENTORY.json` |
| Model Validation Report | `spam_fraud_detector/reports/docs/MODEL_VALIDATION_REPORT.md` |
| Explainability Framework | `spam_fraud_detector/reports/docs/EXPLAINABILITY.md` |

---

## 10. License

MIT © Guy M. Kaptue T. See [../../../../LICENSE](../../../../../../LICENSE).
