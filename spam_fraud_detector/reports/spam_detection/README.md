# Spam Detection Reports

[![CI](https://github.com/GuyKaptue/spam-fraud-detector/actions/workflows/ci.yml/badge.svg)](https://github.com/GuyKaptue/spam-fraud-detector/actions/workflows/ci.yml)

**Author:** Guy M. Kaptue T. — [GitHub](https://github.com/GuyKaptue) — guykaptue24@gmail.com  
**License:** MIT · **Version:** 5.0.0  
**Domain:** Spam Detection (binary classification, moderate class imbalance)

---

## 1. Purpose

This directory contains all reports, figures, and explainability artifacts for the **spam detection** domain of the spam-fraud-detector framework. It documents the end-to-end experimental protocol, model comparison results, statistical validation, and SHAP-based explainability for spam vs. legitimate email classification.

All findings are grounded in actual experimental outputs stored in `spam_fraud_detector/reports/docs/EVIDENCE_INVENTORY.json` and `outputs/stats/spam_detection/`.

---

## 2. Repository Map

```
reports/spam_detection/
├── README.md                    # This file
├── docs/
│   ├── FRAMEWORK.md             # Spam-specific framework documentation
│   └── SPAM_DETECTION.md        # Spam detection case study
├── figures/                     # Spam-specific figures (8 PNGs)
│   ├── spam_class_distribution.png
│   ├── spam_threshold_curve.png
│   ├── spam_roc_curve.png
│   ├── spam_pr_curve.png
│   ├── spam_pr_auc_by_model.png
│   ├── spam_threshold_sensitivity.png
│   ├── spam_holm_significance.png
│   └── spam_roc_vs_pr.png
└── explainability/              # SHAP waterfalls and case files
    ├── GradientBoosting__unscaled__spam__case_*.json/png
    ├── KNN__scaled__spam__case_*.json/png
    ├── LightGBM__unscaled__spam__case_*.json/png
    ├── LogisticRegression__scaled__spam__case_*.json/png
    ├── MLPClassifier__scaled__spam__case_*.json/png
    ├── MultinomialNB__unscaled__spam__case_*.json/png
    ├── RandomForest__unscaled__spam__case_*.json/png
    ├── SGDClassifier__scaled__spam__case_*.json/png
    ├── SVC__scaled__spam__case_*.json/png
    └── XGBoost__unscaled__spam__case_*.json/png
```

---

## 3. Dataset Overview

| Property | Value |
|---|---|
| Source | Kaggle Spam Mails |
| Total samples | 5,172 |
| Test samples | 773 |
| Positive (spam) | 214 (27.7%) |
| Negative (legitimate) | 559 (72.3%) |
| Primary metric | PR-AUC |
| Secondary metrics | F1, ROC-AUC, Precision, Recall |
| Class imbalance | Moderate |

---

## 4. Candidate Models

Ten supervised estimators were evaluated across two preprocessing regimes:

**Scaled group (StandardScaler → estimator):**

| Model | Threshold |
|---|---|
| LogisticRegression | 0.76 |
| KNN | 0.81 |
| SVC | 0.00 |
| SGDClassifier | 0.00 |
| MLPClassifier | 0.85 |

**Unscaled group (estimator only):**

| Model | Threshold |
|---|---|
| RandomForest | 0.46 |
| GradientBoosting | 0.46 |
| XGBoost | 0.35 |
| LightGBM | 0.51 |
| MultinomialNB | 0.02 |

**Anomaly detection baseline (unsupervised):**

| Model | Threshold |
|---|---|
| IsolationForest | 0.35 |
| LOF | 0.05 |
| OneClassSVM | 0.34 |

---

## 5. Performance Results

| Model | Group | PR-AUC | ROC-AUC | F1 | Precision | Recall | Threshold |
|---|---|---|---|---|---|---|---|
| RandomForest | unscaled | **0.9885** | **0.9955** | **0.9447** | 0.9318 | 0.9579 | 0.46 |
| XGBoost | unscaled | 0.9857 | 0.9945 | 0.9385 | 0.9156 | 0.9626 | 0.35 |
| LightGBM | unscaled | 0.9824 | 0.9938 | 0.9438 | 0.9091 | **0.9813** | 0.51 |
| GradientBoosting | unscaled | 0.9846 | 0.9943 | 0.9340 | 0.9429 | 0.9252 | 0.46 |
| MLPClassifier | scaled | 0.9592 | 0.9910 | 0.9399 | 0.8979 | 0.9860 | 0.85 |
| LogisticRegression | scaled | 0.9536 | 0.9884 | 0.9252 | 0.8987 | 0.9533 | 0.76 |
| SGDClassifier | scaled | 0.9575 | 0.9847 | 0.8534 | **0.9702** | 0.7617 | 0.00 |
| SVC | scaled | 0.9764 | 0.9911 | 0.8579 | 0.9595 | 0.7757 | 0.00 |
| MultinomialNB | unscaled | 0.8977 | 0.9755 | 0.8814 | 0.8455 | 0.9206 | 0.02 |
| KNN | scaled | 0.7406 | 0.9178 | 0.7102 | 0.5710 | 0.9393 | 0.81 |

**Best by PR-AUC:** RandomForest (0.9885)  
**Best by Recall:** MLPClassifier (0.9860)  
**Best by Precision:** SGDClassifier (0.9702)

**Anomaly detectors (supplementary only):** PR-AUC ranges 0.3395–0.3506 — not suitable as primary classifiers.

---

## 6. Statistical Validation

Source: `outputs/stats/spam_detection/ab_test.json`

### McNemar Test Results (3 comparisons)

| Comparison | Method | n_discordant | p-value | Holm p | Significant (0.05) | Effect Size (Cohen's h) | Winner |
|---|---|---|---|---|---|---|---|
| RandomForest vs XGBoost | chi2_fallback | 12 | 1.0000 | 1.0000 | No | +0.0000 | tie |
| RandomForest vs GradientBoosting | chi2_fallback | 21 | 0.5127 | 1.0000 | No | +0.0230 | RandomForest |
| XGBoost vs GradientBoosting | chi2_fallback | 21 | 0.5127 | 1.0000 | No | +0.0230 | XGBoost |

**Result:** 3 comparisons, 0 significant at Holm 0.05.  
**Interpretation:** The top-3 spam models are **statistically indistinguishable**. The PR-AUC gap between RandomForest (0.9885) and XGBoost (0.9857) is 0.0028 — real, but within sampling noise for n=773 with 214 positives. This is a **positive finding**: spam detection is saturated on this dataset; any of the three models would serve equally well in production. Model selection should be based on operational factors (inference latency, interpretability, maintenance cost), not marginal metric differences.

---

## 7. Explainability Artifacts

`explainability/` contains SHAP-based explanations for all 10 supervised models:

| Model | Empirical Cases | Hypothetical Cases |
|---|---|---|
| GradientBoosting__unscaled | 2 | 3 |
| KNN__scaled | 2 | 3 |
| LightGBM__unscaled | — | — |
| LogisticRegression__scaled | — | — |
| MLPClassifier__scaled | — | — |
| MultinomialNB__unscaled | — | — |
| RandomForest__unscaled | — | — |
| SGDClassifier__scaled | — | — |
| SVC__scaled | — | — |
| XGBoost__unscaled | — | — |

Each case file includes:
- Score and threshold
- Top SHAP features with values, directions, and explanations
- Recommendation (action, priority, confidence, estimated cost)
- Reviewer notes template (`[NAME]`, `[DATE]`)
- Provenance (model, group, row_id, split, recorded_at, hashes)

Waterfalls are computed on the **validation split only** — never on test.

---

## 8. How to Regenerate

```bash
# Train spam models
make train-spam

# Generate spam-specific figures
make figures
# (filters to spam_*.png via render_doc_figures.py)

# Run spam explainability
make explain-spam
# or: python -m spam_fraud_detector.main --dataset spam --explainer shap --explain-only --persona spam

# Run spam statistical validation
make stats-spam
# or: python -m spam_fraud_detector.main --dataset spam --stats-only
```

---

## 9. Related Documentation

| Document | Location |
|---|---|
| Spam Case Study | `spam_fraud_detector/reports/docs/SPAM_DETECTION.md` |
| Spam Framework | `spam_fraud_detector/reports/spam_detection/docs/FRAMEWORK.md` |
| Framework Whitepaper | `spam_fraud_detector/reports/docs/FRAMEWORK.md` |
| Evidence Inventory | `spam_fraud_detector/reports/docs/EVIDENCE_INVENTORY.json` |
| Model Validation Report | `spam_fraud_detector/reports/docs/MODEL_VALIDATION_REPORT.md` |
| Explainability Framework | `spam_fraud_detector/reports/docs/EXPLAINABILITY.md` |

---

## 10. License

MIT © Guy M. Kaptue T. See [../../../../LICENSE](../../../../../../LICENSE).
