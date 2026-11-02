# Spam Detection
### Case Study: Statistically Validated Model Comparison

## 1. Business Context

Spam detection is a binary classification problem where the cost of false positives (blocking legitimate email) and false negatives (spam reaching the inbox) must be balanced. The dataset contains 5,172 emails with ~29% spam prevalence. On the held-out test set (n=773), 214 messages are spam (27.7%).

## 2. Dataset Overview

| Property | Value |
|---|---|
| Total samples | 5,172 |
| Test samples | 773 |
| Positive (spam) | 214 (27.7%) |
| Negative (legitimate) | 559 (72.3%) |
| Primary metric | PR-AUC |
| Secondary metrics | F1, ROC-AUC, Precision, Recall |

## 3. Experimental Design

Competing classifiers are evaluated on identical train/validation/test splits under a controlled and reproducible environment:

1. Train multiple candidate models (10 estimators across scaled and unscaled groups).
2. Generate predictions on the same holdout test set.
3. Compare model performance using PR-AUC as the primary metric (appropriate for the moderate class imbalance).
4. Apply McNemar significance testing with Holm multiple-comparison correction to determine whether observed differences are statistically meaningful.
5. Record all results through MLflow for reproducibility and auditability.

## 4. Candidate Models

Ten estimators were evaluated, grouped by preprocessing regime:

**Scaled group (StandardScaler → estimator):**
- LogisticRegression (threshold 0.76)
- KNN (threshold 0.81)
- SVC (threshold 0.00)
- SGDClassifier (threshold 0.00)
- MLPClassifier (threshold 0.85)

**Unscaled group (estimator only):**
- RandomForest (threshold 0.46)
- GradientBoosting (threshold 0.46)
- XGBoost (threshold 0.35)
- LightGBM (threshold 0.51)
- MultinomialNB (threshold 0.02)

**Anomaly detection baseline (unsupervised):**
- IsolationForest (threshold 0.35)
- LOF (threshold 0.05)
- OneClassSVM (threshold 0.34)

## 5. Performance Results

Source: `docs/EVIDENCE_INVENTORY.json`, `outputs/dashboard/fact_metrics.csv`

| Model | Group | PR-AUC | ROC-AUC | F1 | Accuracy | Precision | Recall | Threshold |
|---|---|---|---|---|---|---|---|---|
| RandomForest | unscaled | **0.9885** | **0.9955** | **0.9447** | 0.9690 | 0.9318 | 0.9579 | 0.46 |
| XGBoost | unscaled | 0.9857 | 0.9945 | 0.9385 | 0.9651 | 0.9156 | 0.9626 | 0.35 |
| LightGBM | unscaled | 0.9824 | 0.9938 | 0.9438 | 0.9677 | 0.9091 | **0.9813** | 0.51 |
| GradientBoosting | unscaled | 0.9846 | 0.9943 | 0.9340 | 0.9638 | 0.9429 | 0.9252 | 0.46 |
| MLPClassifier | scaled | 0.9592 | 0.9910 | 0.9399 | 0.9651 | 0.8979 | 0.9860 | 0.85 |
| LogisticRegression | scaled | 0.9536 | 0.9884 | 0.9252 | 0.9573 | 0.8987 | 0.9533 | 0.76 |
| SGDClassifier | scaled | 0.9575 | 0.9847 | 0.8534 | 0.9276 | **0.9702** | 0.7617 | 0.00 |
| SVC | scaled | 0.9764 | 0.9911 | 0.8579 | 0.9288 | 0.9595 | 0.7757 | 0.00 |
| MultinomialNB | unscaled | 0.8977 | 0.9755 | 0.8814 | 0.9314 | 0.8455 | 0.9206 | 0.02 |
| KNN | scaled | 0.7406 | 0.9178 | 0.7102 | 0.7878 | 0.5710 | 0.9393 | 0.81 |

## 6. Statistical Validation

Source: `outputs/stats/spam_detection/ab_test.json`

### McNemar Test Results

Pairwise McNemar tests on test-split predictions with Holm family-wise error rate correction. All comparisons used chi2_fallback (n_discordant < 25).

| Model A | Model B | n | n_discordant | n01 | n10 | p-value | Holm p | Significant (0.05) | Winner |
|---|---|---|---|---|---|---|---|---|---|
| RandomForest | XGBoost | 773 | 12 | 6 | 6 | 1.0000 | 1.0000 | No | tie |
| RandomForest | GradientBoosting | 773 | 21 | 9 | 12 | 0.5127 | 1.0000 | No | RandomForest |
| XGBoost | GradientBoosting | 773 | 21 | 9 | 12 | 0.5127 | 1.0000 | No | XGBoost |

### Summary

- **Comparisons:** 3
- **Significant (Holm 0.05):** 0
- **Significant (Holm 0.01):** 0
- **Winner:** No significant winner at Holm 0.05

### Interpretation

The top three spam models (RandomForest, XGBoost, GradientBoosting) show PR-AUC differences of less than 0.004 (0.9885 vs 0.9857 vs 0.9846), but none of these differences are statistically significant after Holm correction. The RandomForest vs XGBoost comparison shows a perfect tie (p=1.0000, n01=n10=6), indicating the models make identical numbers of opposing errors on the test set.

## 7. Error Analysis

### RandomForest (best PR-AUC = 0.9885, threshold 0.46)

Confusion matrix (test set):
- True Negatives: 543
- False Positives: 16
- False Negatives: 8
- True Positives: 206

At the expected-cost-optimal threshold (0.46), RandomForest misclassifies 16 legitimate messages as spam (false positives) and 8 spam messages as legitimate (false negatives).

### Key Failure Patterns

- **False positives (16):** Legitimate messages scored above threshold 0.46. SHAP analysis shows tokens like "money", "remove", "enron" drive spam probability upward even in legitimate contexts.
- **False negatives (8):** Spam messages scored below threshold 0.46. These messages lack strong spam-indicative tokens and resemble legitimate correspondence.

### LightGBM (highest recall = 0.9813)

LightGBM achieves the highest recall (0.9813) but at the cost of lower precision (0.9091 vs RandomForest's 0.9318). This trade-off means LightGBM catches more spam but blocks more legitimate email.

### SGDClassifier (highest precision = 0.9702)

SGDClassifier achieves the highest precision (0.9702) but the lowest recall (0.7617), meaning it blocks very little legitimate email but misses ~24% of spam.

## 8. Explainability Findings

SHAP-based explainability is available for all supervised models. Example case file (GradientBoosting, case F000218, subject "Rosa Petrov"):

- **Score:** 0.458 (threshold 0.46) — near threshold, recommended action: monitor
- **Top drivers increasing spam probability:** "money" (+0.79), "remove" (+0.50)
- **Top drivers decreasing spam probability:** "hotmail" (-1.08), "hanks" (-0.61)

Hypothetical case H-SPAM-001 (subject "Hattie Kowalski", score 0.926):
- **Top tokens increasing spam probability:** "t" (+0.67), "s" (+0.67), "enron" (+0.58)
- **Recommended action:** review

## 9. Operational Recommendations

1. **Deploy RandomForest** as the production model for spam detection — best PR-AUC (0.9885) and best F1 (0.9447) with balanced precision/recall.
2. **Threshold 0.46** is expected-cost-optimal for RandomForest (expected cost 0.0310 at equal FP/FN costs).
3. **No statistically significant difference** exists between RandomForest, XGBoost, and GradientBoosting — model selection can consider operational factors (inference speed, interpretability, maintenance cost) rather than marginal metric differences.
4. **Anomaly detectors are not suitable** as primary spam classifiers (PR-AUC 0.3395–0.3506); use only as supplementary signals.

## 10. Conclusion

RandomForest achieves the highest PR-AUC (0.9885) on the spam detection task, but the performance gap to XGBoost (0.9857) and GradientBoosting (0.9846) is not statistically significant after Holm correction. The framework's statistical validation layer prevents overclaiming based on marginal metric differences. All figures, metrics, and statistical tests are reproducible from `docs/EVIDENCE_INVENTORY.json` and `outputs/stats/spam_detection/ab_test.json`.

<!-- ab-summary:start -->
## Statistical Significance

Pairwise McNemar tests on test-split predictions, with Holm family-wise error rate correction.

- Comparisons: 3
- Significant (Holm 0.05): 0
- Significant (Holm 0.01): 0

### Pairwise comparisons

| Model A | Model B | Method | p-value | Holm p | Significant | Effect Size (h) | Winner |
|---------|---------|--------|---------|--------|-------------|-----------------|--------|
| RandomForest | XGBoost | chi2_fallback | 1 | 1 | ✗ | +0.000 | tie |
| RandomForest | GradientBoosting | chi2_fallback | 0.5127 | 1 | ✗ | +0.023 | RandomForest |
| XGBoost | GradientBoosting | chi2_fallback | 0.5127 | 1 | ✗ | +0.023 | XGBoost |

### Winner
No significant winner at Holm 0.05.

<!-- ab-summary:end -->
