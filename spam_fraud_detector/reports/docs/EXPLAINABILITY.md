# Explainability Framework
### SHAP-Based Model Explainability for Three Personas

## 1. Overview

The framework provides three persona-aware explainers for model explainability:

| Explainer | Backend | Use Case | Persona |
|---|---|---|---|
| `ModelExplainer` | SHAP TreeExplainer/KernelExplainer | Tree/linear models | Fraud Analyst |
| `TextExplainer` | SHAP mapped to TF-IDF tokens | Spam text models | Trust & Safety |
| `AnomalyExplainer` | PermutationExplainer on `decision_function` | Anomaly detectors | Risk Engineer |

Waterfalls are computed on the **validation split only**, never on test. Case files include recommendations (block/escalate/review/monitor/clear) with reviewer notes templates containing `[NAME]` and `[DATE]` placeholders.

Artifacts are stored in `reports/<domain>/docs/explainability/` as PNG waterfalls and JSON case files.

## 2. Fraud Analyst Persona

**TreeSHAP** explanations for tree-based fraud models (RandomForest, XGBoost, GradientBoosting, LightGBM).

### Example: False Negative Case F013398

Source: `reports/fraud_detection/docs/explainability/GradientBoosting__unscaled__fraud__case_F013398.json`

| Field | Value |
|---|---|
| Subject | Yuki Tanaka |
| Model | GradientBoosting (unscaled) |
| Score | 0.101 (threshold 0.31) |
| Ground truth | Fraud |
| Prediction | Legitimate (missed) |
| Error type | False negative |
| Recommendation | Escalate (high priority) |

**Top SHAP drivers:**

| Feature | SHAP Value | Direction | Feature Value |
|---|---|---|---|
| V27 | -0.0485 | Decreases fraud risk | 0.0999 |
| V28 | +0.0485 | Increases fraud risk | 0.1230 |
| V7 | -0.0220 | Decreases fraud risk | 0.3257 |
| V8 | +0.0220 | Increases fraud risk | -0.0377 |
| Amount | +0.0124 | Increases fraud risk | 1.0 |

**Narrative:** "Yuki Tanaka's transaction was scored 0.101 (threshold 0.31)."

**Recommended next steps:**
1. Investigate why V27 (-0.05), V28 (+0.05), V7 (-0.02) did not trigger the model.
2. Add this row to the training set with the fraud label.
3. Consider lowering the threshold or adjusting feature engineering for V27.

### Example: Hypothetical Case H-FRAUD-001

Source: `reports/fraud_detection/docs/explainability/GradientBoosting__unscaled__fraud__hypo_H-FRAUD-001.json`

Hypothetical fraud scenarios are generated with deterministic subjects (seed=42) and scored by the model. Case files include score, top SHAP drivers, and recommendation.

## 3. Trust & Safety Persona

**Token-level SHAP** explanations for TF-IDF spam models. SHAP values are mapped to individual tokens, showing which words increase or decrease spam probability.

### Example: False Negative Case F000218

Source: `reports/spam_detection/docs/explainability/GradientBoosting__unscaled__spam__case_F000218.json`

| Field | Value |
|---|---|
| Subject | Rosa Petrov |
| Model | GradientBoosting (unscaled) |
| Score | 0.458 (threshold 0.46) |
| Ground truth | Spam |
| Prediction | Legitimate (missed) |
| Error type | False negative |
| Recommendation | Monitor (low priority) |

**Top token drivers:**

| Token | SHAP Value | Direction | Count |
|---|---|---|---|
| hotmail | -1.0793 | Decreases spam probability | 3 |
| money | +0.7885 | Increases spam probability | 3 |
| hanks | -0.6079 | Decreases spam probability | 3 |
| remove | +0.4986 | Increases spam probability | 2 |
| enron | +0.4977 | Increases spam probability | 0 |

**Narrative:** "Rosa Petrov's message was scored 0.458 (threshold 0.46)."

**Recommended next steps:**
1. Add sender to watchlist for 14 days.
2. Review if 2+ more messages from same sender are flagged.

### Example: Hypothetical Case H-SPAM-001

Source: `reports/spam_detection/docs/explainability/GradientBoosting__unscaled__spam__hypo_H-SPAM-001.json`

| Field | Value |
|---|---|
| Subject | Hattie Kowalski |
| Scenario | Aggressive promotional message |
| Score | 0.926 (threshold 0.46) |
| Recommendation | Review (medium priority) |

**Top token drivers:** "t" (+0.67), "s" (+0.67), "enron" (+0.58), "the" (+0.48), "hanks" (+0.36)

## 4. Risk Engineer Persona

**Permutation SHAP** on `decision_function` for anomaly detectors (IsolationForest, LOF, OneClassSVM).

### Anomaly Detector Performance (Spam)

| Model | PR-AUC | ROC-AUC |
|---|---|---|
| IsolationForest | 0.3395 | 0.5535 |
| LOF | 0.3435 | 0.5971 |
| OneClassSVM | 0.3506 | 0.5305 |

### Anomaly Detector Performance (Fraud)

| Model | PR-AUC | ROC-AUC |
|---|---|---|
| IsolationForest | 0.1500 | 0.9543 |
| LOF | 0.1133 | 0.9624 |
| OneClassSVM | 0.0288 | 0.7602 |

**Finding:** Anomaly detectors achieve high ROC-AUC on fraud (up to 0.9624) but very low PR-AUC (up to 0.1500), indicating they rank fraud highly among all samples but have poor precision at the positive class. They are suitable as supplementary signals, not primary classifiers.

### 4.A What the anomaly SHAP values mean (and why they look small)

The framework reports SHAP values for both supervised models (RandomForest, XGBoost, etc.) and unsupervised anomaly detectors (IsolationForest, LOF, OneClassSVM). **The two are not comparable in magnitude.**

#### Supervised model SHAP values

For a supervised classifier, `shap_values(X)` decomposes the model's **predicted probability** into per-feature contributions that sum to `f(x) - E[f(X)]`. Units are *probability units*, directly interpretable: a SHAP value of +0.15 on `transaction_amount` means that feature pushed fraud probability up by 15 percentage points relative to the base rate.

#### Anomaly detector SHAP values

For an unsupervised anomaly detector, there is no predicted probability to decompose. The framework uses `shap.KernelExplainer` with **permutation-based perturbation** of the detector's decision function (raw `score_samples` for IsolationForest, `decision_function` for LOF/OneClassSVM). Consequences:

1. **Units are decision-function units, not probability units.** IsolationForest's decision function ranges roughly `[-0.5, 0.5]`; SHAP 0.01 is not "1% probability" — it is 0.01 decision-function units, meaningful for isolation path length but tiny in absolute terms.
2. **Base value is the mean decision-function score**, near zero by construction (most points normal). Small perturbations around near-zero base produce small SHAP values.
3. **Magnitude is sensitive to background sample count** in `KernelExplainer`. Framework uses 100 background samples by default — enough for *relative ranking*, not absolute-magnitude claims.

#### How to read them

| Use case | Supervised SHAP | Anomaly SHAP |
|---|---|---|
| "Which features drive this case?" | Top-3 by absolute value | Top-3 by **rank**, not absolute value |
| "How much did feature X contribute?" | Signed value in probability units | **Do not read absolute value** — not probability |
| "Compare two models' explanations" | Valid — both in probability units | Valid only *within* same anomaly detector |
| "Report to a regulator" | Yes — interpretable units | Only as "detector flagged this because of features A, B, C" — never with numeric magnitude |

**Bottom line:** near-zero anomaly SHAP values are *expected*, not a bug. They answer "which features mattered for the anomaly score," not "how much probability did this feature add." See `MODEL_VALIDATION_REPORT.md` §7.2 for why anomaly detectors are supplementary-only.


### Example: Hypothetical Anomaly Case H-ANOM-001

Source: `reports/fraud_detection/docs/explainability/IsolationForest__anomaly__anomaly__hypo_H-ANOM-001.json`

Hypothetical anomaly cases are scored using permutation-based SHAP on the anomaly detector's decision function.

## 5. Explainability Artifacts Inventory

| Domain | Model | Case Type | Count |
|---|---|---|---|
| spam_detection | GradientBoosting__unscaled | empirical | 2 |
| spam_detection | GradientBoosting__unscaled | hypothetical | 3 |
| spam_detection | KNN__scaled | empirical | 2 |
| fraud_detection | GradientBoosting__unscaled | empirical | 2 |
| fraud_detection | GradientBoosting__unscaled | hypothetical | 3 |
| fraud_detection | IsolationForest__anomaly | hypothetical | 2 |

Each case file includes:
- Score and threshold
- Top SHAP features/tokens with values and directions
- Recommendation (action, priority, confidence, estimated cost)
- Reviewer notes template with `[NAME]` and `[DATE]`
- Suggested next steps
- Subject metadata (name, city, email, etc.)
- Provenance (model, group, row_id, split, recorded_at, hashes)

## 6. How to Regenerate

```bash
# Run explainability for all personas
make explain

# Or per domain
make explain-spam
make explain-fraud
make explain-anomaly
```

All explainability artifacts are deterministic given the same model, background data, and random seed.

## 7. Reading a Case File in 60 Seconds

The `reports/<domain>/docs/explainability/` directory contains one case file per flagged test-split case. Each is a JSON file. Here is how to read one in under a minute, in order.

### Step 1 — Identify the case (5 seconds)

```json
{
  "case_id": "fraud_test_00731",
  "model": "xgboost",
  "predicted_proba": 0.9123,
  "threshold": 0.34,
  "flagged": true,
  "actual_label": 1
}
```

**Read:** model flagged this case at 91.2% fraud probability, above 34% cost-optimal threshold, and it was in fact fraud. True positive.

### Step 2 — Read the top SHAP features (15 seconds)

```json
{
  "top_features": [
    {"name": "V14", "shap_value": 0.41, "feature_value": -3.87},
    {"name": "V17", "shap_value": 0.28, "feature_value": -5.12},
    {"name": "V12", "shap_value": 0.19, "feature_value": -6.04}
  ]
}
```

**Read:** V14 pushed probability up 41 points, V17 up 28, V12 up 19. These three account for 88 of 91 points above base rate. Compact, defensible explanation — driven by three strong PCA components.

### Step 3 — Check the counterfactual (20 seconds)

```json
{
  "counterfactual": {
    "features_to_change": ["V14"],
    "change_magnitude": 1.2,
    "new_proba": 0.28,
    "new_flagged": false
  }
}
```

**Read:** if V14 alone were perturbed by 1.2 standard deviations, case drops below threshold (28% < 34%). **This tells the analyst which single feature to interrogate first.** Actionable lead: "look at V14."

### Step 4 — Check the anomaly signal (10 seconds)

```json
{
  "anomaly": {
    "isolation_forest_score": -0.083,
    "lof_score": -2.1,
    "oneclass_svm_score": -0.4,
    "consensus": "isolation_forest and lof agree"
  }
}
```

**Read:** IsolationForest and LOF both flag as anomalous; OneClassSVM does not. **Consensus** field tells whether detectors agree. **Do not read scores in isolation** — different units (§4.A). Read consensus.

### Step 5 — Record the audit line (10 seconds)

```json
{
  "audit": {
    "evidence_inventory_path": "docs/EVIDENCE_INVENTORY.json",
    "stats_file": "outputs/stats/fraud_detection/ab_test.json",
    "model_run_id": "mlruns/2/abc123...",
    "generated_at": "2026-01-15T10:32:11Z"
  }
}
```

**Read:** every claim traceable to evidence inventory, stats file, MLflow run. **This is the auditable line.** Regulator asks "where did this explanation come from?" — here is the answer.

**Total: under 60 seconds.** Case file designed so analyst goes from "model flagged this" to "here is what to investigate and the audit trail" without opening a notebook.
