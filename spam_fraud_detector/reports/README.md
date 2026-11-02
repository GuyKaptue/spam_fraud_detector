# reports Directory

[![CI](https://github.com/GuyKaptue/spam-fraud-detector/actions/workflows/ci.yml/badge.svg)](https://github.com/GuyKaptue/spam-fraud-detector/actions/workflows/ci.yml)

**Author:** Guy M. Kaptue T. — [GitHub](https://github.com/GuyKaptue) — guykaptue24@gmail.com  
**License:** MIT · **Version:** 5.0.0

---

## 1. Purpose

The `reports/` directory contains all generated and authored documentation artifacts for the spam-fraud-detector framework. It is organized by **domain** (spam_detection, fraud_detection) and contains both **generated** artifacts (figures, explainability waterfalls, case files) and **authored** documentation (FRAMEWORK.md, PAPER.md, SPAM_DETECTION.md, FRAUD_DETECTION.md, EXPLAINABILITY.md, MODEL_VALIDATION_REPORT.md).

All reports are grounded in actual experimental outputs stored in `docs/EVIDENCE_INVENTORY.json` and `outputs/`. Per the v5.0 evidence-first documentation rule, no documentation claim may reference assumed, estimated, or illustrative findings — every number must trace to a generated artifact.

---

## 2. Repository Map

```
reports/
├── README.md                    # This file
├── docs/                        # Authored documentation (domain-agnostic)
│   ├── README.md
│   ├── FRAMEWORK.md             # Architecture whitepaper
│   ├── PAPER.md                 # Publication-ready results
│   ├── SPAM_DETECTION.md        # Spam domain case study
│   ├── FRAUD_DETECTION.md       # Fraud domain validation report
│   ├── EXPLAINABILITY.md        # Three-persona explainability framework
│   ├── MODEL_VALIDATION_REPORT.md # Industry-grade validation report
│   ├── CORE_README_STYLE.md     # Core module authoring rules
│   ├── MAKEFILE.md              # Makefile target reference
│   ├── BADGES.md                # CI badge reference
│   ├── DATA_PIPELINE.md         # Six-stage data pipeline
│   └── EVIDENCE_INVENTORY.json  # Machine-readable evidence (25 models, 2 domains)
├── figures/                     # Generated documentation figures (23 PNGs + 4 GIFs)
│   ├── README.md
│   ├── REGISTRY.json            # Figure registry (30 entries)
│   ├── *.png                    # Static figures (hero, curves, distributions, etc.)
│   └── gifs/                    # Animated figure galleries (4 GIFs)
│       ├── spam_models.gif
│       ├── fraud_models.gif
│       ├── framework_artifacts.gif
│       └── readme_hero.gif
├── spam_detection/              # Spam domain reports
│   ├── README.md
│   ├── docs/
│   │   ├── FRAMEWORK.md
│   │   └── SPAM_DETECTION.md
│   ├── figures/                 # Spam-specific figures
│   └── explainability/          # Spam SHAP waterfalls and case files
├── fraud_detection/             # Fraud domain reports
│   ├── README.md
│   ├── docs/
│   │   ├── FRAMEWORK.md
│   │   └── FRAUD_DETECTION.md
│   ├── figures/                 # Fraud-specific figures
│   └── explainability/          # Fraud SHAP waterfalls and case files
└── .gitignore
```

---

## 3. Domain Structure

Each domain directory (`spam_detection/`, `fraud_detection/`) follows the same internal structure:

| Subdirectory | Contents |
|---|---|
| `docs/` | Domain-specific FRAMEWORK.md and case study (SPAM_DETECTION.md / FRAUD_DETECTION.md) |
| `figures/` | Domain-specific ROC, PR, threshold, class distribution, Holm significance figures |
| `explainability/` | SHAP waterfall PNGs and JSON case files for empirical and hypothetical cases |

### Spam Detection Domain

- **Dataset:** Kaggle Spam Mails (5,172 samples, 27.7% positive)
- **Test set:** 773 samples (214 spam, 559 legitimate)
- **Primary metric:** PR-AUC
- **Best model:** RandomForest (unscaled, PR-AUC = 0.9885)
- **Statistical result:** 3 McNemar comparisons, 0 significant at Holm 0.05 — top-3 models are statistically indistinguishable

### Fraud Detection Domain

- **Dataset:** Kaggle Credit Card Fraud (284,807 samples, 0.17% positive)
- **Test set:** 34,857 samples (60 fraud, 34,797 legitimate)
- **Primary metric:** PR-AUC / AUPRC
- **Best model:** MLPClassifier (scaled, PR-AUC = 0.8518) for PR-AUC; XGBoost (unscaled, F1 = 0.8673) for cost balance
- **Statistical result:** 3 McNemar comparisons, 2 significant at Holm 0.05 — RandomForest and MLPClassifier both significantly better than SVC

---

## 4. Evidence Inventory

`reports/docs/EVIDENCE_INVENTORY.json` is the **single source of truth** for every documented claim. It contains:

| Section | Contents |
|---|---|
| `models` | 25 models across 2 domains with meta, eval, curve, confusion, thresholds |
| `stats` | McNemar A/B test results with Holm correction per domain |
| `dashboard` | 16 fact/dimension tables (fact_metrics, fact_stats, dim_method, etc.) |
| `explainability` | Case file metadata per domain and persona |

Regenerate with:
```bash
make evidence
# or: python scripts/collect_evidence.py
```

---

## 5. Figure Gallery

`reports/docs/figures/` contains 23 static PNG figures and 4 animated GIFs:

| Category | Files |
|---|---|
| Hero / Results | `readme_hero_results.png`, `readme_results_table.png`, `readme_personas_triptych.png` |
| Framework | `framework_pipeline_diagram.png`, `framework_layer_order.png`, `framework_mcnemar_decision_tree.png`, `framework_threshold_cost_curve.png` |
| Spam | `spam_class_distribution.png`, `spam_threshold_curve.png`, `spam_roc_curve.png`, `spam_pr_curve.png`, `spam_pr_auc_by_model.png`, `spam_threshold_sensitivity.png`, `spam_holm_significance.png`, `spam_roc_vs_pr.png` |
| Fraud | `fraud_class_distribution.png`, `fraud_threshold_curve.png`, `fraud_roc_curve.png`, `fraud_pr_curve.png`, `fraud_pr_auc_by_model.png`, `fraud_threshold_sensitivity.png`, `fraud_holm_significance.png`, `fraud_roc_vs_pr.png` |
| Animated | `spam_models.gif`, `fraud_models.gif`, `framework_artifacts.gif`, `readme_hero.gif` |

Regenerate all figures with:
```bash
make figures
# or: python scripts/render_doc_figures.py
make gifs
# or: python scripts/make_gifs.py
```

---

## 6. Explainability Artifacts

`reports/<domain>/docs/explainability/` contains SHAP-based explainability artifacts:

| Artifact Type | Description |
|---|---|
| Empirical case files | JSON case files for real test-set instances with SHAP explanations |
| Hypothetical case files | JSON case files for synthetic subjects with SHAP explanations |
| Waterfall PNGs | Per-instance SHAP waterfall visualizations |
| Summary PNGs | Per-model SHAP summary beeswarm plots |

Three personas are supported:
- **Fraud Analyst** — TreeSHAP for tree-based fraud models
- **Trust & Safety** — Token-level SHAP for TF-IDF spam models
- **Risk Engineer** — Permutation SHAP for anomaly detectors

---

## 7. How to Regenerate All Reports

```bash
# Step 1: Collect evidence from outputs/
make evidence

# Step 2: Render all figures from evidence inventory
make figures
make gifs

# Step 3: Verify docs match inventory
make docs-verify

# Step 4: Run explainability
make explain

# Full pipeline (train + stats + explain + verify)
make pipeline
```

---

## 8. Git Ignore

Generated artifacts under `reports/*/figures/` and `reports/*/docs/generated/` are gitignored. Only authored documentation (`.md` files), `EVIDENCE_INVENTORY.json`, `REGISTRY.json`, and the `gifs/` directory are tracked.

```
# Generated reports (gitignored)
reports/*/figures/
reports/*/docs/generated/

# Exception: documentation figures are committed
!spam_fraud_detector/reports/docs/figures/
!spam_fraud_detector/reports/docs/figures/gifs/
```

---

## 9. License

MIT © Guy M. Kaptue T. See [../../../LICENSE](../../../../LICENSE).
