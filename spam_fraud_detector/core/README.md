# Core Modules

Consolidated narrative documentation for every module in `spam_fraud_detector/core/`.
Per-module `*.README.md` files under `core/` are forbidden; this is the single
source of truth. See `spam_fraud_detector/reports/docs/CORE_README_STYLE.md` for authoring rules.

Author: Guy M. Kaptue T. (guykaptue24@gmail.com). License: MIT.

## Overview

The `core` package implements a **split-first, write-once** benchmarking framework
for binary classification on imbalanced datasets (spam, credit-card fraud). Every
artifact is materialized **once** by `ResultStore` into two synchronized
representations: (1) per-model bundles under `outputs/models/<domain>/all/` and
(2) dashboard tables under `outputs/dashboard/`.

Before any model training, the canonical dataset is split **once** into
`train / val / test`. The `test` split is touched exactly once per final
evaluated model. Any computation reading test without producing a bundle raises
`TestLeakError`.

`make data` runs a six-stage preparation ladder (cleaning, modeling,
analysis, KPI analysis, quality gate, feature engineering)
## How to read this document

Each module section follows the same four-part structure: **What it does**,
**How it does it**, **What it produces**, **How to use it**. Snippets run
on a fresh clone after `make data`.

## `exceptions`

### What it does

The `exceptions` module defines every typed exception raised across `core/`.
Each exception maps to a distinct failure class so callers can catch specific
problems without parsing error messages. The nine exceptions cover data schema
mismatches (`DataSchemaError`), artifact I/O failures (`ArtifactError`),
statistical errors (`StatsError`), contract violations (`ContractViolation`),
missing datasets (`DatasetNotFoundError`), unauthorized test-split access
(`TestLeakError`), data quality gate failures (`DataQualityError`), and split
contract violations (`SplitContractError`), and explainer failures
(`ExplainerError`).

### How it does it

Each exception subclasses `Exception` directly. No inheritance hierarchy beyond
that --- keeping them flat makes `except` clauses readable and avoids accidental
catch-all via base classes. The module exposes `__all__` listing all nine names.

### What it produces

| Artifact | Path | Schema/Shape | When |
|---|---|---|---|
| Exception types | in-memory | Python exceptions | raised at failure |

### How to use it

```python
from spam_fraud_detector.core.exceptions import TestLeakError, DataQualityError
# raise TestLeakError("unauthorized test split access")
```
## `fake_subjects`

### What it does

The `fake_subjects` module builds the synthetic subject layer for hypothetical
persona case files. `load_scenarios(persona)` reads the user-editable
`spam_fraud_detector/core/hypothetical_scenarios.json` and returns the scenario
list for a persona (fraud, spam, anomaly). `subject_generator(scenario_id, seed,
persona)` builds a deterministic `SyntheticSubject` whose name, city, and
contextual fields come from the scenario hint, while numeric fields
(`account_last4`, `customer_tenure_months`, `amount_zscore`, ...) are derived
from a SHA-256 hash of the scenario id so two runs produce identical subjects.

### How it does it

`SyntheticSubject` is a small data holder with `to_dict()` emitting the subject
block embedded in case files. `subject_generator` locates the matching scenario
by `scenario_id`, copies `subject_hint` fields, then derives remaining fields
from a seeded `random.Random` keyed by `scenario_id:seed`, keeping every
fabricated value stable across runs and domains.

### What it produces

| Artifact | Path | Schema/Shape | When |
|---|---|---|---|
| Subject block | in-memory dict | subject_id, full_name, city, persona fields | on subject_generator |
| Scenario list | `core/hypothetical_scenarios.json` | JSON array of scenario dicts | on load_scenarios |

### How to use it

```python
from spam_fraud_detector.core.fake_subjects import load_scenarios, subject_generator
scenarios = load_scenarios("fraud")
subject = subject_generator(scenarios[0]["scenario_id"], seed=42, persona="fraud")
print(subject.to_dict())
```
## `case_renderer`

### What it does

The `case_renderer` module converts a persona case file (empirical or
hypothetical) into a reviewer-facing multi-block text summary. Each case
produces a `CASE <id> — <name>` title block, a `SHAP EXPLANATION — Why was
<name> flagged?` block ranking the top features with signed SHAP values and
arrow indicators, and a `RECOMMENDATION — What to do about <name>'s ...?`
block with action, priority, confidence, reason, next steps, estimated cost if
missed, and a reviewer-notes template. It dispatches on `persona` so fraud,
spam, and anomaly cases use the right nouns, score line, and context line.

### How it does it

`render_case_summary(casefile, width=78)` is pure Python — no dependency on
`rich`, `tabulate`, or any external library — using `textwrap` and Unicode
box-drawing characters (`═`, `─`). It builds three sections:

1. **Header** — `CASE <subject_id> — <full_name>`, model/group, score line
   (`score  · threshold <t> · <error_type>` for fraud/spam,
   `anomaly  (raw <r>) · supervised  · <disagreement>` for anomaly), and
   person-specific context (transaction context for fraud/anomaly; sender +
   subject line for spam).
2. **Explanation block** — top-5 features by `|SHAP|`, ranked with arrows
   (`↑↑↑` dominant, `↑↑` strong, `↑` moderate, `·` weak, `↓` for negative),
   each wrapped with its explanation sentence.
3. **Recommendation block** — `ACTION: <action>` right-aligned with
   `PRIORITY: <priority>`, `CONFIDENCE`, `Why:`, numbered `Next steps:`,
   `Estimated cost if missed:`, and `Reviewer notes:` (wrapped template).

Arrow strength: `↑↑↑` for `|shap| ≥ 0.25` (dominant), `↑↑` for `≥ 0.10`
(strong), `↑` for `≥ 0.02` (moderate), `·` below 0.02 (weak); negative values
flip to `↓`.

### What it produces

| Artifact | Path | Schema/Shape | When |
|---|---|---|---|
| Case summary text | stdout / console | multi-block text, width ≤ 78 | on render_case_summary |

### How to use it

```python
from spam_fraud_detector.core.case_renderer import render_case_summary
summary = render_case_summary(casefile, width=78)
print(summary)
```
## `utils`

### What it does

The `utils` module provides shared helpers used by all other `core/` modules:
dataset configuration via `get_dataset_config`, directory creation via `ensure_dir`,
metric formatting via `format_metrics`, class distribution summaries via
`summarize_class_distribution`, model naming via `safe_model_name`, DataFrame
persistence via `save_dataframe`, seeding via `set_seed`, git commit retrieval via
`get_git_commit`, package version detection via `get_package_versions`, file
hashing via `hash_file`, atomic file writes via `atomic_write_json` and
`atomic_write_text`, and deterministic parquet writes via
`deterministic_parquet` and `deterministic_parquet_bytes`.

### How it does it

Functions are stateless and side-effect-free except where explicitly needed
(`ensure_dir`, `save_dataframe`, `atomic_write_json`). Seeding uses
`numpy.random.default_rng(seed)` for reproducible randomness. File hashing
uses SHA-256 over chunked reads. Atomic writes use temp file + `os.replace`.

### What it produces

| Artifact | Path | Schema/Shape | When |
|---|---|---|---|
| Atomic JSON files | configurable | sorted keys, indent 2 | on write |
| Atomic text files | configurable | UTF-8 | on write |

### How to use it

```python
from spam_fraud_detector.core.utils import set_seed, hash_file
g = set_seed(42)
h = hash_file("data/processed/spam/06_features/train.parquet")
```
## `data_loader`

### What it does

The `data_loader` module provides `KaggleDataLoader`, which downloads, caches,
and prepares Kaggle datasets under `data/`. It supports the canonical raw →
processed layout and the v3.2 layered processed tree. It detects existing
files, downloads only when missing and credentials are configured, and computes
a stable `dataset_hash` over canonical processed files.

### How it does it

The loader checks `data/raw/` and `data/processed/` for existing files. If raw
files are missing and `download=True` with credentials, it downloads via
`kagglehub` or the `kaggle` CLI. Otherwise it raises `DatasetNotFoundError`.
Processed artifacts are built from raw via `_parse_raw()` which extracts
archives, reads CSVs, cleans, and writes parquet/CSV. The `build_processed()`
method is cache-aware: it skips parsing if processed data files already exist.

### What it produces

| Artifact | Path | Schema/Shape | When |
|---|---|---|---|
| Raw extracts | `data/raw/<domain>/` | untouched Kaggle files | on download |
| Cleaned data | `data/processed/<domain>/01_cleaned/` | parquet, CSV | on build |
| Processed data | `data/processed/<domain>/*.parquet` | typed DataFrame | on build |

### How to use it

```python
from spam_fraud_detector.core.data_loader import KaggleDataLoader
loader = KaggleDataLoader(task="spam", data_root="data")
df = loader.load_dataset()
print(f"Loaded {len(df)} rows, hash={loader.dataset_hash}")
```
## `split`

### What it does

The `split` module provides `SplitManager`, which performs a single stratified
split of the canonical dataset into `train / val / test` and persists the split
manifest as `splits/split.json`. It also provides a test-access guard that
raises `TestLeakError` on unauthorized access to the test split.

### How it does it

`SplitManager.split(df)` stratifies by label, shuffles with a seeded RNG, and
writes per-split parquet files under `06_features/`. The manifest records seed,
ratios, row counts, row-index hashes, label distributions, and a timestamp.
`load_split(split)` checks a module-level authorization flag (`_AUTH.allowed`);
test access requires `authorize_test_access()` context manager and is logged to
`outputs/logs/test_access.log`.

### What it produces

| Artifact | Path | Schema/Shape | When |
|---|---|---|---|
| Split manifest | `data/processed/<domain>/splits/split.json` | JSON with seed, ratios, hashes | on split |
| Train split | `data/processed/<domain>/06_features/train.parquet` | subset of input DataFrame | on split |
| Val split | `data/processed/<domain>/06_features/val.parquet` | subset of input DataFrame | on split |
| Test split | `data/processed/<domain>/06_features/test.parquet` | subset of input DataFrame | on split |
| Access log | `outputs/logs/test_access.log` | JSON lines | on test access |

### How to use it

```python
from spam_fraud_detector.core.split import SplitManager, authorize_test_access
sm = SplitManager(domain="spam", seed=42)
manifest = sm.split(df)
with authorize_test_access(run_id="r1", caller="write_model", authorized_by="rs"):
    test_df = sm.load_split("test")
```
## `quality`

### What it does

The `quality` module provides `QualityGate`, which evaluates data quality by
computing null percentage, duplicate rate, row and column counts, and comparing
against configurable thresholds. It produces a blocking gate result written to
`05_quality/gate.json`. A failed gate raises `DataQualityError` and aborts all
downstream pipeline steps so no partial artifacts are written.

### How it does it

`QualityGate.evaluate(df)` computes null percentage as the mean of per-column
null rates, duplicate percentage as `df.duplicated().mean()`, and row/column
counts. It compares against configurable thresholds (`max_null_pct`,
`max_dupe_pct`). Results are written to `05_quality/quality_report.json` and
`05_quality/gate.json` with `pass: bool` and `blocking: true`.

### What it produces

| Artifact | Path | Schema/Shape | When |
|---|---|---|---|
| Quality report | `data/processed/<domain>/05_quality/quality_report.json` | JSON with null_pct, dupe_pct, n_rows, n_cols | on evaluate |
| Gate result | `data/processed/<domain>/05_quality/gate.json` | JSON with pass, blocking | on evaluate |

### How to use it

```python
from spam_fraud_detector.core.quality import QualityGate
qg = QualityGate(domain="spam")
qg.evaluate(df)
```
## `recommendations`

### What it does

The `recommendations` module converts a persona case file (empirical or
hypothetical) into an actionable `recommendation` block. For each persona it
maps the case's score, threshold, error type, and disagreement type to one of
five review actions (`block`, `escalate`, `review`, `monitor`, `clear`) with a
priority, a human-readable reason, suggested next steps, an estimated cost if
missed, a confidence level, and a reviewer notes template. Empirical cases use
the model's error type; hypothetical cases always recommend `review` so the
scenario is validated against real data.

### How it does it

`recommend(persona, casefile)` dispatches to `recommend_fraud`,
`recommend_spam`, or `recommend_anomaly`. Fraud recommendations key off
`error_type × score/threshold` (true positive + high ratio → escalate; false
negative → escalate; false positive → review/low). Spam keys off
`score/threshold` (≥1.8× → block). Anomaly keys off `disagreement_type`
(novel → escalate; sophisticated → review/high; agree_high → escalate;
agree_low → clear). Hypothetical cases bypass error types and recommend review.

### What it produces

| Artifact | Path | Schema/Shape | When |
|---|---|---|---|
| Recommendation block | in dict | action, priority, reason, next steps, cost, confidence, template | on recommend |

### How to use it

```python
from spam_fraud_detector.core.recommendations import recommend
casefile = {"persona": "fraud", "score": 0.983, "threshold": 0.34,
            "error_type": "true_positive", "subject": {"full_name": "Priya",
            "amount_eur": 1.03}, "top_features": []}
rec = recommend("fraud", casefile)
print(rec["action"], rec["priority"], rec["reason"])
```
## `features`

### What it does

The `features` module provides `FeatureEngineer`, which fits encoders and scalers
on the `train` split only and transforms `val` and `test` without re-fitting.
The fitted encoder is persisted as `06_features/feature_encoder.joblib` and
referenced by every bundle's `meta.json`. This ensures no target leakage: the
encoder never sees `val` or `test` labels during fitting.

### How it does it

`fit_transform(train_df, val_df)` fits a `StandardScaler` on numeric columns of
the training data and persists it via `joblib.dump`. `transform(df)` loads the
persisted scaler and applies it to new data. No target leakage: the encoder
never sees `val` or `test` labels.

### What it produces

| Artifact | Path | Schema/Shape | When |
|---|---|---|---|
| Feature encoder | `data/processed/<domain>/06_features/feature_encoder.joblib` | scikit-learn transformer | on fit_transform |

### How to use it

```python
from spam_fraud_detector.core.features import FeatureEngineer
fe = FeatureEngineer(domain="spam")
fe.fit_transform(train_df)
fe.transform(test_df)
```
## `data_pipeline`

### What it does

The `data_pipeline` module provides `DataPipeline`, which orchestrates the
six-stage data preparation ladder (L1-L6): cleaning, modeling, analysis, KPI,
quality, and feature engineering. It loads raw data, creates the layered
directory structure, runs each stage in order, persists artifacts, and returns a
`PipelineResult` with per-layer hashes and quality gate status.

### How it does it

`DataPipeline.run()` loads the raw dataset, creates the layered directory
structure, runs each stage in order, and persists artifacts. The quality gate
is blocking: if it fails, `DataQualityError` is raised and no downstream
artifacts are written. Each layer is idempotent via hash sidecars.

### What it produces

| Artifact | Path | Schema/Shape | When |
|---|---|---|---|
| Cleaned data | `data/processed/<domain>/01_cleaned/data.parquet` | subset of input | on run |
| Quality report | `data/processed/<domain>/05_quality/quality_report.json` | JSON | on run |
| Gate result | `data/processed/<domain>/05_quality/gate.json` | JSON with pass, blocking | on run |
| Split manifest | `data/processed/<domain>/splits/split.json` | JSON | on run |

### How to use it

```python
from spam_fraud_detector.core.data_pipeline import DataPipeline
dp = DataPipeline(domain="spam", data_root="data", seed=42)
result = dp.run()
```

## `comparer`

### What it does

The `comparer` module provides `ClassifierGroupComparer`, which trains and
evaluates multiple classifiers in both scaled and unscaled regimes for binary
classification tasks. It handles data loading, preparation, model training,
evaluation, and optional persistence via ResultStore. Models that cannot run
(e.g., MultinomialNB on negative inputs) are skipped and recorded in the
manifest with a reason.

### How it does it

The comparer splits data into train/test, trains each model in the appropriate
group (scaled models get a `StandardScaler` pipeline), evaluates on test, and
collects metrics. If a ResultStore is provided, it persists bundles via
`write_model()` and records skipped models via `write_skipped()`.

### What it produces

| Artifact | Path | Schema/Shape | When |
|---|---|---|---|
| Model bundle | `outputs/models/<domain>/all/<Model>__<group>.joblib` | serialized pipeline | on compare_groups |
| Metrics | in-memory DataFrame | accuracy, precision, recall, f1, roc_auc, pr_auc | on compare_groups |

### How to use it

```python
from spam_fraud_detector.core.comparer import ClassifierGroupComparer
comparer = ClassifierGroupComparer(task="spam", df=df, store=store)
comparer.load_data()
comparer.prepare_data()
results = comparer.compare_groups()
```
## `result_store`

### What it does

The `result_store` module provides `ResultStore`, the single writer for all
pipeline artifacts. It materializes per-model bundles (`.joblib`, `.meta.json`,
`.eval.json`, `.curve.json`, `.confusion.json`, `.parquet` predictions) and
dashboard tables (`fact_*.parquet`, `dim_*.parquet`, `.csv`, `index.json`,
`filters.json`) in one atomic transaction via a staging directory + `os.replace`.

### How it does it

`ResultStore.write_model()` writes all five bundle files and buffers dashboard
rows in memory. `finalize()` writes dashboard tables, validates invariants,
then atomically replaces `outputs/` with the staging directory. On failure, the
staging directory is cleaned up and `outputs/` is left untouched.

### What it produces

| Artifact | Path | Schema/Shape | When |
|---|---|---|---|
| Model bundle | `outputs/models/<domain>/all/<Model>__<group>.joblib` | serialized pipeline | on write_model |
| Meta JSON | `outputs/models/<domain>/all/<Model>__<group>.meta.json` | provenance + hyperparams | on write_model |
| Eval JSON | `outputs/models/<domain>/all/<Model>__<group>.eval.json` | validation + test metrics | on write_model |
| Curve JSON | `outputs/models/<domain>/all/<Model>__<group>.curve.json` | ROC, PR, calibration | on write_model |
| Confusion JSON | `outputs/models/<domain>/all/<Model>__<group>.confusion.json` | tp/fp/tn/fn | on write_model |
| Predictions | `outputs/predictions/<domain>/all/<Model>__<group>.parquet` | row_id, y_true, y_score, y_pred | on write_model |
| Dashboard tables | `outputs/dashboard/fact_*.parquet` | typed fact/dim tables | on finalize |
| CONTRACT | `outputs/CONTRACT.json` | machine-readable spec | on finalize |

### How to use it

```python
from spam_fraud_detector.core.result_store import ResultStore, RunContext
ctx = RunContext(domain="spam_detection", seed=42)
rs = ResultStore(out_root, ctx)
rs.write_model(name, group, pipeline, eval_val=eval_metrics, eval_test=eval_metrics)
rs.finalize()
```

## `selector`

### What it does

The `selector` module provides `TopKSelector`, which selects the top-K models
from persisted outputs without re-fitting. It reads from the finalized dashboard
tables, filters by domain and split, aggregates mean metric per (model, group),
ranks descending, and copies the top-K bundles to `outputs/models/<domain>/top3/`.
It records the selection via `store.write_selection()` for provenance.

### How it does it

`TopKSelector.select()` reads `manifest.parquet` and `fact_metrics.parquet`,
filters by domain and split, aggregates mean metric per (model, group), ranks
descending, and copies the top-K joblib + meta files. It records the selection
via `store.write_selection()`.

### What it produces

| Artifact | Path | Schema/Shape | When |
|---|---|---|---|
| Top-K bundles | `outputs/models/<domain>/top3/rank_<k>_<Model>.joblib` | copy of bundle | on select |
| Selection JSON | `outputs/models/<domain>/top3/selection.json` | ranking with metrics | on select |
| Selection table | `outputs/dashboard/selection.parquet` | rank, model, group, metrics | on finalize |

### How to use it

```python
from spam_fraud_detector.core.selector import TopKSelector
sel = TopKSelector(outputs, domain="spam_detection", k=3)
ranking = sel.select(store=store)
```
## `stats`

### What it does

The `stats` module provides `StatsEngine`, which performs pairwise statistical
testing (McNemar) between top models on the test split. It applies Holm
correction to p-values and produces a summary. It operates on `val`-derived
metrics for search and reads persisted test metrics from already-written bundles.

### How it does it

`StatsEngine.compute()` loads top models from `selection.json`, loads their
persisted predictions, aligns by `row_id` on the test split, computes
discordant pairs, and runs McNemar. Holm correction is applied to all
pairwise p-values. Results are persisted via `store.write_stats()`.

Effect size helpers `cohen_h()` and `risk_difference()` are exported alongside
`StatsEngine`. `cohen_h(p1, p2)` returns Cohen's h for two proportions
(2·arcsin(√p1) − 2·arcsin(√p2), |h| ≤ π ≈ 3.14; |h| = 0.2 small, 0.5 medium,
0.8 large). `risk_difference(p1, p2)` returns p1 − p2 (−1 ≤ RD ≤ 1).

### What it produces

| Artifact | Path | Schema/Shape | When |
|---|---|---|---|
| A/B test JSON | `outputs/stats/<domain>/ab_test.json` | comparisons with p-values | on compute |
| Summary MD | `outputs/stats/<domain>/summary.md` | markdown summary | on compute |
| Fact stats | `outputs/dashboard/fact_stats.parquet` | model pairs, statistics | on finalize |

### How to use it

```python
from spam_fraud_detector.core.stats import StatsEngine, cohen_h, risk_difference
se = StatsEngine(outputs, domain="spam_detection")
se.compute(store=store)
h = cohen_h(0.8, 0.5)   # Cohen's h for two proportions
rd = risk_difference(0.8, 0.5)  # Absolute risk difference
```

## `visualizer`

### What it does

The `visualizer` module provides `ModelVisualizer`, which generates publication-
ready figures including F1 score bars, ROC and PR curves, confusion matrices,
and group comparison charts. It also exports metrics to Excel workbooks and
plots to PDF documents for sharing and archiving.

### How it does it

`ModelVisualizer` takes a `results` dict from `compare_groups()` and renders
matplotlib/seaborn figures. Figures are saved to `reports/<name>/figures/`.
Excel export uses `pd.ExcelWriter`. PDF export uses `PdfPages`.

### What it produces

| Artifact | Path | Schema/Shape | When |
|---|---|---|---|
| F1 score plot | `reports/<name>/figures/<name>_f1_scores.png` | PNG | on save_f1_scores |
| ROC curves | `reports/<name>/figures/<name>_roc_curves.png` | PNG | on save_roc_curves |
| Confusion matrices | `reports/<name>/figures/<name>_confusion_matrices.png` | PNG | on save_confusion_matrices |
| Excel metrics | `reports/<name>/docs/<name>_model_metrics.xlsx` | Excel workbook | on export_metrics_to_excel |
| PDF plots | `reports/<name>/docs/<name>_model_plots.pdf` | PDF | on export_plots_to_pdf |

### How to use it

```python
from spam_fraud_detector.core.visualizer import ModelVisualizer
viz = ModelVisualizer(results, report_name="spam_detection")
viz.save_f1_scores()
viz.save_roc_curves()
viz.export_metrics_to_excel()
viz.export_plots_to_pdf()
```
## `anomaly`

### What it does

The `anomaly` module provides `AnomalyComparer`, which trains unsupervised
detectors (`IsolationForest`, `LocalOutlierFactor`, `OneClassSVM`) on the
legitimate (negative) class only and scores all samples continuously. This
provides an unsupervised baseline for comparison with supervised models.
Anomaly scores are normalized to `[0, 1]` via monotone min-max so they are
comparable to `predict_proba` outputs. The module raises `DataSchemaError` if
any positive sample leaks into the training data.

### How it does it

`AnomalyComparer.fit()` trains each detector on `X_train` filtered to
`y_train == 0`. It computes `decision_function` on train, val, and test, then
normalizes scores to `[0, 1]` using train statistics. `compare()` computes
ROC-AUC and PR-AUC on val and test. All detectors produce continuous scores,
so they flow through `ResultStore.write_model(group="anomaly")` like any
supervised model.

### What it produces

| Artifact | Path | Schema/Shape | When |
|---|---|---|---|
| Anomaly scores | in-memory arrays | float64, shape (n_samples,) | on fit/compare |
| Metrics | in-memory dict | roc_auc, pr_auc per detector | on compare |

### How to use it

```python
from spam_fraud_detector.core.anomaly import AnomalyComparer
ac = AnomalyComparer(X_train, y_train, X_val, y_val, X_test, y_test)
results = ac.compare()
for name, r in results.items():
    print(name, r["metrics_test"]["pr_auc"])
```

## `explainer`

### What it does

The `explainer` module provides `ModelExplainer`, which produces SHAP
explanations (global summary + per-instance waterfalls) for fitted models.
It selects `TreeExplainer` for tree models and `KernelExplainer` for others.
Waterfalls are computed on the `val` split only, never on `test`. LIME is
available behind an opt-in flag but is documented as non-compliant for
audit claims. The module raises `ExplainerError` if SHAP is missing and the
backend is `"shap"`.

v3.9.1 adds `TextExplainer` for TF-IDF spam models (mapping SHAP values to
tokens) and `AnomalyExplainer` for anomaly detectors (using
`PermutationExplainer` on `decision_function`). The `make_explainer` factory
routes by persona: `"fraud"`, `"spam"`, or `"anomaly"`.

v4.4.8 adds `explain_feature(name, value, shap_value, persona)` for
persona-specific plain-language explanations (statistical deviation for fraud,
token frequency for spam, distribution-tail for anomaly) and
`build_casefile` / `build_hypothetical_casefile` for case-file assembly.
`render_case_waterfall` and `render_hypothetical_case_waterfall` render SHAP
waterfall PNGs for empirical and hypothetical subjects respectively.

### How it does it

`explain_summary(X_sample, path)` picks the appropriate SHAP explainer,
computes shap values, and writes a beeswarm PNG with `dpi=150`,
`bbox_inches="tight"`. `explain_instances(X_sample, indices, out_dir)` writes
one waterfall PNG per index. `top_k_mistakes(y_true, y_score, threshold, k)`
returns indices of top-k false positives and false negatives.

### What it produces

| Artifact | Path | Schema/Shape | When |
|---|---|---|---|
| SHAP summary | `reports/<domain>/docs/explainability/<model>__<group>__shap_summary.png` | PNG | on explain_summary |
| Waterfall plots | `reports/<domain>/docs/explainability/waterfall_<idx>.png` | PNG | on explain_instances |

### How to use it

```python
from spam_fraud_detector.core.explainer import ModelExplainer, top_k_mistakes
ex = ModelExplainer(model, X_background, feature_names=names, backend="shap")
ex.explain_summary(X_val_sample, "reports/fraud_detection/docs/explainability/summary.png")
fps, fns = top_k_mistakes(y_val, scores, threshold=0.5, k=3)
ex.explain_instances(X_val_sample, fps + fns, "reports/fraud_detection/docs/explainability/")
```

Render functions (all take `result_store: ResultStore` and return `Path | None`):
`render_spam_best_confusion_matrix`, `render_spam_waterfall_example`,
`render_explainability_fraud_case_F_012453`, `render_explainability_spam_case_S_000412`,
`render_explainability_anomaly_case_A_013483`, `render_validation_quality_report`,
`render_validation_pr_auc_by_model`, `render_validation_significance_summary`,
`render_validation_threshold_sensitivity`, `render_paper_pipeline_architecture`,
`render_paper_pr_auc_comparison`, `render_paper_significance_forest_plot`,
`render_paper_threshold_star_comparison`, `render_paper_shap_summary_aggregate`,
`render_all`, `render_readme_hero_results`, `render_readme_results_table`,
`render_readme_personas_triptych`, `render_framework_pipeline_diagram`,
`render_framework_layer_order`, `render_framework_mcnemar_decision_tree`,
`render_framework_threshold_cost_curve`, `render_spam_class_distribution`,
`render_fraud_class_distribution`, `render_spam_threshold_curve`,
`render_fraud_threshold_curve`, `render_spam_roc_curve`, `render_fraud_roc_curve`,
`render_spam_pr_curve`, `render_fraud_pr_curve`, `render_spam_threshold_sensitivity`,
`render_fraud_threshold_sensitivity`, `render_spam_pr_auc_by_model`,
`render_fraud_pr_auc_by_model`, `render_spam_holm_significance`,
`render_fraud_holm_significance`, `render_spam_roc_vs_pr`, `render_fraud_roc_vs_pr`,
`explain_feature_from_case`.
## Cross-module flows

### (a) `make data` → L1…L6 → `splits/split.json`

```
make data
  → KaggleDataLoader.download_if_missing()
  → DataPipeline.run()
    → L1: Datenbereinigung (cleaning)
    → L2: Datenmodellierung (modeling)
    → L3: Datenanalyse (analysis)
    → L4: KPI-Analyse (kpi)
    → L5: Datenqualitaet (quality gate — blocking)
    → L6: Feature Engineering (features + split)
  → SplitManager.split() → splits/split.json
```

### (b) `make pipeline` → comparer → result_store.write_model → finalize

```
make pipeline
  → ClassifierGroupComparer.compare_groups()
    → for each model: evaluate_model() → metrics, confusion, curves
    → store.write_model() → bundles + dashboard rows
    → store.write_skipped() → skipped models
  → TopKSelector.select(store=store) → top-3 bundles
  → StatsEngine.compute(store=store) → A/B tests
  → store.finalize() → atomic outputs/ replacement
```

### (c) `make select-top3` / `make stats` → read-only over finalized outputs

```
make select-top3
  → TopKSelector (reads from finalized outputs/dashboard/)
  → store.write_selection() → selection.json + selection.parquet

make stats
  → StatsEngine (reads persisted predictions from bundles)
  → store.write_stats() → ab_test.json + summary.md
```

## Provenance & invariants (reference)

| Invariant | Enforced by | Test |
|---|---|---|
| Split-once discipline | `SplitManager.split()` | `tests/regression/test_split_determinism.py` |
| Test-touched-once | `ResultStore.write_model()` | `tests/integration/test_test_leak.py` |
| Atomic finalize | `ResultStore.finalize()` | `tests/test_result_store.py` |
| No silent drops | `ResultStore.validate()` | `tests/test_result_store.py` |
| Quality gate blocking | `QualityGate.evaluate()` | `tests/unit/test_quality.py` |
| Feature fit on train only | `FeatureEngineer.fit_transform()` | `tests/unit/test_features.py` |
| Test leak guard | `SplitManager.load_split()` | `tests/integration/test_test_leak.py` |
| Single core/README.md | `make docs-check-core` | `tests/unit/test_check_core_readme.py` |

## Author & license

Author: Guy M. Kaptue T. — https://github.com/GuyKaptue — guykaptue24@gmail.com
License: MIT. Repository: https://github.com/GuyKaptue/spam-fraud-detector
Engineering) under `data/processed/<domain>/`.