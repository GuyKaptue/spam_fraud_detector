# Changelog

## v5.0 — Evidence-first documentation: 23 figures, evidence inventory, figure registry

### Added
- `docs/EVIDENCE_INVENTORY.json`: machine-readable evidence inventory (25 models, 2 domains, 15 dashboard tables)
- `docs/figures/REGISTRY.json`: figure registry with 30 entries mapping names to render functions
- `scripts/collect_evidence.py`: collects artifacts into `docs/EVIDENCE_INVENTORY.json`
- `scripts/render_doc_figures.py`: renders all documentation figures deterministically from the inventory
- `scripts/verify_documentation.py`: verifies docs against the evidence inventory
- `docs/FRAMEWORK.md`: v5.0 evidence-first methodology and architecture
- `docs/PAPER.md`: publication-ready results, significance, threshold analysis
- 23 figure PNGs in `docs/figures/` (hero results, results table, personas triptych, pipeline diagram, layer order, McNemar decision tree, threshold cost curve, class distributions, threshold curves, ROC curves, PR curves, threshold sensitivity, PR-AUC by model, Holm significance, ROC vs PR)
- `ModelExplainer`, `TextExplainer`, `AnomalyExplainer` classes reconstructed in `core/explainer.py`
- `make_explainer()` factory function for persona-aware explainer routing
- `top_k_mistakes()` standalone function for top-k FP/FN identification

### Changed
- `spam_fraud_detector/core/explainer.py`: added 22 missing render function stubs; added `ModelExplainer`/`TextExplainer`/`AnomalyExplainer` classes; added `make_explainer()` factory; added `top_k_mistakes()`; fixed `build_hypothetical_casefile` to use `top_tokens` key for spam persona; fixed `AnomalyExplainer` to accept `detector=` keyword
- `scripts/render_doc_figures.py`: fixed curve data access to use nested `roc`/`pr` keys; fixed fraud Holm to use `ab_test.comparisons`; fixed `readme_personas_triptych` for missing directories
- `Makefile`: `figures` target now runs `scripts/render_doc_figures.py`
- `README.md`: version bumped to 5.0.0; added evidence-first documentation section with figure catalog; updated documentation index and repository structure


### Added
- `spam_fraud_detector/core/stats.py`: `cohen_h()` and `risk_difference()`
  effect-size functions; `StatsEngine._check_leakage()` raises
  `SplitContractError` when train/val/test row_ids overlap between compared
  models; `StatsEngine.compute()` now reports `cohen_h`, `risk_difference`,
  `acc_a`, `acc_b` per comparison; full v2.0 schema with `config` block,
  `method` field per comparison, `n01`/`n10`, `family_rank`,
  `significant_holm_0.05`/`0.01`, `winner_significant`; automatic McNemar
  fallback to exact test when `n_discordant < min_discordant_chi2`; zero-
  discordant skip with warning; deterministic Holm tie-break
  `(p_value, model_a, model_b)`.
- `scripts/verify_ab.py`: verifies `ab_test.json` per domain — Holm adjustment
  monotonicity (`p_holm_adjusted >= p_value`, `<= 1.0`), `|cohen_h| <= 2.0`,
  `-1 <= risk_difference <= 1`, `n_discordant <= n`, winner consistency with
  `acc_a`/`acc_b`.
- `scripts/audit_ab_wiring.py`: static AST audit that `split == "test"` filter
  exists, exact McNemar fallback branch exists, deterministic Holm tie-break
  sort key `(p_value, model_a, model_b)` is used, zero-discordant skip with
  warning exists.
- `scripts/inject_stats.py`: injects A/B summary into `SPAM_DETECTION.md` and
  `FRAUD_DETECTION.md` between `<!-- ab-summary:start -->` and
  `<!-- ab-summary:end -->` markers.
- `scripts/export_mlflow.py`: logs `ab_test.json` to MLflow.
- `tests/unit/test_stats_ab.py`: tests for `cohen_h`, `risk_difference`,
  `StatsEngine._check_leakage()` (leakage detection and clean splits).

### Changed
- `spam_fraud_detector/core/stats.py`: `compute()` now includes effect size
  (`cohen_h`, `risk_difference`) and leakage guard in every comparison;
  full v2.0 schema with `config` block, `method` field per comparison,
  `n01`/`n10`, `family_rank`, `significant_holm_0.05/0.01`,
  `winner_significant`; automatic McNemar exact fallback below 25 discordant
  pairs; zero-discordant skip.
- `spam_fraud_detector/core/result_store.py`: `write_model()` now accepts
  `feature_columns` parameter and records it in `meta.json`;
  `write_model()` now writes `feature_columns` to `meta.json`;
  `ResultStore._write_dashboard_tables()` now populates `fact_stats` from
  `store._stats`, adds `fact_stats_summary` and `dim_method` tables, adds
  `L10` (method) filter level to `filters.json`.
- `spam_fraud_detector/main.py`: added `--stats-only` flag and
  `SpamFraudDetector._run_stats_only()` method; `run()` dispatcher handles
  `stats_only`.
- `Makefile`: added `verify-ab` and `docs-stats-check` targets; added both
  to `.PHONY`; `stats-spam`/`stats-fraud` now call `main.py --stats-only`;
  `verify-casefiles` added to `verify-explainability`.
- `spam_fraud_detector/core/__init__.py`: exports `cohen_h` and
  `risk_difference`.
- `outputs/CONTRACT.json`: added `stats_contract` block.
- `spam_fraud_detector/core/README.md`: documents `cohen_h()` and
  `risk_difference()` helper functions.

## v4.4.6 — Recommendations and hypothetical case files

### Added
- `spam_fraud_detector/core/hypothetical_scenarios.json`: user-editable
  hypothetical scenarios (3 fraud, 3 spam, 2 anomaly) read by
  `core.fake_subjects.load_scenarios()`.
- `spam_fraud_detector/core/fake_subjects.py`: `load_scenarios()`,
  `SyntheticSubject`, and `subject_generator()` for deterministic synthetic
  subjects keyed by scenario id + seed.
- `spam_fraud_detector/core/recommendations.py`: persona-specific
  recommendation engine (`recommend_fraud`, `recommend_spam`,
  `recommend_anomaly`, `recommend`) producing `action` × `priority` ×
  `reason` × `suggested_next_steps` × `estimated_cost_if_missed` ×
  `confidence` × `reviewer_notes_template`.
- `spam_fraud_detector/core/explainer.py`: `build_casefile()`,
  `build_hypothetical_casefile()`, `render_case_waterfall()`,
  `render_hypothetical_case_waterfall()`, plus `_shap_row()` /
  `_top_features_from_shap()` helpers.
- Tests: `tests/unit/test_recommendations.py`,
  `tests/unit/test_hypothetical_casefile.py`,
  `tests/integration/test_persona_casefiles.py`.

### Changed
- `spam_fraud_detector/main.py _run_explain_only()` now produces 3 empirical
  + N hypothetical case files per model (JSON + PNG + recommendation block),
  writing paths to `meta.json.explainability.casefiles`.
- `spam_fraud_detector/core/result_store.py record_explainability_direct()`
  accepts `casefiles` and stores them in `meta.json.explainability.casefiles`.
- `scripts/verify_explainability_provenance.py` validates casefile keys,
  `case_type` ∈ {empirical, hypothetical}, PNG existence/size, and the full
  recommendation schema.
- `outputs/CONTRACT.json` gains `explainability.casefiles` block.
- `spam_fraud_detector/core/README.md` gains `fake_subjects` and
  `recommendations` module sections; `explainer` section updated.
- `scripts/check_core_readme.py` `MODULE_ORDER` updated to 16 modules.

## v3.6 — Split-first at ingestion, narrative core/README.md

### Added
- `docs/CORE_README_STYLE.md`: canonical style guide for `core/README.md`.
- `scripts/check_core_readme.py`: narrative-contract checker (structure,
  subheadings, word floor, produces-table, runnable snippet, public-name
  coverage).
- `tests/unit/test_check_core_readme.py`.
- `docs/MAKEFILE.md`, `docs/BADGES.md`.

### Changed
- `spam_fraud_detector/core/README.md` rewritten as a single hand-authored
  narrative covering all twelve modules, three cross-module sequence diagrams,
  and a provenance/invariants reference table.
- `make docs-check-core` now runs `scripts/check_core_readme.py`; the old
  consolidation/hash/owner targets are removed.
- `outputs/CONTRACT.json.docs_policy` simplified (no anchors, no hashes);
  `style_guide` added.
- `.pre-commit-config.yaml` and `ci.yml` reference the new checker; the
  `core-readme-locked` hook and `docs-core-locked` job are removed.
- Root `README.md` updated with Documentation layout subsection.

### Planned (v3.6 pipeline reorder — tracked, not shipped here)
- Move the split to `01_split/` and run it immediately after `data_loader`,
  before cleaning, EDA, KPI, or feature engineering, to remove every remaining
  class of pre-split leakage. The `SplitManager`, `QualityGate`, and
  `FeatureEngineer` APIs stay unchanged; only `DataPipeline.run()` order and
  the layer directory names change. Migration notes are embedded in
  `core/README.md` next to the affected sections.