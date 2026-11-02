# Scripts

Helper scripts for evidence collection, figure rendering, documentation
verification, and repository audits. Each script accepts `--help` for usage.

---

## Scripts by Category

### Evidence Collection

| Script | Purpose |
|---|---|
| `scripts/collect_evidence.py` | Collects model metrics, curves, confusion matrices, and statistical tests from `outputs/` into `docs/EVIDENCE_INVENTORY.json` — the single source of truth for all documentation claims. |

### Figure Generation

| Script | Purpose |
|---|---|
| `scripts/render_doc_figures.py` | Renders all documentation figures (ROC curves, PR curves, confusion matrices, class distributions, threshold sweeps, Holm significance plots) from `docs/EVIDENCE_INVENTORY.json`. Outputs to `docs/figures/`. |
| `scripts/make_gifs.py` | Generates animated GIF galleries from figure PNGs for visual comparison across models. Outputs to `docs/figures/gifs/`. |

### Documentation Verification

| Script | Purpose |
|---|---|
| `scripts/verify_documentation.py` | Verifies that every figure and output path referenced in docs exists in `docs/EVIDENCE_INVENTORY.json`. Fails on missing or mismatched paths. |
| `scripts/check_docs_paths.py` | Validates that all figure paths in documentation point to existing files under `docs/figures/`. |
| `scripts/check_core_readme.py` | Verifies that `spam_fraud_detector/core/README.md` documents every public symbol in `spam_fraud_detector/core/`. |
| `scripts/check_explainability.py` | Verifies that explainability artifacts match the expected structure for each domain and persona. |

### Statistical Verification

| Script | Purpose |
|---|---|
| `scripts/verify_ab.py` | Verifies McNemar test results and Holm corrections in `outputs/stats/<domain>/ab_test.json`. |
| `scripts/verify_thresholds.py` | Verifies threshold sweep data and expected-cost-optimal thresholds per model. |

### Artifact & Provenance Audits

| Script | Purpose |
|---|---|
| `scripts/verify_contract.py` | Verifies dashboard contract: schema consistency, foreign-key joins, Parquet/CSV parity between fact and dimension tables. |
| `scripts/verify_explainability_provenance.py` | Verifies explainability provenance: that every explainability artifact traces back to a model bundle, split, and evidence inventory entry. |
| `scripts/audit_pipeline_order.py` | Audits that pipeline stages execute in the correct order (L1→L6). |
| `scripts/audit_split_paths.py` | Audits that split paths are consistent across domains and splits. |
| `scripts/audit_split_aware_layers.py` | Audits that each pipeline layer produces split-aware outputs. |
| `scripts/audit_no_local_split.py` | Audits that no local (non-stratified) splits are created. |
| `scripts/audit_wiring.py` | Audits that all pipeline components are correctly wired together. |
| `scripts/audit_explainability_personas.py` | Audits that explainability is available for all three personas. |
| `scripts/audit_explainability_provenance.py` | Audits explainability provenance chains. |
| `scripts/audit_threshold_wiring.py` | Audits that threshold selection is consistent across models. |
| `scripts/audit_cost_matrix.py` | Audits that cost matrices are consistent across models. |

### CI Guard Scripts

| Script | Purpose |
|---|---|
| `scripts/ci_guard_explain_anomaly.py` | CI guard: verifies that anomaly explainability produced zero failures. Runs after `make explain-anomaly` in CI. |

---

## How to Run

```bash
# Collect evidence from outputs/
make evidence
# or: python scripts/collect_evidence.py

# Render all figures
make figures
# or: python scripts/render_doc_figures.py

# Generate animated GIFs
make gifs
# or: python scripts/make_gifs.py

# Verify documentation against inventory
make docs-verify
# or: python scripts/verify_documentation.py

# Run all audits
make audit
# or: python scripts/audit_*.py (individual scripts)

# Verify explainability provenance
python scripts/verify_explainability_provenance.py
```

---

## Adding a New Script

1. Place the script in `scripts/`.
2. Add a `--help` option for usage information.
3. Add the script to the appropriate category table above.
4. Add a Makefile target if the script should be runnable via `make`.
5. Add any required CI checks to `.github/workflows/ci.yml`.

---

**License:** MIT © Guy M. Kaptue T. See [../../LICENSE](../../../../LICENSE).
