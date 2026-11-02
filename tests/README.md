# spam-fraud-detector — Test Suite

[![CI](https://github.com/GuyKaptue/spam-fraud-detector/actions/workflows/ci.yml/badge.svg)](https://github.com/GuyKaptue/spam-fraud-detector/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/GuyKaptue/spam-fraud-detector/branch/main/graph/badge.svg)](https://codecov.io/gh/GuyKaptue/spam-fraud-detector)

Industry-grade test suite for the spam & fraud detection experimentation framework.
Covers unit, integration, and regression testing with a coverage target of **≥90%**.

**Author:** Guy M. Kaptue T. — [GitHub](https://github.com/GuyKaptue) — guykaptue24@gmail.com  
**License:** MIT  
**Test Runner:** pytest  
**Coverage Target:** ≥90%

---

## 1. Purpose

This directory contains the automated test suite that validates the correctness,
reproducibility, and statistical integrity of the spam-fraud-detector framework.
Every test is designed to catch regressions early and to guarantee that
documentation claims remain grounded in actual experimental outputs (per the
v5.0 evidence-first documentation rule).

Tests are grouped into three categories by scope and execution cost:

| Category | Location | Purpose | Speed |
|---|---|---|---|
| **Unit** | `tests/unit/` | Verify individual functions and classes in isolation | Fast |
| **Integration** | `tests/integration/` | Verify cross-module behavior and end-to-end persona workflows | Medium |
| **Regression** | `tests/regression/` | Guard against previously fixed bugs and data drift | Medium |

---

## 2. Repository Structure

```
tests/
├── README.md                 # This file
├── conftest.py               # Shared pytest fixtures
├── test_result_store.py      # ResultStore end-to-end tests
├── unit/                     # Unit tests (isolated components)
│   ├── test_ab_leakage.py
│   ├── test_ab_reproducibility.py
│   ├── test_ab_statistics.py
│   ├── test_anomaly_explainer.py
│   ├── test_case_renderer.py
│   ├── test_check_core_readme.py
│   ├── test_check_docs_paths.py
│   ├── test_deterministic_parquet.py
│   ├── test_explainer_factory.py
│   ├── test_hypothetical_casefile.py
│   ├── test_recommendations.py
│   ├── test_result_store_explainability_persona.py
│   ├── test_threshold_sweep_cost.py
│   ├── test_text_explainer.py
│   ├── test_utils.py
│   └── test_write_model_threshold_policy.py
├── integration/              # Integration tests (cross-module)
│   ├── test_persona_casefiles.py
│   ├── test_selector_expected_cost.py
│   ├── test_thresholds_bundle.py
│   └── test_three_personas.py
├── regression/               # Regression tests (bug & drift guards)
│   ├── test_anomaly_failure_logged.py
│   ├── test_anomaly_layer_runs.py
│   ├── test_explain_console_output.py
│   ├── test_explain_is_read_only.py
│   ├── test_finalize_merges_domains.py
│   ├── test_reproduce_thresholds.py
│   ├── test_selector_uses_current_staging.py
│   ├── test_threshold_backward_compat.py
│   ├── test_selector_uses_current_staging.py
│   ├── test_threshold_layer_runs.py
│   └── test_anomaly_layer_runs.py
└── __pycache__/              # Python bytecode cache (gitignored)
```

---

## 3. Running Tests

### Prerequisites

```bash
# Install the package and dev dependencies
pip install -e ".[dev]"

# Verify pytest is available
pytest --version
```

### Run All Tests

```bash
# Full test suite (unit + integration + regression)
make test

# Or directly with pytest
pytest -q tests/
```

### Run by Category

```bash
# Unit tests only (fastest, no external dependencies)
make test-unit
# or: pytest -q tests/unit/

# Integration tests only (requires outputs/ artifacts)
make test-integration
# or: pytest -q tests/integration/

# Regression tests only (guards against known bugs)
pytest -q tests/regression/
```

### Run with Coverage

```bash
make test-cov
# or: pytest --cov=spam_fraud_detector.core --cov-report=term-missing
```

Coverage must stay at **≥90%** for `spam_fraud_detector.core`. If coverage drops
below this threshold, CI fails.

### Run Specific Test Files

```bash
pytest -q tests/unit/test_ab_statistics.py tests/integration/test_three_personas.py
```

### Run by Test Name Pattern

```bash
pytest -q -k "holm" tests/
pytest -q -k "anomaly" tests/unit/test_anomaly_explainer.py
```

---

## 4. Test Categories

### 4.1 Unit Tests (`tests/unit/`)

Unit tests validate individual functions and classes in isolation. They do not
require trained model bundles, MLflow runs, or external data. These tests run in
seconds and are the first line of defense against regressions.

| Test File | What It Validates |
|---|---|
| `test_ab_statistics.py` | Cohen's h bounds, risk difference, Holm adjustment, winner consistency |
| `test_ab_reproducibility.py` | Determinism of statistical comparisons across runs |
| `test_ab_leakage.py` | Train/test leakage guard in `StatsEngine` |
| `test_anomaly_explainer.py` | `AnomalyExplainer` behavior with anomaly detectors |
| `test_case_renderer.py` | Case file rendering output format and content |
| `test_check_core_readme.py` | Core README documents all public symbols |
| `test_check_docs_paths.py` | Docs path references resolve to canonical locations |
| `test_deterministic_parquet.py` | Parquet writes are byte-identical across runs |
| `test_explainer_factory.py` | `make_explainer()` factory routing by persona |
| `test_hypothetical_casefile.py` | Hypothetical case file generation and structure |
| `test_recommendations.py` | Recommendation logic per persona |
| `test_result_store_explainability_persona.py` | Explainability recording per persona |
| `test_text_explainer.py` | `TextExplainer` token-level SHAP mapping |
| `test_utils.py` | Utility functions (safe_model_name, etc.) |
| `test_write_model_threshold_policy.py` | Threshold policy enforcement in model writes |

### 4.2 Integration Tests (`tests/integration/`)

Integration tests verify cross-module behavior and end-to-end persona workflows.
They may require trained model bundles in `outputs/` or `reports/`.

| Test File | What It Validates |
|---|---|
| `test_persona_casefiles.py` | ResultStore integration with persona case files |
| `test_selector_expected_cost.py` | Expected-cost model selection logic |
| `test_thresholds_bundle.py` | Threshold bundle persistence and retrieval |
| `test_three_personas.py` | All three personas (spam, fraud, anomaly) produce artifacts |

### 4.3 Regression Tests (`tests/regression/`)

Regression tests guard against previously fixed bugs and data drift. Each test
is named after the issue it prevents from recurring.

| Test File | Guards Against |
|---|---|
| `test_anomaly_failure_logged.py` | Anomaly failures being silently dropped |
| `test_anomaly_layer_runs.py` | Anomaly layer execution in pipeline |
| `test_explain_console_output.py` | Explain-only mode console output format |
| `test_explain_is_read_only.py` | Explain-only mode not mutating model bundles |
| `test_finalize_merges_domains.py` | Store finalize merges spam and fraud domains |
| `test_reproduce_thresholds.py` | Threshold reproducibility across runs |
| `test_selector_uses_current_staging.py` | Model selection reads current staging area |
| `test_threshold_backward_compat.py` | Threshold format backward compatibility |
| `test_threshold_layer_runs.py` | Threshold layer execution in pipeline |

---

## 5. Writing New Tests

### Naming Conventions

- **File names:** `test_<feature>_<aspect>.py` (e.g., `test_ab_statistics.py`)
- **Test classes:** `Test<Feature>` (e.g., `TestCohenH`)
- **Test methods:** `test_<scenario>_<expected_behavior>` (e.g., `test_bounds`, `test_monotonic`)
- **Directory:** Place in `unit/`, `integration/`, or `regression/` by scope

### Test Structure Template

```python
"""Brief description of what this test module covers."""
from __future__ import annotations

import pytest

from spam_fraud_detector.core.something import Something


class TestSomething:
    """Tests for the Something class."""

    def test_happy_path(self):
        """Normal usage produces expected output."""
        result = Something.do_work(input_data)
        assert result.is_valid

    def test_edge_case_empty_input(self):
        """Empty input is handled gracefully."""
        result = Something.do_work([])
        assert result.is_empty

    def test_raises_on_invalid_input(self):
        """Invalid input raises ValueError."""
        with pytest.raises(ValueError, match="invalid"):
            Something.do_work(None)
```

### Guidelines

1. **One assertion per test** when possible. Multiple related assertions are
   acceptable if they test the same logical outcome.
2. **Use descriptive test names** that read like sentences:
   - `test_holm_adjustment_never_decreases_p_value` ✓
   - `test_holm1` ✗
3. **Mock external dependencies** (MLflow, Kaggle API, network calls). Tests
   must run offline.
4. **Use fixtures from `conftest.py`** for shared setup (temporary directories,
   sample data, mocked stores).
5. **No hardcoded paths** to user-specific directories. Use `tmp_path` fixture
   or relative paths from the repo root.
6. **Add to the appropriate directory:** `unit/` for isolated logic,
   `integration/` for cross-module, `regression/` for bug guards.

---

## 6. CI/CD Integration

The test suite runs in GitHub Actions (`.github/workflows/ci.yml`):

| Job | Tests Run | Purpose |
|---|---|---|
| `lint-and-type` | `ruff check`, `mypy --strict` | Code quality and type safety |
| `unit-tests` | `pytest tests/unit --cov=... --cov-fail-under=80` | Unit tests with coverage gate |
| `explain-read-only` | `tests/regression/test_explain_is_read_only.py` | Explain-only is read-only |
| `full-pipeline` | All tests + stats + explain + audit | End-to-end pipeline validation |

Coverage gate: **80% minimum** for `spam_fraud_detector.core` in CI.
Local target: **≥90%** for `make test-cov`.

---

## 7. Troubleshooting

| Problem | Likely Cause | Fix |
|---|---|---|
| `ImportError: No module named 'spam_fraud_detector'` | Package not installed | `pip install -e ".[dev]"` |
| `tests/integration/*` skipped | `outputs/` not present | Run `make retrain` first |
| Coverage below 80% | Missing test coverage | Add unit tests for uncovered lines |
| `FileNotFoundError` in regression tests | Test data missing | Run `make pipeline` to generate artifacts |
| `AssertionError` in `test_ab_statistics.py` | Statistical formula changed | Verify against `docs/EVIDENCE_INVENTORY.json` |

---

## 8. Maintenance

### Adding a New Test Category

1. Create the directory under `tests/` (e.g., `tests/performance/`).
2. Add an `__init__.py` if needed for package imports.
3. Add a `pytest` marker in `pyproject.toml` or `setup.cfg`.
4. Add a `make test-<category>` target to the `Makefile`.
5. Document the category in this README (§4).

### Deprecating a Test

1. Move the test file to `tests/archived/` (create if needed).
2. Add a comment explaining why it was deprecated.
3. Update this README to remove the test from the category tables.
4. Remove any CI references to the test.

---

## 9. Related Documentation

| Document | Location | Relevance to Tests |
|---|---|---|
| Evidence Inventory | `spam_fraud_detector/reports/docs/EVIDENCE_INVENTORY.json` | Source of truth for statistical claims |
| Framework Whitepaper | `spam_fraud_detector/reports/docs/FRAMEWORK.md` | Architecture and reproducibility contract |
| Paper-Ready Results | `spam_fraud_detector/reports/docs/PAPER.md` | Statistical validation methodology |
| CI Workflow | `.github/workflows/ci.yml` | Test execution in CI/CD |
| Makefile | `Makefile` | `test`, `test-unit`, `test-integration`, `test-cov` targets |

---

## 10. License

MIT © Guy M. Kaptue T. See [LICENSE](../../LICENSE).
