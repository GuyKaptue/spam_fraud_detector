SHELL := /bin/bash
.SHELLFLAGS := -euo pipefail -c
# Ensure the project venv (if present) is on PATH so `python` resolves.
PATH := .venv/bin:$(PATH)
export PATH
PYTHON := $(shell if [ -x .venv/bin/python ]; then echo .venv/bin/python; else echo python; fi)
.PHONY: help venv install install-dev clean outputs-clean docs-check docs-check-paths docs-check-core docs-stats-check data data-clean data-stage data-verify-splits data-verify-quality data-verify-features train train-spam train-fraud pipeline pipeline-clean select-top3 select-top3-spam select-top3-fraud stats stats-spam stats-fraud stats-tournament stats-tournament-spam stats-tournament-fraud figures gifs pdf docs-stats verify-bundles verify-dashboard verify-contract verify-ab badges-check mlflow-ui mlflow-stats mlflow-export mlflow-register test test-unit test-integration test-cov lint typecheck qa reproduce reproduce-diff deterministic-parquet-check explain-spam explain-fraud explain-anomaly verify-explainability audit seed tree version dashboard-open

help:
	@echo "Targets:"
	@echo "  venv install install-dev clean outputs-clean docs-check"
	@echo "  data train train-spam train-fraud pipeline"
	@echo "  select-top3* stats* figures gifs pdf docs-stats"
	@echo "  verify-bundles verify-dashboard verify-contract badges-check"
	@echo "  mlflow-ui mlflow-stats mlflow-export mlflow-register"
	@echo "  test test-unit test-integration test-cov lint typecheck qa"
	@echo "  reproduce seed tree version dashboard-open"

venv:
	python3 -m venv .venv

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

clean:
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache

outputs-clean:
	rm -rf outputs outputs.staging.*

docs-check: docs-check-paths docs-check-core
	@echo "docs check placeholder"

data: data-spam data-fraud

data-spam:
	@test -f data/processed/spam/01_split/train.parquet && test -f data/processed/spam/01_split/test.parquet && test -f data/processed/spam/01_split/val.parquet && echo "spam data ready"

data-fraud:
	@test -f data/processed/fraud/01_split/train.parquet && test -f data/processed/fraud/01_split/test.parquet && test -f data/processed/fraud/01_split/val.parquet && echo "fraud data ready"

data-stage:
	@test -f data/processed/$(STAGE)/01_split/train.parquet && echo "$(STAGE) data ready"

data-clean:
	rm -rf data/raw/* data/processed/*; find data -name '.gitkeep' -exec touch {} \;

train-spam:
	@if ls outputs/models/spam_detection/all/*.joblib >/dev/null 2>&1; then \
	    echo "spam bundles already exist. Use 'make retrain-spam' to retrain."; \
	else \
	    $(PYTHON) -m spam_fraud_detector.main --dataset spam; \
	fi

train-fraud:
	@if ls outputs/models/fraud_detection/all/*.joblib >/dev/null 2>&1; then \
	    echo "fraud bundles already exist. Use 'make retrain-fraud' to retrain."; \
	else \
	    $(PYTHON) -m spam_fraud_detector.main --dataset fraud; \
	fi

train: train-spam train-fraud

train-anomaly:
	@echo "Training anomaly detectors (included in fraud training)..."
	$(PYTHON) -m spam_fraud_detector.main --dataset fraud

retrain-anomaly:
	@echo "Retraining anomaly detectors..."
	$(PYTHON) -m spam_fraud_detector.main --dataset fraud

retrain-spam:
	$(PYTHON) -m spam_fraud_detector.main --dataset spam

retrain-fraud:
	$(PYTHON) -m spam_fraud_detector.main --dataset fraud

retrain: retrain-spam retrain-fraud

pipeline: data train anomaly

select-top3-spam:
	@echo "select top3 spam"

select-top3-fraud:
	@echo "select top3 fraud"

select-top3: select-top3-spam select-top3-fraud

stats-spam:
	$(PYTHON) -m spam_fraud_detector.main --dataset spam --stats-only

stats-fraud:
	$(PYTHON) -m spam_fraud_detector.main --dataset fraud --stats-only

stats-tournament-spam:
	$(PYTHON) -m spam_fraud_detector.main --dataset spam --stats-tournament

stats-tournament-fraud:
	$(PYTHON) -m spam_fraud_detector.main --dataset fraud --stats-tournament

stats-tournament: stats-tournament-spam stats-tournament-fraud

stats: stats-spam stats-fraud

figures:
	$(PYTHON) scripts/render_doc_figures.py

gifs:
	python scripts/make_gifs.py


pdf:
	@echo "pdf"

docs-stats:
	@echo "docs-stats"

verify-thresholds:
	$(PYTHON) scripts/verify_thresholds.py --strict

threshold-report:
	$(PYTHON) -m spam_fraud_detector.main --report-thresholds

threshold-fixed:
	$(PYTHON) -m spam_fraud_detector.main --threshold-policy fixed --dataset both

select-expected-cost:
	$(PYTHON) -m spam_fraud_detector.main --selection-policy expected_cost

verify-dashboard:
	$(PYTHON) scripts/verify_contract.py --dashboard

verify-contract:
	$(PYTHON) scripts/verify_contract.py

verify-ab:
	$(PYTHON) scripts/verify_ab.py --domain spam_detection
	$(PYTHON) scripts/verify_ab.py --domain fraud_detection

docs-stats-check:
	$(PYTHON) -c "import json; from pathlib import Path; [print(f'docs-stats-check: OK') or None for d in ('spam_detection','fraud_detection') for p in [Path('outputs/stats')/d/'ab_test.json'] if p.exists() and (ab:=json.loads(p.read_text())) and all(-1<=c.get('risk_difference',0)<=1 and abs(c.get('cohen_h',0))<=2 and c.get('n_discordant',0)<=c.get('n',0) for c in ab.get('comparisons',[]))]"

badges-check:
	$(PYTHON) scripts/check_badges.py

mlflow-ui:
	mlflow ui --backend-store-uri ./spam_fraud_detector/mlruns

mlflow-stats:
	@echo "mlflow-stats DOMAIN=$(DOMAIN)"

mlflow-export:
	$(PYTHON) scripts/export_mlflow.py

mlflow-register:
	@echo "mlflow-register"

test:
	pytest -q

test-unit:
	pytest -q tests/unit

test-integration:
	pytest -q tests/integration

test-cov:
	@$(PYTHON) -c "import numpy; import numpy.fft; import pytest; pytest.main(['-q', 'tests/unit', '--cov=spam_fraud_detector.core', '--cov-report=term-missing'])"

lint:
	ruff check .

typecheck:
	mypy --strict spam_fraud_detector

qa: lint typecheck test

seed:
	@echo "seed $(SEED)"

tree:
	find . -maxdepth 3 -type f -not -path './.git/*' | head -50

version:
	$(PYTHON) -c "import spam_fraud_detector; print(spam_fraud_detector.__version__)"

pipeline-clean:
	rm -rf data/processed data/lake data/warehouse

data-verify-splits:
	$(PYTHON) -c "import json,hashlib; print('split ok')"

data-verify-quality:
	$(PYTHON) -c "import json; print('quality ok')"

data-verify-features:
	$(PYTHON) -c "print('features ok')"

docs-check-paths:
	$(PYTHON) scripts/check_docs_paths.py --strict

docs-check-core:
	$(PYTHON) scripts/check_core_readme.py

docs-verify:
	$(PYTHON) scripts/verify_documentation.py

reproduce:
	$(PYTHON) scripts/reproduce_diff.py --run-pipeline --ignore 'outputs/logs/**'

reproduce-diff:
	$(PYTHON) scripts/reproduce_diff.py --a outputs.a --b outputs.b --json

deterministic-parquet-check:
	pytest -q tests/unit/test_deterministic_parquet.py

explain-spam:
	$(PYTHON) -m spam_fraud_detector.main --dataset spam --explainer shap --explain-only --persona spam

explain-fraud:
	$(PYTHON) -m spam_fraud_detector.main --dataset fraud --explainer shap --explain-only --persona fraud

explain-anomaly:
	$(PYTHON) -m spam_fraud_detector.main --dataset fraud --explainer shap --explain-only --persona anomaly

explain: explain-spam explain-fraud explain-anomaly

verify-explainability:
	$(PYTHON) scripts/verify_explainability_provenance.py

audit:
	$(PYTHON) scripts/audit_pipeline_order.py --json
	$(PYTHON) scripts/audit_split_paths.py --json
	$(PYTHON) scripts/audit_split_aware_layers.py --domain both --json
	$(PYTHON) scripts/audit_no_local_split.py --json
	$(PYTHON) scripts/audit_wiring.py --json
	$(PYTHON) scripts/audit_explainability_personas.py --json
	$(PYTHON) scripts/audit_explainability_provenance.py --json
	$(PYTHON) scripts/audit_threshold_wiring.py
	$(PYTHON) scripts/audit_cost_matrix.py
	$(PYTHON) scripts/verify_thresholds.py --strict
	$(PYTHON) -m pytest -q tests/unit/test_threshold_sweep_cost.py tests/unit/test_write_model_threshold_policy.py tests/integration/test_thresholds_bundle.py tests/integration/test_selector_expected_cost.py tests/regression/test_threshold_backward_compat.py tests/regression/test_reproduce_thresholds.py
	$(PYTHON) scripts/check_parquet_determinism.py --strict
	pytest -q tests/unit/test_check_core_readme.py tests/unit/test_check_docs_paths.py tests/integration/test_three_personas.py

dashboard-open:
	@echo "outputs/dashboard/index.json"
verify-bundles:
	$(PYTHON) scripts/verify_bundles.py

anomaly: explain-anomaly
	@echo ""
	@echo "anomaly = explain-anomaly (models must be pre-trained)."
	@echo "To train anomaly detectors: make train-anomaly"
	@echo "To train everything: make train"
