# Makefile Reference

## Documentation targets

- `docs-check`: Run all documentation checks (includes `docs-check-core`).
- `docs-check-core`: Validate `core/README.md` against the narrative contract
  (structure, subheadings, word floor, produces-table, runnable snippet,
  public-name coverage). Runs `scripts/check_core_readme.py` in < 2s.

`core/README.md` is hand-authored; see `docs/CORE_README_STYLE.md`. The check
lives in `scripts/check_core_readme.py`.

## Data targets

- `data`: Acquisition + preparation (L1-L6).
- `data-clean`: Remove processed/lake/warehouse (keeps raw).
- `data-stage`: Run a single stage via `STAGE=...`.
- `data-verify-splits`: Verify split manifest.
- `data-verify-quality`: Verify quality gate.
- `data-verify-features`: Verify feature artifacts.

## Pipeline targets

- `pipeline`: Full training + write-once + atomic finalize.
- `pipeline-clean`: Remove processed/lake/warehouse.

## Test targets

- `test`: Run all tests.
- `test-unit`: Run unit tests.
- `test-integration`: Run integration tests.
- `test-cov`: Run with coverage.

## Quality targets

- `lint`: ruff check.
- `typecheck`: mypy strict.
- `qa`: lint + typecheck + test.