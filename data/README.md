# Data

Canonical local dataset layout.

- `raw/{spam,fraud}/` untouched Kaggle extracts (git-ignored except `.gitkeep`).
- `processed/{spam,fraud}/` deterministic cleaned artifacts consumed by `KaggleDataLoader`.

Run `make data` to ensure folders and build processed artifacts.
