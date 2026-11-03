"""Verify explain-only is read-only: bundle and split hashes unchanged after run."""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest


def _tree_hash(directory: Path, exclude: tuple = ("*.meta.json",)) -> str:
    """Return a deterministic hash of all files under directory, excluding
    files matching any pattern in ``exclude`` (e.g. meta.json which gets
    explainability info added during explain-only runs — that is expected)."""
    h = hashlib.sha256()
    for p in sorted(directory.rglob("*")):
        if p.is_file():
            # Skip excluded files (e.g. meta.json gains explainability block)
            if any(p.match(pat) for pat in exclude):
                continue
            rel = p.relative_to(directory)
            h.update(rel.as_posix().encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def _run_explain(persona: str, task: str) -> subprocess.CompletedProcess:
    return subprocess.run(  # noqa: PLW1510
        [sys.executable, "-m", "spam_fraud_detector.main",
         "--dataset", task,
         "--explainer", "shap",
         "--explain-only", "--persona", persona],
        capture_output=True, text=True, timeout=300,
    )


def test_explain_fraud_is_read_only():
    """Explain-only must not modify bundles or splits."""
    domain = "fraud_detection"
    task = "fraud"
    bundle_dir = Path("outputs/models") / domain / "all"
    val_path = Path("data/processed") / task / "01_split" / "val.parquet"

    if not bundle_dir.exists() or not val_path.exists():
        pytest.skip(f"Bundles or val split missing for {task}")

    bundle_hash_before = _tree_hash(bundle_dir, exclude=("*.meta.json",))
    val_hash_before = _tree_hash(val_path.parent)

    proc = _run_explain("fraud", task)
    assert proc.returncode == 0, proc.stderr

    bundle_hash_after = _tree_hash(bundle_dir, exclude=("*.meta.json",))
    val_hash_after = _tree_hash(val_path.parent)

    assert bundle_hash_before == bundle_hash_after, (
        "Explain-only modified bundle files (excluding meta.json)!"
    )
    assert val_hash_before == val_hash_after, (
        "Explain-only modified val split!"
    )


def test_explain_anomaly_is_read_only():
    """Explain-only anomaly must not modify bundles or splits."""
    domain = "fraud_detection"
    task = "fraud"
    bundle_dir = Path("outputs/models") / domain / "all"
    val_path = Path("data/processed") / task / "01_split" / "val.parquet"

    if not bundle_dir.exists() or not val_path.exists():
        pytest.skip(f"Bundles or val split missing for {task}")

    bundle_hash_before = _tree_hash(bundle_dir, exclude=("*.meta.json",))
    val_hash_before = _tree_hash(val_path.parent)

    proc = _run_explain("anomaly", task)
    assert proc.returncode == 0, proc.stderr

    bundle_hash_after = _tree_hash(bundle_dir, exclude=("*.meta.json",))
    val_hash_after = _tree_hash(val_path.parent)

    assert bundle_hash_before == bundle_hash_after, (
        "Explain-only modified bundle files (excluding meta.json)!"
    )
    assert val_hash_before == val_hash_after, (
        "Explain-only modified val split!"
    )