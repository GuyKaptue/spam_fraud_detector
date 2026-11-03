"""Unit tests for scripts/check_docs_paths.py."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "check_docs_paths.py"


def _run(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=str(cwd),
        capture_output=True,
        text=True, check=False,
    )


def test_stray_root_docs_fails(tmp_path: Path) -> None:
    # create a synthetic repo tree with a stray docs/ at root
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "foo.md").write_text("# foo")
    (tmp_path / "spam_fraud_detector" / "reports" / "docs").mkdir(parents=True)
    (tmp_path / "spam_fraud_detector" / "reports" / "docs" / "CORE_README_STYLE.md").write_text("# style")
    result = _run(tmp_path, "--strict")
    assert result.returncode != 0
    assert "forbidden root directory" in result.stdout or "forbidden root directory" in result.stderr


def test_canonical_reference_passes(tmp_path: Path) -> None:
    (tmp_path / "spam_fraud_detector" / "reports" / "docs").mkdir(parents=True)
    (tmp_path / "spam_fraud_detector" / "reports" / "docs" / "MAKEFILE.md").write_text("# makefile")
    (tmp_path / "README.md").write_text("See spam_fraud_detector/reports/docs/MAKEFILE.md")
    result = _run(tmp_path, "--strict")
    # should pass since no root docs/ and no root-relative refs
    assert result.returncode == 0


def test_ignore_comment_passes(tmp_path: Path) -> None:
    (tmp_path / "spam_fraud_detector" / "reports" / "docs").mkdir(parents=True)
    (tmp_path / "README.md").write_text("# see docs/FRAMEWORK.md  # docs-path-lint: ignore")
    result = _run(tmp_path, "--strict")
    # the ignore comment should make it pass
    assert result.returncode == 0


def test_json_output(tmp_path: Path) -> None:
    (tmp_path / "spam_fraud_detector" / "reports" / "docs").mkdir(parents=True)
    result = _run(tmp_path, "--json")
    import json
    data = json.loads(result.stdout)
    assert "ok" in data
    assert "messages" in data