"""Unit tests for scripts/check_core_readme.py."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
CORE = REPO / "spam_fraud_detector" / "core"
README = CORE / "README.md"
SCRIPT = REPO / "scripts" / "check_core_readme.py"


def _run(tmp_path: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--json"],
        cwd=str(tmp_path),
        capture_output=True,
        text=True, check=False,
    )


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "spam_fraud_detector" / "core").mkdir(parents=True)
    (tmp_path / "scripts").mkdir()
    for p in (REPO / "spam_fraud_detector" / "core").glob("*.py"):
        shutil.copy(p, tmp_path / "spam_fraud_detector" / "core" / p.name)
    shutil.copy(README, tmp_path / "spam_fraud_detector" / "core" / "README.md")
    shutil.copy(SCRIPT, tmp_path / "scripts" / "check_core_readme.py")
    return tmp_path


def test_pass_on_committed_readme(workspace: Path) -> None:
    proc = _run(workspace)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_short_what_it_does_fails(workspace: Path) -> None:
    readme = workspace / "spam_fraud_detector" / "core" / "README.md"
    text = readme.read_text(encoding="utf-8")
    # Replace the entire "What it does" paragraph for split module
    section_start = text.find("## `split`")
    section_end = text.find("## `quality`")
    section = text[section_start:section_end]
    # Find "### What it does" within the section
    wd_start = section.find("### What it does")
    wd_end = section.find("### How it does it")
    old_para = section[wd_start:wd_end]
    new_para = "### What it does\n\nShort.\n\n"
    text = text.replace(old_para, new_para, 1)
    readme.write_text(text, encoding="utf-8")
    proc = _run(workspace)
    assert proc.returncode == 1
    assert "words, need >= 40" in proc.stdout


def test_stray_markdown_fails(workspace: Path) -> None:
    (workspace / "spam_fraud_detector" / "core" / "notes.md").write_text("x", encoding="utf-8")
    proc = _run(workspace)
    assert proc.returncode == 1
    assert "expected exactly 1 .md" in proc.stdout


def test_unmentioned_public_name_fails(workspace: Path) -> None:
    readme = workspace / "spam_fraud_detector" / "core" / "README.md"
    text = readme.read_text(encoding="utf-8")
    text = text.replace("StatsEngine", "the-stats-thing")
    readme.write_text(text, encoding="utf-8")
    proc = _run(workspace)
    assert proc.returncode == 1
    assert "public symbol 'StatsEngine' not mentioned" in proc.stdout


def test_invalid_python_snippet_fails(workspace: Path) -> None:
    readme = workspace / "spam_fraud_detector" / "core" / "README.md"
    text = readme.read_text(encoding="utf-8")
    text = text.replace(
        "from spam_fraud_detector.core.utils import set_seed, hash_file",
        "from spam_fraud_detector.core.utils import set_seed hash_file",
        1,
    )
    readme.write_text(text, encoding="utf-8")
    proc = _run(workspace)
    assert proc.returncode == 1
    assert "invalid Python snippet" in proc.stdout


def test_json_output(workspace: Path) -> None:
    proc = _run(workspace)
    import json
    data = json.loads(proc.stdout)
    assert data["status"] == "pass"
    assert data["errors"] == []
    assert data["modules_checked"] == 17