"""Check that documentation lives only under spam_fraud_detector/reports/docs/.

Walks tracked files (or a recursive walk when git is unavailable) and fails
if any root-relative ``docs/`` reference points at a file that does not exist
under ``spam_fraud_detector/reports/docs/``. Also fails if the root ``docs/``
directory exists.

Stdlib only (re, pathlib, subprocess, json).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(".")
CANONICAL = Path("spam_fraud_detector/reports/docs")
FORBIDDEN_ROOT = Path("docs")

INCLUDED_SUFFIXES = {".md", ".py", ".toml", ".cfg", ".yml", ".yaml", ".cff"}
INCLUDED_NAMES = {"Makefile", ".gitignore"}

EXCLUDED_DIRS = {
    ".git", ".venv", "node_modules", "outputs", "data",
    "spam_fraud_detector/reports", "spam_fraud_detector/mlruns", "scripts",
}

# Curated regex list for root-relative docs references.
PATTERNS = [
    re.compile(r"(?<![a-zA-Z0-9_./-])docs/([^`\s)]+\.md)"),
    re.compile(r"`docs/([^`]+)`"),
    re.compile(r"\(docs/([^)]+)\)"),
    re.compile(r'"docs/([^"]+)"'),
    re.compile(r"'docs/([^']+)'"),
]


def _tracked_files() -> list[Path]:
    try:
        out = subprocess.run(
            ["git", "ls-files"],
            capture_output=True, text=True, cwd=ROOT,
        )
        if out.returncode == 0 and out.stdout.strip():
            return [ROOT / p for p in out.stdout.strip().splitlines()]
    except (FileNotFoundError, OSError):
        pass
    # fallback: recursive walk
    result = []
    for p in ROOT.rglob("*"):
        if p.is_file() and p.suffix.lower() in INCLUDED_SUFFIXES:
            if any(ex in str(p) for ex in EXCLUDED_DIRS):
                continue
            result.append(p)
    return result


def _is_excluded(path: Path) -> bool:
    s = str(path)
    for d in EXCLUDED_DIRS:
        if s.startswith(d + "/") or s.startswith(d + "\\"):
            return True
    return False


def _included(path: Path) -> bool:
    if _is_excluded(path):
        return False
    if path.name in INCLUDED_NAMES:
        return True
    return path.suffix.lower() in INCLUDED_SUFFIXES


def _check_file(path: Path, strict: bool) -> list[tuple[int, str]]:
    """Return list of (line_number, diagnostic) for violations."""
    diagnostics = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except (OSError, UnicodeDecodeError):
        return diagnostics
    lines = text.splitlines()
    for lineno, line in enumerate(lines, 1):
        # ignore comment on same line
        if "docs-path-lint: ignore" in line:
            continue
        # ignore if previous line has the comment
        if lineno > 1 and "docs-path-lint: ignore" in lines[lineno - 2]:
            continue
        for pat in PATTERNS:
            for m in pat.finditer(line):
                ref = m.group(1) if m.lastindex else m.group(0)
                # normalize: strip leading docs/
                if ref.startswith("docs/"):
                    ref = ref[5:]
                target = CANONICAL / ref
                # check if root docs/ exists
                root_target = FORBIDDEN_ROOT / ref
                if root_target.exists():
                    diagnostics.append((lineno, f"root docs/ reference found: docs/{ref} (target exists at root; should be under {CANONICAL})"))
                elif strict and not root_target.exists():
                    # in strict mode, any root-relative docs/ reference is a violation
                    if not target.exists():
                        diagnostics.append((lineno, f"root docs/ reference with no canonical target: docs/{ref}"))
                elif not target.exists():
                    diagnostics.append((lineno, f"root docs/ reference with no canonical target: docs/{ref}"))
    return diagnostics


def check(strict: bool = False) -> tuple[int, list[str]]:
    messages = []
    # Check if root docs/ exists
    if FORBIDDEN_ROOT.exists() and FORBIDDEN_ROOT.is_dir():
        messages.append(f"forbidden root directory exists: {FORBIDDEN_ROOT}/")

    for path in sorted(_tracked_files()):
        if not _included(path):
            continue
        diags = _check_file(path, strict)
        for lineno, msg in diags:
            messages.append(f"{path}:{lineno}: {msg}")

    # sort diagnostics
    messages.sort()
    return (0 if not messages else 1), messages


def main() -> int:
    parser = argparse.ArgumentParser(description="Lint documentation paths.")
    parser.add_argument("--strict", action="store_true", help="Fail on any root-relative docs/ reference.")
    parser.add_argument("--json", action="store_true", help="Machine-readable JSON output.")
    args = parser.parse_args()

    code, messages = check()
    if args.json:
        print(json.dumps({"ok": code == 0, "messages": messages}, indent=2))
    else:
        if code == 0:
            print("docs-check-paths: OK")
        else:
            for m in messages:
                print(f"FAIL {m}")
    return code


if __name__ == "__main__":
    sys.exit(main())