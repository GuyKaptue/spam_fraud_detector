"""Verify case files have proper names (no placeholder 'Subject NNN').

Checks that every case file JSON under reports/<domain>/docs/explainability/
has a full_name that does not start with "Subject " and does not equal the
subject_id (i.e., not a raw filename fallback).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def verify_casefiles(strict: bool = False) -> int:
    """Return 0 if all case files have proper names, 1 otherwise."""
    errors = []
    reports_dir = Path("reports")

    for domain_dir in sorted(reports_dir.glob("*_detection")):
        explain_dir = domain_dir / "docs" / "explainability"
        if not explain_dir.exists():
            continue

        for json_path in sorted(explain_dir.glob("*.json")):
            try:
                casefile = json.loads(json_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                errors.append(f"{json_path}: invalid JSON: {exc}")
                continue

            subject = casefile.get("subject", {}) or {}
            full_name = subject.get("full_name", "")
            subject_id = subject.get("subject_id", "")

            # Check for placeholder names
            if full_name.startswith("Subject "):
                errors.append(
                    f"{json_path}: placeholder name '{full_name}' "
                    f"(subject_id={subject_id})"
                )

            # Check that full_name != subject_id (raw filename fallback)
            if full_name == subject_id:
                errors.append(
                    f"{json_path}: full_name equals subject_id '{subject_id}' "
                    f"(raw filename fallback)"
                )

    if errors:
        print("verify-casefiles: FAIL")
        for e in errors:
            print(f"  {e}")
        return 1

    print("verify-casefiles: OK")
    return 0


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Verify case files have proper names.")
    parser.add_argument("--strict", action="store_true", help="Fail on any placeholder name.")
    args = parser.parse_args()

    return verify_casefiles(strict=args.strict)


if __name__ == "__main__":
    sys.exit(main())