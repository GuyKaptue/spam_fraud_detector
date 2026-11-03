"""Verify documentation against evidence inventory.

Checks:
1. Every figure reference in docs points to an existing file in docs/figures/
2. Every outputs/ path referenced in docs exists in the inventory
3. No TODO, TBD, coming soon, FIXME, XXX in any doc
4. No placeholder names in explainability docs
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

INVENTORY_PATH = Path("spam_fraud_detector/reports/docs/EVIDENCE_INVENTORY.json")
DOCS_DIR = Path("spam_fraud_detector/reports/docs")
FIGURES_DIR = DOCS_DIR / "figures"


def load_inventory() -> dict:
    """Load the evidence inventory."""
    if not INVENTORY_PATH.exists():
        print(f"ERROR: {INVENTORY_PATH} does not exist. Run `make evidence` first.")
        sys.exit(1)
    with open(INVENTORY_PATH, encoding="utf-8") as f:
        return json.load(f)


def _find_figures_in_text(text: str) -> list[str]:
    """Find all figure references in markdown text."""
    pattern = r'(?:!\[.*?\]\()?(spam_fraud_detector/reports/docs/figures/[^)]+\.png)'
    return re.findall(pattern, text)


def _find_output_paths_in_text(text: str) -> list[str]:
    """Find all outputs/ references in docs text."""
    pattern = r'(?:outputs/[^)\s\)]+)'
    return re.findall(pattern, text)


def verify_casefiles(inventory: dict) -> list[str]:
    """Verify all case file references in docs exist and point to real files."""
    errors = []

    for doc_path in sorted(DOCS_DIR.glob("*.md")):
        text = doc_path.read_text(encoding="utf-8")
        figure_refs = _find_figures_in_text(text)

        for fig_ref in figure_refs:
            fig_path = Path(fig_ref)
            if not fig_path.exists():
                errors.append(f"{doc_path}: figure not found: {fig_path}")

    return errors


def verify_output_paths(inventory: dict) -> list[str]:
    """Verify all outputs/ paths in docs exist in inventory."""
    errors = []

    # Build set of all artifact paths from inventory
    inventory_paths = set()

    # Models
    for models in inventory.get("models", {}).values():
        for model_data in models.values():
            if model_data.get("bundle_path"):
                inventory_paths.add(model_data["bundle_path"])
            if model_data.get("meta_path"):
                inventory_paths.add(model_data["meta_path"])
            if model_data.get("thresholds_path"):
                inventory_paths.add(model_data["thresholds_path"])

    # Stats
    for data in inventory.get("stats", {}).values():
        if data.get("ab_test_path"):
            inventory_paths.add(data["ab_test_path"])
        if data.get("summary_md_path"):
            inventory_paths.add(data["summary_md_path"])

    # Dashboard
    for table_data in inventory.get("dashboard", {}).values():
        if table_data.get("parquet"):
            inventory_paths.add(table_data["parquet"])
        if table_data.get("csv"):
            inventory_paths.add(table_data["csv"])

    # Explainability
    for data in inventory.get("explainability", {}).values():
        for cf in data.get("casefiles", []):
            if cf.get("json"):
                inventory_paths.add(cf["json"])
        for summary in data.get("summaries", []):
            if summary:
                inventory_paths.add(summary)

    # Check all docs references
    for doc_path in sorted(DOCS_DIR.glob("*.md")):
        text = doc_path.read_text(encoding="utf-8")
        output_refs = _find_output_paths_in_text(text)

        for ref in output_refs:
            # Strip trailing backticks and punctuation from markdown
            ref = re.sub(r'[`"\)).,;]+$', '', ref)
            # Skip directory references (ending with /) or bare "outputs/"
            if ref.endswith("/") or ref == "outputs/":
                continue
            # Skip wildcard/template paths
            if "*" in ref or "{" in ref or "<" in ref:
                continue
            # Normalize: strip leading "outputs/" to match inventory paths
            norm_ref = ref.removeprefix("outputs/")
            norm_ref = norm_ref.rstrip("/")
            if norm_ref not in inventory_paths:
                errors.append(f"{doc_path}: output path not in inventory: {ref}")

    return errors


def verify_no_placeholders() -> list[str]:
    """Check no placeholder text in docs."""
    errors = []

    placeholder_patterns = [
        r"TODO",
        r"TBD",
        r"coming soon",
        r"FIXME",
        r"XXX",
        r"Subject \d+",
    ]

    for doc_path in sorted(DOCS_DIR.glob("*.md")):
        text = doc_path.read_text(encoding="utf-8")
        for pattern in placeholder_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                errors.append(f"{doc_path}: placeholder text found: {match}")

    return errors


def main() -> int:
    inventory = load_inventory()

    all_errors = []

    # Check figure references
    errors = verify_casefiles(inventory)
    all_errors.extend(errors)

    # Check output paths
    errors = verify_output_paths(inventory)
    all_errors.extend(errors)

    # Check placeholders
    errors = verify_no_placeholders()
    all_errors.extend(errors)

    if all_errors:
        print("verify-documentation: FAIL")
        for e in all_errors:
            print(f"  {e}")
        return 1

    print("verify-documentation: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
