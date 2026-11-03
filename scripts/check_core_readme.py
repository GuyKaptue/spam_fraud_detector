"""Check core/README.md content requirements (v3.5)."""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

CORE_DIR = Path("spam_fraud_detector/core")
README = CORE_DIR / "README.md"
MODULE_ORDER = ["comparer", "data_loader", "data_pipeline", "exceptions", "features", "fake_subjects", "case_renderer", "quality", "recommendations", "result_store", "selector", "split", "stats", "utils", "visualizer", "anomaly", "explainer"]
SUBHEADINGS = ["What it does", "How it does it", "What it produces", "How to use it"]

def _module_names() -> list:
    return sorted(p.stem for p in CORE_DIR.glob("*.py") if p.name != "__init__.py")


def _public_symbols(module: str) -> list:
    path = CORE_DIR / f"{module}.py"
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    symbols = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
            symbols.append(node.name)
    return symbols


def _find_section(text: str, module: str) -> str | None:
    heading = f"## `{module}`"
    idx = text.find(heading)
    if idx == -1:
        return None
    next_idx = text.find("\n## ", idx + len(heading))
    if next_idx == -1:
        return text[idx:]
    return text[idx:next_idx]
def _count_words(text: str) -> int:
    return len(text.split())


def check_all(json_output: bool = False, module_filter: str | None = None) -> tuple:
    errors = []
    warnings = []
    modules_to_check = [module_filter] if module_filter else _module_names()

    # 1. README exists and is the only .md under core/
    if not README.exists():
        errors.append("core/README.md does not exist")
        if json_output:
            print(json.dumps({"errors": errors, "warnings": warnings, "modules_checked": 0, "status": "fail"}, indent=2))
        return 1, errors, warnings

    md_files = list(CORE_DIR.glob("*.md"))
    if len(md_files) != 1:
        errors.append(f"expected exactly 1 .md under core/, found {len(md_files)}: {[p.name for p in md_files]}")

    text = README.read_text(encoding="utf-8")

    # 8. No HTML anchors, owner blocks, hash blocks
    for marker in ["<a id=", "<!-- core-owners:", "<!-- core-section-hashes:"]:
        if marker in text:
            errors.append(f"forbidden content found: {marker}")

    # 2. All module headings present in order
    headings_found = []
    for line in text.splitlines():
        m = re.match(r"^## `(\w+)`", line)
        if m:
            headings_found.append(m.group(1))

    for module in modules_to_check:
        if module not in headings_found:
            errors.append(f"missing section heading for module: {module}")

    if not module_filter:
        missing = [m for m in MODULE_ORDER if m not in headings_found]
        extra = [m for m in headings_found if m not in MODULE_ORDER]
        if missing:
            errors.append(f"missing module headings: {missing}")
        if extra:
            errors.append(f"unexpected module headings: {extra}")

    # 3-7. Per-module checks
    for module in modules_to_check:
        section = _find_section(text, module)
        if section is None:
            continue

        # 3. All four subheadings
        for sub in SUBHEADINGS:
            if f"### {sub}" not in section:
                errors.append(f"{module}: missing subheading '### {sub}'")

        # 4. What it does paragraph >= 40 words
        what_does_match = re.search(r"### What it does\s*\n(.*?)(?=### |\Z)", section, re.DOTALL)
        if what_does_match:
            what_does_text = what_does_match.group(1).strip()
            clean = re.sub(r"```.*?```", "", what_does_text, flags=re.DOTALL)
            clean = re.sub(r"\|.*?\|", "", clean)
            words = _count_words(clean)
            if words < 40:
                errors.append(f"{module}: 'What it does' paragraph too short ({words} words, need >= 40)")

        # 5. What it produces contains a markdown table
        produces_match = re.search(r"### What it produces\s*\n(.*?)(?=### |\Z)", section, re.DOTALL)
        if produces_match:
            produces_text = produces_match.group(1)
            if "|" not in produces_text:
                errors.append(f"{module}: 'What it produces' missing markdown table")

        # 6. How to use it contains >=1 python fenced block that parses
        how_to_match = re.search(r"### How to use it\s*\n(.*?)(?=### |\Z)", section, re.DOTALL)
        if how_to_match:
            how_to_text = how_to_match.group(1)
            blocks = re.findall(r"```python\s*\n(.*?)```", how_to_text, re.DOTALL)
            if not blocks:
                errors.append(f"{module}: 'How to use it' missing python code block")
            else:
                for block in blocks:
                    try:
                        ast.parse(block)
                    except SyntaxError as e:
                        errors.append(f"{module}: invalid Python snippet: {e}")
                        break

        # 7. Every public class/def mentioned in README
        symbols = _public_symbols(module)
        for sym in symbols:
            if sym not in text:
                errors.append(f"{module}: public symbol '{sym}' not mentioned in README")

    if json_output:
        result = {
            "errors": errors,
            "warnings": [],
            "modules_checked": len(modules_to_check),
            "status": "pass" if not errors else "fail",
        }
        print(json.dumps(result, indent=2))
    else:
        if errors:
            print("docs-check-core: FAIL")
            for e in errors:
                print(f"  - {e}")
        else:
            print("docs-check-core: OK")
            print(f"  {len(modules_to_check)} modules checked")

    return (1 if errors else 0), errors, []

def main() -> int:
    parser = argparse.ArgumentParser(description="Check core/README.md content requirements.")
    parser.add_argument("--json", action="store_true", help="Machine-readable JSON output.")
    parser.add_argument("--module", help="Check a single module.")
    args = parser.parse_args()

    code, _errors, _warnings = check_all(json_output=args.json, module_filter=args.module)
    return code


if __name__ == "__main__":
    sys.exit(main())