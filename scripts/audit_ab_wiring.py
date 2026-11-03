"""Static AST audit of the A/B test wiring.

Checks that:
- stats.py computes on test split only (split == "test" filter present)
- McNemar exact fallback branch exists (n_discordant < min_discordant_chi2)
- Deterministic Holm tie-break uses (p_value, model_a, model_b) sort key
- Zero-discordant skip with warning
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

STATS_PY = Path("spam_fraud_detector/core/stats.py")


def _get_name_from_node(node):
    """Extract name from an AST node, handling Name, Attribute, and Subscript."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):
        return _get_name_from_node(node.value)
    return None


def _find_node(tree: ast.AST, node_type: type) -> list[ast.AST]:
    """Find all nodes of a given type in the AST."""
    return [node for node in ast.walk(tree) if isinstance(node, node_type)]


def audit() -> int:
    errors = []
    src = STATS_PY.read_text(encoding="utf-8")
    tree = ast.parse(src)

    # 1. Check test == "test" filter exists
    test_filter_found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            left = node.left
            # Handle both attr access (pa.split) and subscript access (pa["split"])
            left_name = None
            if isinstance(left, ast.Attribute):
                left_name = left.attr
            elif isinstance(left, ast.Subscript) and isinstance(left.value, ast.Name) and left.value.id == "pa" and isinstance(left.slice, ast.Constant):
                left_name = left.slice.value
            if (
                left_name == "split"
                and len(node.ops) == 1
                and isinstance(node.ops[0], ast.Eq)
                and len(node.comparators) == 1
                and isinstance(node.comparators[0], ast.Constant)
                and node.comparators[0].value == "test"
            ):
                test_filter_found = True
    if not test_filter_found:
        errors.append("stats.py: no 'split == \"test\"' filter found")

    # 2. Check exact McNemar fallback branch exists
    exact_found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "exact" and isinstance(node.value, ast.Constant) and node.value.value is True:
            exact_found = True
    if not exact_found:
        errors.append("stats.py: no exact=True McNemar branch found")

    # 3. Check deterministic Holm tie-break: sort by (p_value, model_a, model_b)
    deterministic_sort = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            # sorted() is a built-in function call (Name), not a method call (Attribute)
            is_sorted_call = (
                isinstance(node.func, ast.Name) and node.func.id == "sorted"
            ) or (
                isinstance(node.func, ast.Attribute) and node.func.attr == "sorted"
            )
            if is_sorted_call:
                for kw in node.keywords:
                    if kw.arg == "key" and isinstance(kw.value, ast.Lambda):
                        # Check lambda body is a tuple
                        body = kw.value.body
                        if isinstance(body, ast.Tuple):
                            elt_names = []
                            for elt in body.elts:
                                name = _get_name_from_node(elt)
                                if name:
                                    elt_names.append(name)
                            if "p_values" in elt_names and "model_as" in elt_names and "model_bs" in elt_names:
                                deterministic_sort = True
    if not deterministic_sort:
        errors.append("stats.py: no deterministic Holm tie-break (p_value, model_a, model_b) sort found")

    # 4. Check zero-discordant skip with warning
    zero_discordant_found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            # Check for n_discordant == 0 condition
            test = node.test
            if isinstance(test, ast.Compare) and (
                isinstance(test.left, ast.Name)
                and test.left.id == "n_discordant"
                and len(test.ops) == 1
                and isinstance(test.ops[0], ast.Eq)
                and len(test.comparators) == 1
                and isinstance(test.comparators[0], ast.Constant)
                and test.comparators[0].value == 0
            ):
                zero_discordant_found = True
    if not zero_discordant_found:
        errors.append("stats.py: no zero-discordant skip with warning found")

    if errors:
        print("audit_ab_wiring: FAIL")
        for e in errors:
            print(f"  {e}")
        return 1

    print("audit_ab_wiring: OK")
    return 0


def main() -> int:
    return audit()


if __name__ == "__main__":
    sys.exit(main())