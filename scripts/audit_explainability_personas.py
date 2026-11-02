"""Audit: persona wiring exists in source (not just artifacts)."""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

EXPLAINER = Path("spam_fraud_detector/core/explainer.py")
RESULT_STORE = Path("spam_fraud_detector/core/result_store.py")
MAIN = Path("spam_fraud_detector/main.py")
CONTRACT = Path("outputs/CONTRACT.json")


def _has_subclass(src: str, base: str, *names: str) -> list[str]:
    tree = ast.parse(src)
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for base_node in node.bases:
                base_name = None
                if isinstance(base_node, ast.Name):
                    base_name = base_node.id
                elif isinstance(base_node, ast.Attribute):
                    base_name = base_node.attr
                if base_name == base and node.name in names:
                    found.append(node.name)
    return found


def _has_function(src: str, name: str) -> bool:
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return True
    return False


def _function_has_kwarg(src: str, func_name: str, kwarg: str) -> bool:
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            for arg in node.args.args + node.args.kwonlyargs:
                if arg.arg == kwarg:
                    return True
            # check defaults in kwonly
            for arg in node.args.kwonlyargs:
                if arg.arg == kwarg:
                    return True
    return False


def _function_contains_string(src: str, func_name: str, string: str) -> bool:
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            for child in ast.walk(node):
                if isinstance(child, ast.Constant) and isinstance(child.value, str):
                    if string in child.value:
                        return True
    return False


def audit() -> tuple[int, list[str]]:
    msgs = []

    # 1. TextExplainer and AnomalyExplainer are subclasses of ModelExplainer
    if EXPLAINER.exists():
        src = EXPLAINER.read_text(encoding="utf-8")
        found = _has_subclass(src, "ModelExplainer", "TextExplainer", "AnomalyExplainer")
        if "TextExplainer" not in found:
            msgs.append("explainer.py: TextExplainer not defined as subclass of ModelExplainer")
        if "AnomalyExplainer" not in found:
            msgs.append("explainer.py: AnomalyExplainer not defined as subclass of ModelExplainer")

        # 2. make_explainer has branches for all three personas
        if not _has_function(src, "make_explainer"):
            msgs.append("explainer.py: make_explainer factory not defined")
        else:
            tree = ast.parse(src)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == "make_explainer":
                    # check for persona string constants
                    persona_strings = set()
                    for child in ast.walk(node):
                        if isinstance(child, ast.Constant) and isinstance(child.value, str):
                            if child.value in ("fraud", "spam", "anomaly"):
                                persona_strings.add(child.value)
                    if not {"fraud", "spam", "anomaly"}.issubset(persona_strings):
                        missing = {"fraud", "spam", "anomaly"} - persona_strings
                        msgs.append(f"explainer.py: make_explainer missing persona branches: {missing}")

    # 3. record_explainability accepts persona kwarg
    if RESULT_STORE.exists():
        src = RESULT_STORE.read_text(encoding="utf-8")
        if not _has_function(src, "record_explainability"):
            msgs.append("result_store.py: record_explainability() not defined")
        elif not _function_has_kwarg(src, "record_explainability", "persona"):
            msgs.append("result_store.py: record_explainability() missing 'persona' kwarg")

        # 4. _write_dashboard_tables writes dim_explainability
        if not _function_contains_string(src, "_write_dashboard_tables", "dim_explainability"):
            msgs.append("result_store.py: _write_dashboard_tables does not write dim_explainability")

    # 5. main.py references make_explainer
    if MAIN.exists():
        src = MAIN.read_text(encoding="utf-8")
        if "make_explainer" not in src:
            msgs.append("main.py: make_explainer not referenced")

    # 6. CONTRACT.json explainability.personas
    if CONTRACT.exists():
        try:
            contract = json.loads(CONTRACT.read_text())
            dp = contract.get("explainability", {})
            personas = dp.get("personas", [])
            if personas != ["fraud", "spam", "anomaly"]:
                msgs.append(f"CONTRACT.json: explainability.personas = {personas}, expected ['fraud', 'spam', 'anomaly']")
        except (json.JSONDecodeError, OSError) as e:
            msgs.append(f"CONTRACT.json: parse error: {e}")

    return (0 if not msgs else 1), msgs


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit explainability persona wiring.")
    parser.add_argument("--json", action="store_true", help="Machine-readable JSON output.")
    args = parser.parse_args()

    code, msgs = audit()
    if args.json:
        print(json.dumps({"ok": code == 0, "messages": msgs}, indent=2))
    else:
        if code == 0:
            print("audit_explainability_personas: OK")
        else:
            for m in msgs:
                print(f"FAIL {m}")
    return code


if __name__ == "__main__":
    sys.exit(main())