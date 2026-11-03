"""Check explainability artifacts for persona compliance."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _is_tree_model(name: str) -> bool:
    return any(t in name for t in ["XGB", "LGBM", "RandomForest", "GradientBoosting", "ExtraTrees"])


def check(domain: str = "both", strict: bool = False) -> tuple[int, list[str]]:
    msgs = []
    domains = ["spam", "fraud"] if domain == "both" else [domain]
    for d in domains:
        domain_name = f"{d}_detection"
        meta_dir = Path("outputs") / "models" / domain_name / "all"
        if not meta_dir.exists():
            continue
        for meta_path in meta_dir.glob("*.meta.json"):
            try:
                meta = json.loads(meta_path.read_text())
            except (json.JSONDecodeError, OSError):
                msgs.append(f"{meta_path}: invalid JSON")
                continue
            group = meta.get("group", "")
            name = meta.get("model", meta_path.stem.split("__")[0])
            explainability = meta.get("explainability")
            is_tree = _is_tree_model(name)
            is_anomaly = group == "anomaly"

            if is_anomaly:
                if explainability is not None and explainability.get("persona") != "anomaly":
                    msgs.append(f"{meta_path}: anomaly model must have persona=anomaly or null")
            elif is_tree:
                expected_persona = "fraud" if d == "fraud" else "spam"
                if explainability is not None and explainability.get("persona") not in (expected_persona, None):
                    msgs.append(f"{meta_path}: tree model must have persona={expected_persona} or null")
                if explainability is not None:
                    sp = explainability.get("summary_path")
                    if sp and not Path(sp).exists():
                        msgs.append(f"{meta_path}: summary_path not found: {sp}")
                    if sp and not sp.startswith("reports/"):
                        msgs.append(f"{meta_path}: summary_path must be relative under reports/")
                    for wp in explainability.get("waterfall_paths", []):
                        if wp and not Path(wp).exists():
                            msgs.append(f"{meta_path}: waterfall_path not found: {wp}")
                    if explainability.get("background_hash") and not explainability["background_hash"].startswith("sha256:"):
                        msgs.append(f"{meta_path}: invalid background_hash format")
                    if explainability.get("feature_names_hash") and not explainability["feature_names_hash"].startswith("sha256:"):
                        msgs.append(f"{meta_path}: invalid feature_names_hash format")

            if strict and is_tree and explainability is not None and not explainability.get("summary_path"):
                    msgs.append(f"{meta_path}: tree model missing summary_path")

    return (0 if not msgs else 1), msgs


def main() -> int:
    parser = argparse.ArgumentParser(description="Check explainability artifacts.")
    parser.add_argument("--strict", action="store_true", help="Fail on missing required artifacts.")
    parser.add_argument("--json", action="store_true", help="Machine-readable JSON output.")
    parser.add_argument("--domain", default="both", help="Domain: spam, fraud, or both.")
    args = parser.parse_args()

    code, msgs = check(domain=args.domain, strict=args.strict)
    if args.json:
        print(json.dumps({"ok": code == 0, "violations": msgs}, indent=2))
    else:
        if code == 0:
            print("check-explainability: OK")
        else:
            for m in msgs:
                print(f"FAIL {m}")
    return code


if __name__ == "__main__":
    sys.exit(main())