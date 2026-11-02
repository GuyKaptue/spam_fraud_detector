"""Verify explainability provenance hashes."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def _sha256_json(obj) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


ALLOWED_ACTIONS = {"block", "escalate", "review", "monitor", "clear"}
ALLOWED_PRIORITIES = {"high", "medium", "low"}
ALLOWED_CONFIDENCES = {"high", "medium", "low"}
RECOMMENDATION_FIELDS = [
    "action", "priority", "reason", "suggested_next_steps",
    "estimated_cost_if_missed", "confidence", "reviewer_notes_template",
]


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
                continue
            expl = meta.get("explainability")
            if expl is None or expl.get("backend") == "off":
                continue
            # check sidecar
            sp = expl.get("summary_path")
            if sp:
                sidecar = Path(sp + ".provenance.json")
                if strict and not sidecar.exists():
                    msgs.append(f"{meta_path}: missing provenance sidecar: {sidecar}")
                    continue
                if sidecar.exists():
                    try:
                        sidecar_data = json.loads(sidecar.read_text())
                        if sidecar_data.get("background_hash") != expl.get("background_hash"):
                            msgs.append(f"{meta_path}: background_hash mismatch in sidecar")
                        if sidecar_data.get("feature_names_hash") != expl.get("feature_names_hash"):
                            msgs.append(f"{meta_path}: feature_names_hash mismatch in sidecar")
                    except (json.JSONDecodeError, OSError):
                        msgs.append(f"{meta_path}: invalid sidecar JSON")
            # check paths don't contain .. or absolute
            if sp and (".." in sp or sp.startswith("/")):
                msgs.append(f"{meta_path}: invalid summary_path: {sp}")
            for wp in expl.get("waterfall_paths", []):
                if wp and (".." in wp or wp.startswith("/")):
                    msgs.append(f"{meta_path}: invalid waterfall_path: {wp}")
            # check casefiles + recommendations
            casefiles = expl.get("casefiles") or []
            for cf in casefiles:
                for key in ("json", "png", "subject_id", "subject_name", "case_type"):
                    if key not in cf:
                        msgs.append(f"{meta_path}: casefile missing key {key}: {cf}")
                ct = cf.get("case_type")
                if ct not in ("empirical", "hypothetical"):
                    msgs.append(f"{meta_path}: invalid case_type {ct}: {cf}")
                if ct == "hypothetical" and "scenario_id" not in cf:
                    msgs.append(f"{meta_path}: hypothetical casefile missing scenario_id: {cf}")
                for key in ("json", "png"):
                    p = cf.get(key)
                    if p and not Path(p).exists():
                        msgs.append(f"{meta_path}: casefile {key} missing: {p}")
                    elif p and key == "png" and Path(p).exists() and Path(p).stat().st_size < 5000:
                        msgs.append(f"{meta_path}: casefile png too small: {p}")
                jp = cf.get("json")
                if jp and Path(jp).exists():
                    try:
                        jdata = json.loads(Path(jp).read_text())
                        rec = jdata.get("recommendation")
                        if not isinstance(rec, dict):
                            msgs.append(f"{jp}: missing recommendation block")
                            continue
                        for f in RECOMMENDATION_FIELDS:
                            if f not in rec:
                                msgs.append(f"{jp}: recommendation missing field {f}")
                        if rec.get("action") not in ALLOWED_ACTIONS:
                            msgs.append(f"{jp}: invalid recommendation action: {rec.get('action')}")
                        if rec.get("priority") not in ALLOWED_PRIORITIES:
                            msgs.append(f"{jp}: invalid recommendation priority: {rec.get('priority')}")
                        if rec.get("confidence") not in ALLOWED_CONFIDENCES:
                            msgs.append(f"{jp}: invalid recommendation confidence: {rec.get('confidence')}")
                        if not rec.get("suggested_next_steps"):
                            msgs.append(f"{jp}: recommendation suggested_next_steps empty")
                        tmpl = rec.get("reviewer_notes_template", "")
                        if "[NAME]" not in tmpl or "[DATE]" not in tmpl:
                            msgs.append(f"{jp}: reviewer_notes_template missing [NAME]/[DATE]")
                    except (json.JSONDecodeError, OSError) as exc:
                        msgs.append(f"{meta_path}: invalid casefile JSON {jp}: {exc}")
    return (0 if not msgs else 1), msgs


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify explainability provenance.")
    parser.add_argument("--strict", action="store_true", help="Fail on missing sidecars.")
    parser.add_argument("--json", action="store_true", help="Machine-readable JSON output.")
    parser.add_argument("--domain", default="both", help="Domain: spam, fraud, or both.")
    args = parser.parse_args()

    code, msgs = check(domain=args.domain, strict=args.strict)
    if args.json:
        print(json.dumps({"ok": code == 0, "violations": msgs}, indent=2))
    else:
        if code == 0:
            print("verify-explainability: OK")
        else:
            for m in msgs:
                print(f"FAIL {m}")
    return code


if __name__ == "__main__":
    sys.exit(main())