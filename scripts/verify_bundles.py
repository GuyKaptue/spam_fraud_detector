"""Verify bundles: 5 files per model, SHA-256 match."""
import hashlib
import json
import sys
from pathlib import Path


def main() -> int:
    root = Path("outputs")
    if not root.exists():
        print("no outputs dir; nothing to verify")
        return 0
    errors = []
    for domain in ["spam_detection", "fraud_detection"]:
        mpath = root / "models" / domain / "manifest.json"
        if not mpath.exists():
            continue
        manifest = json.loads(mpath.read_text())
        for m in manifest.get("models", []):
            if m.get("skipped"):
                continue
            art = m.get("artifact")
            if not art:
                continue
            base = root / Path(art).with_suffix("")
            for ext in [".joblib", ".meta.json", ".eval.json", ".curve.json", ".confusion.json"]:
                p = Path(str(base) + ext)
                if not p.exists():
                    errors.append(f"missing {p}")
            joblib_path = Path(str(base) + ".joblib")
            meta_path = Path(str(base) + ".meta.json")
            if joblib_path.exists() and meta_path.exists():
                h = hashlib.sha256()
                with open(joblib_path, "rb") as f:
                    for chunk in iter(lambda: f.read(8192), b""):
                        h.update(chunk)
                sha = "sha256:" + h.hexdigest()
                meta = json.loads(meta_path.read_text())
                if meta.get("artifact_sha256") and meta["artifact_sha256"] != sha:
                    errors.append(f"sha mismatch for {joblib_path}")
    if errors:
        for e in errors:
            print(e)
        return 1
    print("verify-bundles: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())