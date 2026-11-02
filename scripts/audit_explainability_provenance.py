"""Run persona + provenance checkers as a single audit."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

CHECKERS = [
    ("check_explainability", ["python", "scripts/check_explainability.py", "--strict", "--json"]),
    ("verify_explainability_provenance", ["python", "scripts/verify_explainability_provenance.py", "--strict", "--json"]),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit explainability provenance.")
    parser.add_argument("--json", action="store_true", help="Machine-readable JSON output.")
    args = parser.parse_args()

    results = {}
    ok = True
    for name, cmd in CHECKERS:
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())
        try:
            data = json.loads(proc.stdout)
        except (json.JSONDecodeError, ValueError):
            data = {"ok": proc.returncode == 0, "messages": [proc.stderr or proc.stdout]}
        results[name] = {"ok": data.get("ok", proc.returncode == 0), "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip()}
        if not results[name]["ok"]:
            ok = False

    if args.json:
        print(json.dumps({"ok": ok, "results": results}, indent=2))
    else:
        for name, r in results.items():
            print(f"{'OK  ' if r['ok'] else 'FAIL'} {name}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())