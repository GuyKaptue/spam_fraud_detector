"""Compare two directory trees for reproducibility.

Compares files by SHA-256, ignoring unstable fields (git_commit, created_at)
and files matching ignore patterns. Does not use diff -r.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

DEFAULT_IGNORES = [
    ".DS_Store",
    "__pycache__/**",
    "*.pyc",
    "outputs/logs/**",
]

DEFAULT_IGNORE_KEYS = {
    "CONTRACT.json": {"git_commit"},
    "meta.json": {"created_at"},
    "split.json": {"created_at"},
}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _should_ignore(path: str, ignores: list[str]) -> bool:
    import fnmatch
    for pattern in ignores:
        if fnmatch.fnmatch(path, pattern):
            return True
        if fnmatch.fnmatch(path, pattern.lstrip("/")):
            return True
    return False


def _normalize_json(path: Path, content: bytes) -> bytes:
    """Drop unstable keys from JSON files."""
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return content
    name = path.name
    ignore_keys = set()
    for key_pattern, keys in DEFAULT_IGNORE_KEYS.items():
        if name == key_pattern or name.endswith(key_pattern):
            ignore_keys |= keys
    if not ignore_keys:
        return content
    def _strip(obj):
        if isinstance(obj, dict):
            return {k: _strip(v) for k, v in obj.items() if k not in ignore_keys}
        if isinstance(obj, list):
            return [_strip(x) for x in obj]
        return obj
    stripped = _strip(data)
    return (json.dumps(stripped, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _collect_files(root: Path) -> dict[str, Path]:
    result = {}
    if not root.exists():
        return result
    for p in root.rglob("*"):
        if p.is_file():
            rel = str(p.relative_to(root))
            result[rel] = p
    return result


def compare(a: Path, b: Path, ignores: list[str] | None = None) -> dict:
    if ignores is None:
        ignores = DEFAULT_IGNORES[:]
    files_a = _collect_files(a)
    files_b = _collect_files(b)

    diffs = []
    all_paths = sorted(set(files_a) | set(files_b))

    for rel in all_paths:
        if _should_ignore(rel, ignores):
            continue
        in_a = rel in files_a
        in_b = rel in files_b
        if in_a and not in_b:
            diffs.append({"path": rel, "reason": "only_in_a"})
            continue
        if in_b and not in_a:
            diffs.append({"path": rel, "reason": "only_in_b"})
            continue
        # both exist
        bytes_a = files_a[rel].read_bytes()
        bytes_b = files_b[rel].read_bytes()
        if bytes_a == bytes_b:
            continue
        # try JSON normalization
        norm_a = _normalize_json(files_a[rel], bytes_a)
        norm_b = _normalize_json(files_b[rel], bytes_b)
        if norm_a == norm_b:
            continue
        # check if JSON key differs
        try:
            ja = json.loads(bytes_a)
            jb = json.loads(bytes_b)
            # find differing keys
            if isinstance(ja, dict) and isinstance(jb, dict):
                for key in set(ja) | set(jb):
                    if ja.get(key) != jb.get(key):
                        diffs.append({"path": rel, "reason": "json_key_differ", "key": key})
                        break
                else:
                    diffs.append({"path": rel, "reason": "bytes_differ"})
            else:
                diffs.append({"path": rel, "reason": "bytes_differ"})
        except (json.JSONDecodeError, UnicodeDecodeError):
            diffs.append({"path": rel, "reason": "bytes_differ"})

    return {"ok": len(diffs) == 0, "differences": diffs}


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two directory trees for reproducibility.")
    parser.add_argument("--a", type=Path, default=Path("outputs.a"), help="First directory.")
    parser.add_argument("--b", type=Path, default=Path("outputs.b"), help="Second directory.")
    parser.add_argument("--ignore", action="append", default=[], help="Ignore pattern (glob).")
    parser.add_argument("--json", action="store_true", help="Machine-readable JSON output.")
    args = parser.parse_args()

    ignores = DEFAULT_IGNORES + args.ignore
    result = compare(args.a, args.b, ignores)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        if result["ok"]:
            print("reproduce-diff: OK")
        else:
            for d in result["differences"]:
                print(f"DIFF {d['path']}: {d['reason']}" + (f" ({d.get('key', '')})" if d.get("key") else ""))

    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())