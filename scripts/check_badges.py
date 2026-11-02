"""Check badge URLs in README (skip DOI placeholder)."""
import re
import sys
from pathlib import Path

import requests


def main() -> int:
    text = Path("README.md").read_text()
    urls = re.findall(r"https?://[^\s\)]+", text)
    bad = []
    for u in urls:
        if "XXXXXXX" in u:
            continue
        try:
            r = requests.head(u, timeout=10, allow_redirects=True)
            if r.status_code >= 400:
                bad.append((u, r.status_code))
        except Exception as e:
            bad.append((u, str(e)))
    if bad:
        for u, s in bad[:20]:
            print("BAD", s, u)
        return 1
    print("badges-check: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())