"""Inject A/B test summary into SPAM_DETECTION.md and FRAUD_DETECTION.md.

Reads `outputs/stats/<domain>/ab_test.json` and writes a
'## Statistical Significance' section into the corresponding
domain doc file, between markers:

    <!-- ab-summary:start -->
    ... content ...
    <!-- ab-summary:end -->

Idempotent: re-running replaces the block between markers.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


DOMAIN_DOC_MAP = {
    "spam_detection": "docs/SPAM_DETECTION.md",
    "fraud_detection": "docs/FRAUD_DETECTION.md",
}


def _fmt_p(p: float) -> str:
    if p < 0.001:
        return "<0.001"
    return f"{p:.4g}"


def _build_summary_md(ab: dict) -> str:
    """Build the markdown block to inject."""
    comparisons = ab.get("comparisons", [])
    config = ab.get("config", {})

    if not comparisons:
        return (
            "\n## Statistical Significance\n"
            "\nNo A/B comparisons available. Run `make stats` first.\n"
            "\n"
        )

    n_comp = len(comparisons)
    n_sig = sum(1 for c in comparisons if c.get("significant_holm_0.05"))
    n_sig_01 = sum(1 for c in comparisons if c.get("significant_holm_0.01"))

    lines = [
        "## Statistical Significance",
        "",
        "Pairwise McNemar tests on test-split predictions, with Holm "
        "family-wise error rate correction.",
        "",
        f"- Comparisons: {n_comp}",
        f"- Significant (Holm 0.05): {n_sig}",
        f"- Significant (Holm 0.01): {n_sig_01}",
        "",
        "### Pairwise comparisons",
        "",
        "| Model A | Model B | Method | p-value | Holm p | Significant | Effect Size (h) | Winner |",
        "|---------|---------|--------|---------|--------|-------------|-----------------|--------|",
    ]
    for c in comparisons:
        ma = c["model_a"]
        mb = c["model_b"]
        p = c["p_value"]
        ph = c["p_holm_adjusted"]
        sig = "✓" if c.get("significant_holm_0.05") else "✗"
        winner = c.get("winner", "tie")
        h = c.get("cohen_h", 0.0)
        lines.append(
            f"| {ma} | {mb} | {c.get('method', 'mcnemar')} | "
            f"{_fmt_p(p)} | {_fmt_p(ph)} | {sig} | {h:+.3f} | {winner} |"
        )

    lines.append("")
    # Best winner
    winners = [c for c in comparisons if c.get("winner_significant")]
    if winners:
        best = min(winners, key=lambda c: c["p_holm_adjusted"])
        lines.append(
            f"### Winner\n"
            f"{best['model_a']} vs {best['model_b']} — "
            f"p={_fmt_p(best['p_value'])}, Holm p={_fmt_p(best['p_holm_adjusted'])}"
        )
    else:
        lines.append("### Winner\nNo significant winner at Holm 0.05.")

    lines.append("")
    return "\n".join(lines) + "\n"


def inject_stats(domain: str) -> bool:
    """Inject A/B summary into the domain doc."""
    doc_path = Path(DOMAIN_DOC_MAP[domain])
    ab_path = Path("outputs/stats") / domain / "ab_test.json"

    if not ab_path.exists():
        # Try staging
        for p in Path("outputs").parent.glob("outputs.staging.*"):
            candidate = p / "stats" / domain / "ab_test.json"
            if candidate.exists():
                ab_path = candidate
                break
        else:
            print(f"inject-stats: no ab_test.json found for domain '{domain}'")
            return False

    ab = json.loads(ab_path.read_text(encoding="utf-8"))
    block = _build_summary_md(ab)

    doc_path = Path(DOMAIN_DOC_MAP[domain])
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    if not doc_path.exists():
        doc_path.write_text(f"# {domain.replace('_', ' ').title()}\n\n", encoding="utf-8")

    doc_text = doc_path.read_text(encoding="utf-8")
    start_marker = "<!-- ab-summary:start -->"
    end_marker = "<!-- ab-summary:end -->"

    if start_marker in doc_text and end_marker in doc_text:
        # Replace existing block
        before = doc_text.split(start_marker)[0]
        after = doc_text.split(end_marker)[1]
        new_text = (
            before
            + start_marker + "\n"
            + block
            + end_marker
            + after
        )
    else:
        # Append at end
        new_text = doc_text.rstrip() + "\n\n" + start_marker + "\n" + block + end_marker + "\n"

    doc_path.write_text(new_text, encoding="utf-8")
    n_comp = len(ab.get("comparisons", []))
    print(f"inject-stats: wrote {doc_path} ({n_comp} comparisons)")
    return True


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Inject A/B test summary into domain docs.")
    parser.add_argument("--domain", default="spam_detection", help="Domain to inject into.")
    args = parser.parse_args()

    return 0 if inject_stats(args.domain) else 1


if __name__ == "__main__":
    sys.exit(main())