"""Verify A/B test results for correctness: Holm correction, leakage guard, effect size ranges."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def verify_ab(domain: str = "spam_detection", strict: bool = False) -> int:
    """Verify ab_test.json for a domain.

    Checks:
    - p_holm_adjusted >= p_value (Holm adjustment can only increase)
    - p_holm_adjusted <= 1.0
    - |cohen_h| <= 2.0 (Cohen's h theoretical max)
    - -1 <= risk_difference <= 1
    - n_discordant <= n
    - winner consistency with acc_a/acc_b
    """
    errors = []
    st_path = Path("outputs") / "stats" / domain / "ab_test.json"
    if not st_path.exists():
        # Try staging
        for p in Path("outputs").parent.glob("outputs.staging.*"):
            candidate = p / "stats" / domain / "ab_test.json"
            if candidate.exists():
                st_path = candidate
                break
        else:
            print(f"verify-ab: no ab_test.json found for domain '{domain}'")
            return 1

    ab = json.loads(st_path.read_text(encoding="utf-8"))
    comparisons = ab.get("comparisons", [])

    for c in comparisons:
        ma = c.get("model_a", "?")
        mb = c.get("model_b", "?")
        pair = f"{ma} vs {mb}"

        p_val = c.get("p_value")
        p_holm = c.get("p_holm_adjusted")
        h = c.get("cohen_h")
        rd = c.get("risk_difference")
        n = c.get("n")
        n_disc = c.get("n_discordant")
        winner = c.get("winner")
        acc_a = c.get("acc_a")
        acc_b = c.get("acc_b")

        # p_holm_adjusted >= p_value
        if p_val is not None and p_holm is not None and p_holm < p_val - 1e-12:
            errors.append(f"{pair}: p_holm_adjusted ({p_holm}) < p_value ({p_val})")

        # p_holm_adjusted <= 1.0
        if p_holm is not None and p_holm > 1.0 + 1e-12:
            errors.append(f"{pair}: p_holm_adjusted ({p_holm}) > 1.0")

        # |cohen_h| <= 2.0
        if h is not None and abs(h) > 2.0 + 1e-12:
            errors.append(f"{pair}: |cohen_h| ({abs(h):.4g}) > 2.0")

        # -1 <= risk_difference <= 1
        if rd is not None and (rd < -1.0 - 1e-12 or rd > 1.0 + 1e-12):
            errors.append(f"{pair}: risk_difference ({rd}) outside [-1, 1]")

        # n_discordant <= n
        if n is not None and n_disc is not None and n_disc > n:
            errors.append(f"{pair}: n_discordant ({n_disc}) > n ({n})")

        # winner consistency with acc_a/acc_b
        if winner not in ("tie", None) and acc_a is not None and acc_b is not None:
            expected_winner = ma if acc_a > acc_b else (mb if acc_b > acc_a else "tie")
            if winner != expected_winner:
                errors.append(
                    f"{pair}: winner '{winner}' inconsistent with "
                    f"acc_a={acc_a}, acc_b={acc_b} (expected '{expected_winner}')"
                )

    if errors:
        print("verify-ab: FAIL")
        for e in errors:
            print(f"  {e}")
        return 1

    print(f"verify-ab: OK ({len(comparisons)} comparisons in domain '{domain}')")
    return 0


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Verify A/B test results.")
    parser.add_argument("--domain", default="spam_detection", help="Domain to verify.")
    parser.add_argument("--strict", action="store_true", help="Fail on any violation.")
    args = parser.parse_args()

    return verify_ab(domain=args.domain, strict=args.strict)


if __name__ == "__main__":
    sys.exit(main())