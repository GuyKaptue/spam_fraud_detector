"""Statistical testing between top models using persisted predictions.

Implements the v4.5 A/B test contract:
- Pairwise McNemar on test split predictions
- Automatic fallback to exact McNemar when n_discordant < min_discordant_chi2
- Holm correction with deterministic tie-break (p_value, model_a, model_b)
- Effect size: Cohen's h and risk difference
- Leakage guard: row_id disjointness across train/val/test splits
- Zero-discordant skip with warning
- Full v2.0 schema: config block, method field, n01/n10, family_rank,
  significant_holm_0.05/0.01, winner_significant
- Sequential pairwise tournament (champion-challenger): model[0] vs model[1],
  winner vs model[2], etc., until all models evaluated; final winner is overall
  champion; detailed round-by-round documentation
"""
from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from scipy import stats as sp_stats

from .result_store import ResultStore

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_DISCORDANT_CHI2 = 25
ALPHA_FAMILY = 0.05


# ---------------------------------------------------------------------------
# Effect size helpers
# ---------------------------------------------------------------------------

def cohen_h(p1: float, p2: float) -> float:
    """Cohen's h effect size for two proportions.

    h = 2 * arcsin(sqrt(p1)) - 2 * arcsin(sqrt(p2))

    Interpretation: |h| = 0.2 small, 0.5 medium, 0.8 large.
    Range: [-pi, pi] ≈ [-3.14, 3.14].
    """
    p1 = max(0.0, min(1.0, p1))
    p2 = max(0.0, min(1.0, p2))
    return 2.0 * math.asin(math.sqrt(p1)) - 2.0 * math.asin(math.sqrt(p2))


def risk_difference(p1: float, p2: float) -> float:
    """Absolute risk difference: p1 - p2."""
    return p1 - p2


# ---------------------------------------------------------------------------
# McNemar
# ---------------------------------------------------------------------------

def _mcnemar(n01: int, n10: int, n_both_right: int, n_both_wrong: int) -> tuple[float, float, str]:
    """Run McNemar test on discordant pairs.

    Returns (statistic, p_value, method).

    When n_discordant = n01 + n10 < 25, uses exact binomial test.
    Otherwise uses chi-square with continuity correction.
    """
    n_discordant = n01 + n10
    if n_discordant == 0:
        return 0.0, 1.0, "chi2_fallback"

    try:
        from statsmodels.stats.contingency_tables import mcnemar

        table = [[n_both_right, n10], [n01, n_both_wrong]]
        if n_discordant < MIN_DISCORDANT_CHI2:
            res = mcnemar(table, exact=True, correction=False)
            method = "mcnemar_exact"
        else:
            res = mcnemar(table, exact=False, correction=True)
            method = "mcnemar_chi2_corrected"
        return float(res.statistic), float(res.pvalue), method
    except Exception:
        # Fallback chi-square
        stat = float((n01 - n10) ** 2 / n_discordant) if n_discordant > 0 else 0.0
        pval = float(sp_stats.chi2.sf(stat, 1)) if n_discordant > 0 else 1.0
        return stat, pval, "chi2_fallback"


# ---------------------------------------------------------------------------
# Holm correction (deterministic)
# ---------------------------------------------------------------------------

def _holm_adjust(p_values: List[float]) -> List[float]:
    """Holm step-down adjustment.

    Deterministic tie-break: sort by (p_value, model_a, model_b).
    Adjusted p-value for rank r (1-indexed): min(1, p * (m - r + 1)).
    """
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: (p_values[i], i))
    adjusted = [0.0] * m
    for rank, idx in enumerate(order, start=1):
        adjusted[idx] = min(1.0, p_values[idx] * (m - rank + 1))
    return adjusted


class StatsEngine:
    def __init__(self, out_root, domain: str) -> None:
        self.out_root = Path(out_root)
        self.domain = domain

    def _staging(self) -> Path:
        if (self.out_root / "dashboard").exists():
            return self.out_root
        for p in self.out_root.parent.glob("outputs.staging.*"):
            if (p / "dashboard").exists():
                return p
        return self.out_root

    def _load_preds(self, model, group):
        st = self._staging()
        path = st / "dashboard" / "fact_predictions" / f"{self.domain}__{model}__{group}.parquet"
        if not path.exists():
            # try predictions dir
            path = st / "predictions" / self.domain / "all" / f"{model}__{group}.parquet"
        if not path.exists():
            return None
        return pd.read_parquet(path)

    @staticmethod
    def _check_leakage(pa: pd.DataFrame, pb: pd.DataFrame, model_a: str, model_b: str) -> None:
        """Raise SplitContractError if train/val/test splits overlap between models.

        A/B tests are only valid if the same rows are not used for both training
        and evaluation across compared models. We check row_id disjointness
        across splits within each model's predictions.
        """
        from .exceptions import SplitContractError

        for label, p in ((model_a, pa), (model_b, pb)):
            if p is None:
                continue
            splits = p["split"].astype(str).unique()
            row_ids_by_split = {}
            for s in splits:
                row_ids_by_split[s] = set(p[p["split"].astype(str) == s]["row_id"].values)

            # Check train/val/test are disjoint
            all_splits = list(row_ids_by_split.keys())
            for i in range(len(all_splits)):
                for j in range(i + 1, len(all_splits)):
                    overlap = row_ids_by_split[all_splits[i]] & row_ids_by_split[all_splits[j]]
                    if overlap:
                        raise SplitContractError(
                            f"Leakage detected for {label}: split '{all_splits[i]}' and "
                            f"split '{all_splits[j]}' share {len(overlap)} row_ids. "
                            f"A/B test requires non-overlapping splits."
                        )

    def compute(self, store: Optional[ResultStore] = None) -> Dict[str, Any]:
        st = self._staging()
        # determine top models from selection
        sel_path = st / "models" / self.domain / "top3" / "selection.json"
        selection = json.loads(sel_path.read_text()) if sel_path.exists() else {"ranking": []}
        ranking = selection.get("ranking", [])
        models = [(r["model"], r["group"]) for r in ranking]
        # fallback: top 3 by pr_auc from fact_metrics
        if not models:
            fm = pd.read_parquet(st / "dashboard" / "fact_metrics.parquet")
            ftest = fm[(fm["domain"] == self.domain) & (fm["split"] == "test") & (fm["metric"] == "pr_auc")]
            agg = ftest.groupby(["model", "group"], as_index=False)["value"].mean().sort_values("value", ascending=False)
            models = list(zip(agg["model"].head(3), agg["group"].head(3)))

        comparisons = []
        # pairwise McNemar between top models on test split
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                ma, ga = models[i]
                mb, gb = models[j]
                pa = self._load_preds(ma, ga)
                pb = self._load_preds(mb, gb)
                if pa is None or pb is None:
                    continue
                # align by row_id within test split
                ta = pa[pa["split"] == "test"].sort_values("row_id")
                tb = pb[pb["split"] == "test"].sort_values("row_id")
                if len(ta) == 0 or len(ta) != len(tb):
                    continue
                # Leakage guard: ensure train/val/test splits don't overlap
                self._check_leakage(ta, tb, ma, mb)
                # discordant counts
                a_wrong = (ta["y_pred"].values != ta["y_true"].values)
                b_wrong = (tb["y_pred"].values != tb["y_true"].values)
                n_both_right = int((~a_wrong & ~b_wrong).sum())
                n_both_wrong = int((a_wrong & b_wrong).sum())
                n01 = int((a_wrong & ~b_wrong).sum())  # a wrong, b right
                n10 = int((~a_wrong & b_wrong).sum())  # a right, b wrong
                n_discordant = n01 + n10
                n_total = int(len(ta))

                # Zero-discordant skip
                if n_discordant == 0:
                    warnings.warn(
                        f"Zero discordant pairs for {ma} vs {mb}; skipping McNemar",
                        stacklevel=1,
                    )
                    continue

                # McNemar
                stat, pval, method = _mcnemar(n01, n10, n_both_right, n_both_wrong)

                # Effect sizes
                acc_a = float((~a_wrong).mean())
                acc_b = float((~b_wrong).mean())
                h = cohen_h(acc_a, acc_b)
                rd = risk_difference(acc_a, acc_b)

                winner = ma if n10 > n01 else (mb if n01 > n10 else "tie")

                comparisons.append({
                    "model_a": ma,
                    "model_b": mb,
                    "method": method,
                    "statistic": round(stat, 6),
                    "p_value": round(pval, 10),
                    "n": n_total,
                    "n_discordant": n_discordant,
                    "n01": n01,
                    "n10": n10,
                    "winner": winner,
                    "acc_a": round(acc_a, 6),
                    "acc_b": round(acc_b, 6),
                    "cohen_h": round(float(h), 6),
                    "risk_difference": round(float(rd), 6),
                })

        # Holm correction with deterministic tie-break
        if comparisons:
            p_values = [c["p_value"] for c in comparisons]
            model_as = [c["model_a"] for c in comparisons]
            model_bs = [c["model_b"] for c in comparisons]
            order = sorted(range(len(p_values)), key=lambda i: (p_values[i], model_as[i], model_bs[i]))
            m = len(p_values)
            adjusted = [0.0] * m
            for rank, idx in enumerate(order, start=1):
                adjusted[idx] = min(1.0, p_values[idx] * (m - rank + 1))
            for i, c in enumerate(comparisons):
                c["p_holm_adjusted"] = float(adjusted[i])
                c["family_rank"] = order.index(i) + 1
                c["significant_holm_0.05"] = bool(c["p_holm_adjusted"] <= 0.05)
                c["significant_holm_0.01"] = bool(c["p_holm_adjusted"] <= 0.01)
                c["winner_significant"] = bool(
                    c["p_holm_adjusted"] <= 0.05 and c["winner"] != "tie"
                )

        # Config block
        config = {
            "method_default": "mcnemar_chi2_corrected",
            "family_size": len(comparisons),
            "alpha_family": ALPHA_FAMILY,
            "min_discordant_chi2": MIN_DISCORDANT_CHI2,
        }

        ab = {
            "domain": self.domain,
            "config": config,
            "comparisons": comparisons,
        }

        summary_lines = [f"# Statistical Summary — {self.domain}", ""]
        if comparisons:
            summary_lines.append("## Family")
            summary_lines.append(
                f"- Comparisons: {len(comparisons)}"
            )
            summary_lines.append(
                f"- Significant (Holm 0.05): {sum(1 for c in comparisons if c['significant_holm_0.05'])}"
            )
            summary_lines.append(
                f"- Significant (Holm 0.01): {sum(1 for c in comparisons if c['significant_holm_0.01'])}"
            )
            summary_lines.append("")
            summary_lines.append("## Pairwise")
            for c in comparisons:
                summary_lines.append(
                    f"- {c['model_a']} vs {c['model_b']}: "
                    f"p={c['p_value']:.4g}, holm={c['p_holm_adjusted']:.4g}, "
                    f"method={c['method']}, winner={c['winner']}"
                )
            summary_lines.append("")
            summary_lines.append("## Winner")
            winners = [c for c in comparisons if c["winner_significant"]]
            if winners:
                # Best winner = lowest p_holm_adjusted
                best = min(winners, key=lambda c: c["p_holm_adjusted"])
                summary_lines.append(
                    f"- Best: {best['model_a']} vs {best['model_b']} "
                    f"(p={best['p_value']:.4g}, holm={best['p_holm_adjusted']:.4g})"
                )
            else:
                summary_lines.append("- No significant winner at Holm 0.05")
        else:
            summary_lines.append("## Pairwise")
            summary_lines.append("- No comparisons available")
            summary_lines.append("")
            summary_lines.append("## Winner")
            summary_lines.append("- No significant winner at Holm 0.05")
        summary_md = "\n".join(summary_lines) + "\n"

        if store is not None:
            store.write_stats(ab, summary_md)
        else:
            st2 = self._staging()
            stats_dir = st2 / "stats" / self.domain
            stats_dir.mkdir(parents=True, exist_ok=True)
            (stats_dir / "ab_test.json").write_text(json.dumps(ab, indent=2))
            (stats_dir / "summary.md").write_text(summary_md)
        return ab

    # ------------------------------------------------------------------ #
    # Sequential pairwise tournament (champion-challenger)
    # ------------------------------------------------------------------ #

    def compute_tournament(
        self,
        models: Optional[List[tuple[str, str]]] = None,
        store: Optional[ResultStore] = None,
    ) -> Dict[str, Any]:
        """Run sequential pairwise A/B testing (tournament/challenger format).

        The first two models compete head-to-head; the winner advances to face
        the next model; this continues until ALL supervised models have been
        evaluated. The final surviving model is the overall champion.

        Each round performs a full pairwise McNemar comparison on the test
        split with full statistics (p-value, Holm-corrected p-value, effect
        sizes, accuracy, etc.). Rounds are recorded in order and can be
        rendered as a markdown table.

        Parameters
        ----------
        models : list of (model_name, group) tuples, optional
            Ordered list of models to compare. If None, ALL supervised model
            bundles are discovered from the staging directory (sorted by
            PR-AUC descending). Anomaly models are excluded.
        store : Optional[ResultStore]
            ResultStore instance for persisting tournament stats.

        Returns
        -------
        dict with keys:
          - domain : domain name
          - config : tournament config (method, alpha_family, n_models)
          - rounds : list of round dicts, each containing:
              - round : round number (1-indexed)
              - model_a : current champion model name
              - group_a : current champion model group
              - model_b : challenger model name
              - group_b : challenger model group
              - winner : winning model name
              - winner_group : winning model group
              - eliminated : eliminated model name
              - advance_to_next : whether winner advances
              - method : McNemar method used
              - statistic : McNemar chi-square statistic
              - p_value : McNemar p-value
              - significant : whether p < alpha
              - acc_a / acc_b : accuracies of both models
              - cohen_h / risk_difference : effect sizes
              - n01 / n10 / n_discordant : discordant pair counts
          - champion : final winning model name
          - champion_group : final winning model group
          - summary_md : markdown summary text
        """
        st = self._staging()
        import json

        # ---------------------------------------------------------------
        # Model discovery: ALL supervised models (not just top-3)
        # ---------------------------------------------------------------
        if models is None:
            # Discover all supervised model bundles from staging
            bundle_dir = st / "models" / self.domain / "all"
            anomaly_groups = {"anomaly"}
            discovered: List[tuple[str, str]] = []
            if bundle_dir.exists():
                for meta_path in sorted(bundle_dir.glob("*.meta.json")):
                    try:
                        meta = json.loads(meta_path.read_text())
                        group = meta.get("group", "")
                        name = meta.get("model_name") or meta.get("model", "")
                        if group in anomaly_groups:
                            continue  # anomaly models excluded from supervised tournament
                        if name and (name, group) not in discovered:
                            discovered.append((name, group))
                    except Exception:
                        continue
            models = discovered

            # Fallback: selection.json or fact_metrics
            if not models:
                sel_path = st / "models" / self.domain / "top3" / "selection.json"
                selection = json.loads(sel_path.read_text()) if sel_path.exists() else {"ranking": []}
                ranking = selection.get("ranking", [])
                models = [(r["model"], r["group"]) for r in ranking]
            if not models:
                try:
                    fm = pd.read_parquet(st / "dashboard" / "fact_metrics.parquet")
                    ftest = fm[(fm["domain"] == self.domain) & (fm["split"] == "test") & (fm["metric"] == "pr_auc")]
                    agg = ftest.groupby(["model", "group"], as_index=False)["value"].mean().sort_values("value", ascending=False)
                    models = list(zip(agg["model"].head(20), agg["group"].head(20)))
                except Exception:
                    models = []

        if len(models) < 2:
            raise ValueError(
                f"Tournament requires at least 2 models, got {len(models)}: {models}"
            )

        # Print initial model roster for transparency
        print(f"\n{'='*70}")
        print(f"TOURNAMENT ROUND-ROBIN: {self.domain.upper()}")
        print(f"{'='*70}")
        print(f"Total supervised models in tournament: {len(models)}")
        for idx, (mname, mgroup) in enumerate(models, 1):
            print(f"  {idx:2d}. {mname} ({mgroup})")
        n_comparisons = len(models) - 1
        alpha_bonferroni = ALPHA_FAMILY / n_comparisons if n_comparisons > 0 else ALPHA_FAMILY
        print(f"Multiple comparison correction: Bonferroni alpha = {alpha_bonferroni:.6f} "
              f"(alpha={ALPHA_FAMILY} / {n_comparisons} comparisons)")
        print(f"{'='*70}\n")

        rounds: List[Dict[str, Any]] = []
        current_champion_name, current_champion_group = models[0]

        print(
            f"[tournament] Opening champion: {current_champion_name} "
            f"({current_champion_group}) — first model in roster."
        )

        for i in range(1, len(models)):
            challenger_name, challenger_group = models[i]
            round_num = i  # Round 1 = model[0] vs model[1], etc.

            print(f"\n{'─'*70}")
            print(f"[tournament] ROUND {round_num}/{len(models)-1}")
            print(f"{'─'*70}")
            print(
                f"  Champion (A): {current_champion_name} ({current_champion_group})"
            )
            print(
                f"  Challenger (B): {challenger_name} ({challenger_group})"
            )

            pa = self._load_preds(current_champion_name, current_champion_group)
            pb = self._load_preds(challenger_name, challenger_group)

            if pa is None or pb is None:
                reason = "missing predictions file"
                missing = []
                if pa is None:
                    missing.append(current_champion_name)
                if pb is None:
                    missing.append(challenger_name)
                print(
                    f"  [RESULT] FAILED — {reason}: "
                    f"{', '.join(missing)} have no persisted predictions"
                )
                raise ValueError(
                    f"Tournament round {round_num}: {current_champion_name} vs "
                    f"{challenger_name} — {reason}"
                )

            # Filter to test split only
            ta_all = pa[pa["split"] == "test"]
            tb_all = pb[pb["split"] == "test"]

            if len(ta_all) == 0 or len(tb_all) == 0:
                reason = "empty test predictions"
                print(
                    f"  [RESULT] FAILED — {reason}: "
                    f"{current_champion_name} has {len(ta_all)} test rows, "
                    f"{challenger_name} has {len(tb_all)} test rows"
                )
                raise ValueError(
                    f"Tournament round {round_num}: {current_champion_name} vs "
                    f"{challenger_name} — {reason}"
                )

            # Align by actual row_id (dataset row identifier), not positional index
            common_row_ids = set(ta_all["row_id"]) & set(tb_all["row_id"])
            n_common = len(common_row_ids)
            n_a = len(ta_all)
            n_b = len(tb_all)

            if n_common == 0:
                reason = "no common row_ids between models"
                print(
                    f"  [RESULT] FAILED — {reason}: "
                    f"{current_champion_name} has {n_a} rows, "
                    f"{challenger_name} has {n_b} rows, zero overlap"
                )
                raise ValueError(
                    f"Tournament round {round_num}: {current_champion_name} vs "
                    f"{challenger_name} — {reason}. "
                    f"This indicates different test splits were used."
                )

            if n_common < n_a or n_common < n_b:
                print(
                    f"  [WARNING] Partial row overlap: {n_common}/{n_a} rows for "
                    f"{current_champion_name}, {n_common}/{n_b} rows for "
                    f"{challenger_name}. Using intersection."
                )

            ta = ta_all[ta_all["row_id"].isin(common_row_ids)].sort_values("row_id")
            tb = tb_all[tb_all["row_id"].isin(common_row_ids)].sort_values("row_id")

            # Verify alignment
            if not (ta["row_id"].values == tb["row_id"].values).all():
                raise ValueError(
                    f"Tournament round {round_num}: row_id alignment failed "
                    f"for {current_champion_name} vs {challenger_name}"
                )

            # Leakage guard
            self._check_leakage(ta, tb, current_champion_name, challenger_name)

            # Discordant counts
            a_wrong = (ta["y_pred"].values != ta["y_true"].values)
            b_wrong = (tb["y_pred"].values != tb["y_true"].values)
            n_both_right = int((~a_wrong & ~b_wrong).sum())
            n_both_wrong = int((a_wrong & b_wrong).sum())
            n01 = int((a_wrong & ~b_wrong).sum())  # a wrong, b right
            n10 = int((~a_wrong & b_wrong).sum())  # a right, b wrong
            n_discordant = n01 + n10
            n_total = int(len(ta))

            print(f"  [DATA] Test rows compared: {n_total}")
            print(f"  [DATA] Both correct: {n_both_right}, Both wrong: {n_both_wrong}")
            print(f"  [DATA] Discordant: n01={n01} ({current_champion_name} wrong, {challenger_name} right), "
                  f"n10={n10} ({current_champion_name} right, {challenger_name} wrong)")

            # Zero-discordant skip
            if n_discordant == 0:
                print(
                    f"  [RESULT] SKIPPED — zero discordant pairs; "
                    f"treating as tie, {current_champion_name} advances."
                )
                current_champion_name, current_champion_group = challenger_name, challenger_group
                rounds.append({
                    "round": round_num,
                    "model_a": current_champion_name,
                    "group_a": current_champion_group,
                    "model_b": challenger_name,
                    "group_b": challenger_group,
                    "winner": current_champion_name,
                    "winner_group": current_champion_group,
                    "eliminated": challenger_name,
                    "advance_to_next": True,
                    "skipped": True,
                    "reason": "zero discordant pairs",
                })
                continue

            # McNemar test
            stat, pval, method = _mcnemar(n01, n10, n_both_right, n_both_wrong)

            # Effect sizes and accuracy
            acc_a = float((~a_wrong).mean())
            acc_b = float((~b_wrong).mean())
            h = cohen_h(acc_a, acc_b)
            rd = risk_difference(acc_a, acc_b)

            # Winner: higher accuracy wins; ties go to current champion
            if acc_b > acc_a:
                winner = challenger_name
                winner_group = challenger_group
                eliminated = current_champion_name
                win_reason = f"higher accuracy ({acc_b:.4f} > {acc_a:.4f})"
            elif acc_a > acc_b:
                winner = current_champion_name
                winner_group = current_champion_group
                eliminated = challenger_name
                win_reason = f"higher accuracy ({acc_a:.4f} > {acc_b:.4f})"
            else:
                # Exact tie on accuracy: current champion retains title
                winner = current_champion_name
                winner_group = current_champion_group
                eliminated = challenger_name
                win_reason = f"exact tie ({acc_a:.4f} == {acc_b:.4f}); champion retains title"

            # Significance checks
            significant = pval <= ALPHA_FAMILY
            bonferroni_significant = pval <= alpha_bonferroni
            sig_str = "SIGNIFICANT" if significant else "not significant"
            bonf_str = "BONF-SIGNIFICANT" if bonferroni_significant else "bonf-not-sig"

            round_result = {
                "round": round_num,
                "model_a": current_champion_name,
                "group_a": current_champion_group,
                "model_b": challenger_name,
                "group_b": challenger_group,
                "winner": winner,
                "winner_group": winner_group,
                "eliminated": eliminated,
                "advance_to_next": True,
                "skipped": False,
                "method": method,
                "statistic": round(stat, 6),
                "p_value": round(pval, 10),
                "n": n_total,
                "n_discordant": n_discordant,
                "n01": n01,
                "n10": n10,
                "acc_a": round(acc_a, 6),
                "acc_b": round(acc_b, 6),
                "cohen_h": round(float(h), 6),
                "risk_difference": round(float(rd), 6),
                "significant": significant,
                "bonferroni_significant": bonferroni_significant,
                "alpha_bonferroni": round(alpha_bonferroni, 6),
                "threshold": ALPHA_FAMILY,
                "win_reason": win_reason,
            }
            rounds.append(round_result)

            print(f"  [STATS] McNemar {method}: statistic={stat:.4f}, p-value={pval:.4g}")
            print(f"  [STATS] Cohen's h={float(h):+.4f}, Risk difference={float(rd):+.4f}")
            print(f"  [STATS] p < {ALPHA_FAMILY}: {significant} ({sig_str})")
            print(f"  [STATS] p < Bonferroni ({alpha_bonferroni:.6f}): {bonferroni_significant} ({bonf_str})")
            print(f"  [RESULT] WINNER: {winner} ({winner_group}) — {win_reason}")
            print(f"  [RESULT] {winner} advances to next round as champion.")

            # Winner advances to next round
            current_champion_name, current_champion_group = winner, winner_group

        tournament = {
            "domain": self.domain,
            "config": {
                "method_default": "mcnemar_chi2_corrected",
                "alpha_family": ALPHA_FAMILY,
                "alpha_bonferroni": round(alpha_bonferroni, 6),
                "n_models": len(models),
                "n_comparisons": n_comparisons,
                "format": "sequential_pairwise_champion_challenger",
            },
            "rounds": rounds,
            "champion": current_champion_name,
            "champion_group": current_champion_group,
        }

        # Build markdown summary
        summary_lines = [f"# Sequential Pairwise A/B Tournament — {self.domain}", ""]
        summary_lines.append(
            f"- Total models: {len(models)}"
        )
        summary_lines.append(
            "- Format: Sequential pairwise (champion-challenger), ALL supervised models included"
        )
        summary_lines.append(
            f"- Method: McNemar test (chi-square corrected, exact fallback if n_discordant < {MIN_DISCORDANT_CHI2})"
        )
        summary_lines.append(
            f"- Alpha: {ALPHA_FAMILY}"
        )
        summary_lines.append(
            f"- Bonferroni alpha: {alpha_bonferroni:.6f} ({ALPHA_FAMILY} / {n_comparisons} comparisons)"
        )
        summary_lines.append("")
        summary_lines.append("## Model Roster")
        summary_lines.append("")
        for idx, (mname, mgroup) in enumerate(models, 1):
            summary_lines.append(f"{idx}. {mname} ({mgroup})")
        summary_lines.append("")
        summary_lines.append("## Round-by-Round Results")
        summary_lines.append("")
        summary_lines.append(
            "| Round | Champion (A) | Group | Challenger (B) | Group | "
            "Acc A | Acc B | p-value | Significant | Bonferroni | Winner | Why |"
        )
        summary_lines.append(
            "|-------|--------------|-------|----------------|-------|-------|-------|---------|-----------|------------|--------|-----|"
        )
        for r in rounds:
            if r.get("skipped"):
                summary_lines.append(
                    f"| {r['round']} | {r['model_a']} | {r['group_a']} | "
                    f"{r['model_b']} | {r['group_b']} | SKIPPED | SKIPPED | "
                    f"{r.get('reason', 'N/A')} | - | - | {r['winner']} | {r.get('reason', 'N/A')} |"
                )
            else:
                summary_lines.append(
                    f"| {r['round']} | {r['model_a']} | {r['group_a']} | "
                    f"{r['model_b']} | {r['group_b']} | "
                    f"{r['acc_a']:.4f} | {r['acc_b']:.4f} | "
                    f"{r['p_value']:.4g} | "
                    f"{'✓' if r['significant'] else '✗'} | "
                    f"{'✓' if r.get('bonferroni_significant', False) else '✗'} | "
                    f"{r['winner']} | {r.get('win_reason', '')} |"
                )
        summary_lines.append("")
        summary_lines.append("## Champion")
        summary_lines.append(
            f"- **Overall Winner: {current_champion_name}** ({current_champion_group})"
        )
        summary_lines.append("")
        summary_lines.append("## Round Details")
        for r in rounds:
            if r.get("skipped"):
                summary_lines.append(
                    f"### Round {r['round']}: {r['model_a']} vs {r['model_b']} (SKIPPED)"
                )
                summary_lines.append(f"- Reason: {r.get('reason', 'N/A')}")
                summary_lines.append(f"- Winner advances: {r['winner']}")
            else:
                summary_lines.append(
                    f"### Round {r['round']}: {r['model_a']} vs {r['model_b']}"
                )
                summary_lines.append(f"- Groups: {r['group_a']} vs {r['group_b']}")
                summary_lines.append(f"- Accuracy: {r['acc_a']:.4f} vs {r['acc_b']:.4f}")
                summary_lines.append(f"- Method: {r['method']}")
                summary_lines.append(f"- McNemar statistic: {r['statistic']:.4f}")
                summary_lines.append(f"- p-value: {r['p_value']:.4g}")
                summary_lines.append(f"- Significant (p < {ALPHA_FAMILY}): {r['significant']}")
                summary_lines.append(f"- Cohen's h: {r['cohen_h']:+.4f}")
                summary_lines.append(f"- Risk difference: {r['risk_difference']:+.4f}")
                summary_lines.append(f"- Discordant pairs: n01={r['n01']}, n10={r['n10']}, total={r['n_discordant']}")
                summary_lines.append(f"- Winner: {r['winner']} ({r['winner_group']}) — {r.get('win_reason', '')}")
                summary_lines.append(f"- {r['winner']} advances to next round as champion.")
            summary_lines.append("")

        summary_md = "\n".join(summary_lines) + "\n"

        if store is not None:
            store.write_stats(tournament, summary_md)
        else:
            st2 = self._staging()
            stats_dir = st2 / "stats" / self.domain
            stats_dir.mkdir(parents=True, exist_ok=True)
            (stats_dir / "ab_test_tournament.json").write_text(json.dumps(tournament, indent=2))
            (stats_dir / "summary_tournament.md").write_text(summary_md)
        return tournament
