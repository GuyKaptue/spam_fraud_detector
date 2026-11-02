"""Top-K model selection from persisted outputs (no re-fitting)."""
from __future__ import annotations

import os
import shutil
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .result_store import ResultStore


class TopKSelector:
    def __init__(self, out_root, domain: str, k: int = 3, metric: str = "pr_auc",
                 selection_policy: str = "auto") -> None:
        self.out_root = Path(out_root)
        self.domain = domain
        self.k = k
        self.metric = metric
        self.selection_policy = selection_policy

    def _dashboard_dir(self) -> Path:
        # staging or finalized
        for cand in [self.out_root]:
            # try finalized first
            pass
        # prefer finalized outputs
        if (self.out_root / "dashboard").exists():
            return self.out_root / "dashboard"
        # search staging
        # No longer needed - outputs go directly to outputs/
        return self.out_root / "dashboard"

    def select(self, store: Optional[ResultStore] = None) -> List[Dict[str, Any]]:
        # Prefer the store's own staging dir when available (avoids picking a
        # stale outputs.staging.* directory from a previous run).
        dash = None
        if store is not None:
            cand = Path(store.staging) / "dashboard" if getattr(store, "staging", None) else None
            if cand and cand.exists():
                dash = cand
            else:
                cand2 = Path(store.out_root) / "dashboard" if getattr(store, "out_root", None) else None
                if cand2 and cand2.exists():
                    dash = cand2
        if dash is None:
            dash = self._dashboard_dir()
        manifest = pd.read_parquet(dash / "manifest.parquet")
        fact = pd.read_parquet(dash / "fact_metrics.parquet")
        # filter domain, non-skipped, test split, metric
        man = manifest[(manifest["domain"] == self.domain) & (~manifest["skipped"].astype(bool))]
        ftest = fact[(fact["domain"] == self.domain) & (fact["split"] == "test") & (fact["metric"] == self.metric)]
        # aggregate mean per (model, group)
        agg = ftest.groupby(["model", "group"], as_index=False)["value"].mean()
        agg = agg.rename(columns={"value": self.metric})
        # join manifest for artifact path
        merged = agg.merge(man[["model", "group", "artifact"]], on=["model", "group"], how="inner")

        # resolve selection policy
        policy = self.selection_policy
        staging = None
        if store is not None and getattr(store, "staging", None) is not None:
            staging = Path(store.staging)
        if staging is None:
            for p in self.out_root.parent.glob("outputs.staging.*"):
                if (p / "dashboard").exists():
                    staging = p
                    break
            if staging is None:
                staging = self.out_root

        if policy == "auto":
            has_cost = False
            all_dir = staging / "models" / self.domain / "all"
            if all_dir.exists():
                for meta_path in all_dir.glob("*.meta.json"):
                    try:
                        m = json.loads(meta_path.read_text())
                        if m.get("threshold_policy") == "expected_cost":
                            has_cost = True
                            break
                    except Exception:
                        pass
            policy = "expected_cost" if has_cost else "pr_auc"

        cost_map = {}
        try:
            fts = pd.read_parquet(dash / "fact_thresholds_at_star.parquet")
            fts = fts[fts["domain"] == self.domain]
            for _, r in fts.iterrows():
                cost_map[(r["model"], r["group"])] = (r.get("expected_cost_at_star"), r.get("threshold_star"))
        except Exception:
            pass

        if policy == "expected_cost":
            merged["expected_cost_at_star"] = merged.apply(
                lambda r: (cost_map.get((r["model"], r["group"]), (None, None))[0]), axis=1)
            merged = merged.sort_values("expected_cost_at_star", ascending=True, na_position="last").reset_index(drop=True)
        else:
            merged = merged.sort_values(self.metric, ascending=False).reset_index(drop=True)
        top = merged.head(self.k).copy()
        # determine staging dir for writing bundles (already resolved above)
        ranking = []
        top3_dir = (staging / "models" / self.domain / "top3") if staging else (self.out_root / "models" / self.domain / "top3")
        top3_dir.mkdir(parents=True, exist_ok=True)
        for i, row in top.iterrows():
            rank = i + 1
            model = row["model"]
            group = row["group"]
            stem = f"{model}__{group}"
            src_root = staging if staging else self.out_root
            src_joblib = src_root / "models" / self.domain / "all" / f"{stem}.joblib"
            src_meta = src_root / "models" / self.domain / "all" / f"{stem}.meta.json"
            dst_joblib = top3_dir / f"rank_{rank}_{model}.joblib"
            dst_meta = top3_dir / f"rank_{rank}_{model}.meta.json"
            try:
                os.link(src_joblib, dst_joblib)
            except Exception:
                shutil.copyfile(src_joblib, dst_joblib)
            if src_meta.exists():
                shutil.copyfile(src_meta, dst_meta)
            artifact_rel = str(Path("models") / self.domain / "top3" / f"rank_{rank}_{model}.joblib")
            ec, ts = cost_map.get((model, group), (None, None))
            ranking.append({
                "rank": rank,
                "model": model,
                "group": group,
                "selection_policy": policy,
                self.metric: float(row[self.metric]) if self.metric in row else None,
                "pr_auc": float(row[self.metric]) if self.metric == "pr_auc" else None,
                "f1": None,
                "recall_at_prec_0.90": None,
                "threshold": None,
                "threshold_star": (float(ts) if ts is not None else None),
                "expected_cost_at_star": (float(ec) if ec is not None else None),
                "artifact": artifact_rel,
            })
        # write selection via store if provided
        if store is not None:
            store.write_selection(ranking)
        else:
            # write selection.json directly
            import json
            sel_path = top3_dir / "selection.json"
            sel_path.write_text(json.dumps({"domain": self.domain, "ranking": ranking}, indent=2))
        return ranking
