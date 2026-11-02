# spam_fraud_detector/main.py

import sys
import logging
import warnings

# Import necessary classes from the src module
try:
    from spam_fraud_detector.core.data_loader import KaggleDataLoader
    from spam_fraud_detector.core.comparer import ClassifierGroupComparer
    from spam_fraud_detector.core.visualizer import ModelVisualizer
except ImportError as e:
    print(f"Error: {e}")
    print("Ensure that the classes (KaggleDataLoader, ClassifierGroupComparer, ModelVisualizer) are correctly defined within the 'src' package.")
    sys.exit(1)

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*use_label_encoder.*")
warnings.filterwarnings("ignore", message=".*X does not have valid feature names.*")
warnings.filterwarnings("ignore", message=".*LocalOutlierFactor was fitted with feature names.*")
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost.*")
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm.*")

logging.getLogger("xgboost").setLevel(logging.ERROR)
logging.getLogger("lightgbm").setLevel(logging.ERROR)


class SpamFraudDetector:
    """
    BenchmarkRunner Class
    ---------------------
    Handles the complete benchmarking pipeline for a specified task ('spam' or 'fraud').

    Steps:
        1. Load and preprocess data
        2. Compare classifiers and display metrics
        3. Visualize and export results
        4. (optional) Run explainability

    Modes:
        - training path (default): full pipeline.
        - explain-only path (--explain-only): loads existing bundles and runs
          SHAP explanations. Never retrains, never re-selects, never recomputes stats.
    """

    def __init__(
        self,
        task: str,
        save_format: str = "csv",
        report_name: str = None,
        seed: int = 42,
        skip_train: bool = False,
        skip_stats: bool = False,
        with_anomaly: bool = True,
        explainer: str = "shap",
        persona: str = "all",
        threshold_policy: str = "auto",
        selection_policy: str = "auto",
        strict_anomaly: bool = False,
        explain_only: bool = False,
        stats_only: bool = False,
        stats_tournament: bool = False,
        verbose: bool = False,
    ):
        self.task = task.lower()
        self.save_format = save_format
        self.report_name = report_name or f"{self.task}_detection"
        self.seed = seed
        self.skip_train = skip_train
        self.skip_stats = skip_stats
        self.with_anomaly = with_anomaly
        self.include_anomaly = with_anomaly
        self.explainer = explainer
        self.persona = persona
        self.threshold_policy = threshold_policy
        self.selection_policy = selection_policy
        self.strict_anomaly = strict_anomaly
        self.explain_only = explain_only
        self.stats_only = stats_only
        self.stats_tournament = stats_tournament
        self.verbose = verbose
        self.results = None
        self.store = None

        if self.task not in ["spam", "fraud"]:
            raise ValueError(f"Task '{self.task}' not recognized. Choose between 'spam' and 'fraud'.")

    # ------------------------------------------------------------------ #
    # Data loading & training
    # ------------------------------------------------------------------ #

    def _load_data(self):
        """Load and save the dataset."""
        print(f"  [1/3] Downloading/Loading data for {self.task}...")
        loader = KaggleDataLoader(task=self.task)
        df = loader.download_load_and_save(save_as=self.save_format)
        print("  Data loaded successfully.")
        return df

    def _spam_vocabulary(self) -> list:
        """Load the TF-IDF vocabulary from the persisted feature encoder.

        Returns the token list in vocabulary-index order so SHAP values can be
        mapped back to tokens. Used by the spam persona.
        """
        from pathlib import Path
        import joblib

        for cand in ("07_features", "06_features"):
            p = Path("data/processed/spam") / cand / "feature_encoder.joblib"
            if p.exists():
                enc = joblib.load(p)
                vocab = getattr(enc, "vocabulary_", None)
                if vocab is not None:
                    return [t for t, _ in sorted(vocab.items(), key=lambda kv: kv[1])]
                if hasattr(enc, "get_feature_names_out"):
                    return list(enc.get_feature_names_out())

        raise FileNotFoundError(
            "Missing data/processed/spam/{07_features,06_features}/feature_encoder.joblib. "
            "Run `make data` first."
        )

    def _compare_classifiers(self, df):
        """Prepare, train, and evaluate classifiers."""
        print("\n  [2/3] Preparing and comparing classifiers...")

        # Build a ResultStore so bundles + dashboards are written atomically.
        from spam_fraud_detector.core.result_store import ResultStore, RunContext
        from spam_fraud_detector.core.utils import get_git_commit, get_package_versions
        from pathlib import Path

        domain = f"{self.task}_detection"
        ctx = RunContext(
            domain=domain,
            seed=self.seed,
            git_commit=get_git_commit(),
            package_versions=get_package_versions(),
        )
        self.store = ResultStore(Path("outputs"), ctx)

        comparer = ClassifierGroupComparer(task=self.task, df=df, store=self.store, verbose=self.verbose)
        comparer.strict_anomaly = self.strict_anomaly
        comparer.threshold_policy = self.threshold_policy
        comparer.include_anomaly = self.include_anomaly
        comparer.load_data()
        comparer.prepare_data()
        results = comparer.compare_groups()
        print("  Comparison complete.")
        return results

    def _display_results(self, results):
        """Display model performance metrics and confusion matrices."""
        print("\n" + "=" * 25 + " PERFORMANCE METRICS " + "=" * 25)
        print("\n Scaled Models Performance:\n", results.get("scaled_metrics", "N/A"))
        print("\n" + "-" * 75)
        print("\n Unscaled Models Performance:\n", results.get("unscaled_metrics", "N/A"))
        print("\n" + "-" * 75)
        print("\n Group-Level Comparison:\n", results.get("group_comparison", "N/A"))

        print("\n" + "=" * 25 + " CONFUSION MATRICES " + "=" * 26)
        print("\n Confusion Matrices (Scaled Models):")
        scaled_matrices = results.get("scaled_confusion_matrices", {})
        for model_name, matrix in scaled_matrices.items():
            print(f"\n {model_name}:\n{matrix}")

        print("\n Confusion Matrices (Unscaled Models):")
        unscaled_matrices = results.get("unscaled_confusion_matrices", {})
        for model_name, matrix in unscaled_matrices.items():
            print(f"\n {model_name}:\n{matrix}")

    def _visualize_and_export(self, results):
        """Generate plots and export results."""
        print("\n  [3/3] Generating visualizations and exporting results...")

        yticklabels = ["Not Spam", "Spam"] if self.task == "spam" else ["Legitimate", "Fraud"]
        visualizer = ModelVisualizer(
            results=results,
            report_name=self.report_name,
            yticklabels=yticklabels,
        )

        visualizer.save_f1_scores()
        visualizer.save_group_comparison()
        visualizer.save_confusion_matrices()
        visualizer.save_roc_curves()
        visualizer.save_precision_recall_curves()
        visualizer.export_metrics_to_excel()
        visualizer.export_plots_to_pdf()
        print("  Visualizations and exports completed.")

    # ------------------------------------------------------------------ #
    # Explain-only path (three distinct personas)
    # ------------------------------------------------------------------ #

    def _run_explain_only(self) -> None:
        """Explain-only path. Loads existing bundles; runs SHAP; writes meta.json only.

        Never retrains, never re-selects, never recomputes stats.

        Three distinct persona jobs:

          - fraud: SHAP summary of every supervised fraud model.
          - spam:  token-level SHAP summary of every supervised spam model.
          - anomaly: does NOT re-explain the anomaly score. Instead, produces
                     a **comparison report** between each anomaly detector and
                     the best supervised fraud model, highlighting:
                       * novel transactions (anomaly high, supervised low),
                       * sophisticated fraud (supervised high, anomaly low).
        """
        from pathlib import Path
        import json
        import hashlib
        import joblib
        import numpy as np
        from datetime import datetime, timezone

        from spam_fraud_detector.core.result_store import ResultStore, RunContext
        from spam_fraud_detector.core.explainer import (
            make_explainer, build_casefile, build_hypothetical_casefile,
            render_case_waterfall, render_hypothetical_case_waterfall,
            _shap_row, _top_features_from_shap,
        )
        from spam_fraud_detector.core.fake_subjects import load_scenarios, subject_generator
        from spam_fraud_detector.core.recommendations import recommend
        from spam_fraud_detector.core.case_renderer import render_case_summary
        from spam_fraud_detector.core.utils import get_git_commit, get_package_versions, atomic_write_json
        from spam_fraud_detector.core.exceptions import DatasetNotFoundError

        # sklearn emits "X does not have valid feature names ..." whenever an
        # estimator fitted on a DataFrame is called with an array (or vice
        # versa). Results are correct; silence the noise for this explain run.
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names",
        )

        domain = f"{self.task}_detection"
        bundle_dir = Path("outputs/models") / domain / "all"
        if not bundle_dir.exists() or not list(bundle_dir.glob("*.joblib")):
            raise DatasetNotFoundError(
                f"No bundles at {bundle_dir}. Run `make train` (without --explain-only) first."
            )

        if self.explainer == "off":
            print("[explain] skipped: --explainer off")
            return

        val_path = Path("data/processed") / self.task / "01_split" / "val.parquet"
        if not val_path.exists():
            raise DatasetNotFoundError(
                f"No val split at {val_path}. Run `make train-{self.task}` first."
            )
        import pandas as pd
        val = pd.read_parquet(val_path)
        label_col = "Class" if self.task == "fraud" else "Prediction"
        drop_cols = ["Time"] if self.task == "fraud" else ["Email No."]
        feature_cols = [c for c in val.columns if c not in (label_col, "_row_id") and c not in drop_cols]
        feature_names_hash = "sha256:" + hashlib.sha256(
            ",".join(feature_cols).encode("utf-8")
        ).hexdigest()

        ctx = RunContext(
            domain=domain,
            seed=self.seed,
            git_commit=get_git_commit(),
            package_versions=get_package_versions(),
        )
        store = ResultStore(Path("outputs"), ctx)

        out_dir = Path("reports") / domain / "docs" / "explainability"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Pre-compute supervised val scores so the anomaly branch can compare.
        supervised_scores_by_model = {}
        if self.persona in ("anomaly", "all"):
            for bundle in sorted(bundle_dir.glob("*.joblib")):
                meta_path = bundle.with_suffix(".meta.json")
                if not meta_path.exists():
                    continue
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if meta.get("group") == "anomaly":
                    continue
                try:
                    model = joblib.load(bundle)
                    fitted_cols = meta.get("feature_columns")
                    if fitted_cols is None:
                        fitted_cols = feature_cols
                    X_val = val[[c for c in fitted_cols if c in val.columns]]
                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba(X_val)[:, 1]
                    elif hasattr(model, "decision_function"):
                        probs = model.decision_function(X_val)
                    else:
                        continue
                    supervised_scores_by_model[meta["model_name"]] = probs
                except Exception as exc:
                    logging.getLogger(__name__).warning(
                        "supervised pre-computation skipped %s: %s",
                        meta.get("model_name", "?"), exc,
                    )
                    continue

        explained = skipped = failed = 0
        scenario_pool = {p: load_scenarios(p) for p in ("fraud", "spam", "anomaly")}
        recorded_at = datetime.now(timezone.utc).isoformat()

        for bundle in sorted(bundle_dir.glob("*.joblib")):
            meta_path = bundle.with_suffix(".meta.json")
            if not meta_path.exists():
                continue
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            group = meta["group"]
            name = meta["model_name"]
            threshold = float(meta.get("threshold", 0.5))

            if group == "anomaly":
                persona = "anomaly"
            else:
                persona = "fraud" if self.task == "fraud" else "spam"

            if self.persona not in (persona, "all"):
                skipped += 1
                continue

            try:
                model = joblib.load(bundle)
                fitted_cols = meta.get("feature_columns") or feature_cols
                X_val = val[[c for c in fitted_cols if c in val.columns]]
                y_true_arr = val[label_col].values

                background_hash = "sha256:" + hashlib.sha256(
                    "\n".join(sorted(str(i) for i in val.index)).encode("utf-8")
                ).hexdigest()
                feature_names_hash = "sha256:" + hashlib.sha256(
                    ",".join(fitted_cols).encode("utf-8")
                ).hexdigest()

                if persona == "spam":
                    vocab = self._spam_vocabulary()
                    explainer = make_explainer("spam", model, X_val,
                                               vocabulary=vocab, backend="shap")
                else:
                    explainer = make_explainer(persona, model, X_val,
                                               feature_names=fitted_cols,
                                               backend="shap")

                casefiles = []
                provenance_base = {
                    "model": name,
                    "group": group,
                    "dataset_hash": ctx.dataset_hash,
                    "background_hash": background_hash,
                    "feature_names_hash": feature_names_hash,
                    "split": "val",
                    "recorded_at": recorded_at,
                }

                if persona == "anomaly":
                    # ── anomaly empirical cases (top-3 anomaly scores) ──
                    sup_name = (next(iter(supervised_scores_by_model))
                                if supervised_scores_by_model else None)
                    raw_scores = model.decision_function(X_val)
                    s_min, s_max = float(raw_scores.min()), float(raw_scores.max())
                    if s_max - s_min > 1e-12:
                        anomaly_scores = (raw_scores - s_min) / (s_max - s_min)
                    else:
                        anomaly_scores = np.zeros_like(raw_scores)
                    for ridx in np.argsort(anomaly_scores)[::-1][:3]:
                        ridx = int(ridx)
                        row = X_val.iloc[[ridx]]
                        sup_score = float(supervised_scores_by_model[sup_name][ridx]) if sup_name else 0.0
                        if anomaly_scores[ridx] > 0.7 and sup_score < 0.5:
                            disagreement = "novel"
                        elif sup_score > 0.5 and anomaly_scores[ridx] < 0.3:
                            disagreement = "sophisticated"
                        elif anomaly_scores[ridx] > 0.7 and sup_score > 0.5:
                            disagreement = "agree_high"
                        else:
                            disagreement = "agree_low"
                        subject = subject_generator(
                            f"A-{ridx:06d}",
                            seed=42, persona="anomaly")
                        sv, _ = _shap_row(explainer, row)
                        tf = _top_features_from_shap(row, fitted_cols, sv, 5, "anomaly")
                        provenance = dict(provenance_base)
                        provenance["row_id"] = ridx
                        casefile = {
                            "schema_version": "1.0.0",
                            "case_type": "empirical",
                            "persona": "anomaly",
                            "subject": subject.to_dict(),
                            "score": float(anomaly_scores[ridx]),
                            "anomaly_score": float(anomaly_scores[ridx]),
                            "raw_score": float(raw_scores[ridx]),
                            "supervised_score": sup_score,
                            "disagreement_type": disagreement,
                            "threshold": float(threshold),
                            "y_true": None,
                            "y_pred": int(anomaly_scores[ridx] >= threshold),
                            "top_features": tf,
                            "provenance": provenance,
                            "narrative": (
                                f"{subject.full_name}'s transaction was scored "
                                f"anomaly {anomaly_scores[ridx]:.3f} vs supervised {sup_score:.3f} "
                                f"({disagreement})."
                            ),
                        }
                        casefile["recommendation"] = recommend("anomaly", casefile)
                        png_path = out_dir / f"{name}__anomaly__case_{subject.subject_id}.png"
                        json_path = out_dir / f"{name}__anomaly__case_{subject.subject_id}.json"
                        render_case_waterfall(explainer, row, casefile, png_path, top_k=10)
                        atomic_write_json(json_path, casefile)
                        casefiles.append({
                            "json": str(json_path),
                            "png": str(png_path),
                            "subject_id": subject.subject_id,
                            "subject_name": subject.full_name,
                            "case_type": "empirical",
                            "narrative": casefile.get("narrative", ""),
                            "recommendation": casefile.get("recommendation", {}),
                            "subject": casefile.get("subject", {}),
                            "score": casefile.get("score") or casefile.get("anomaly_score"),
                            "threshold": casefile.get("threshold"),
                            "error_type": casefile.get("error_type"),
                            "top_features": casefile.get("top_features", []),
                            "provenance": casefile.get("provenance", {}),
                            "persona": persona,
                            "anomaly_score": casefile.get("anomaly_score"),
                            "supervised_score": casefile.get("supervised_score"),
                            "raw_score": casefile.get("raw_score"),
                        })
                else:
                    # ── supervised empirical cases (fraud / spam) ─────────
                    if hasattr(model, "predict_proba"):
                        scores = model.predict_proba(X_val)[:, 1]
                    elif hasattr(model, "decision_function"):
                        scores = model.decision_function(X_val)
                    else:
                        scores = None
                    if scores is not None:
                        scores = np.asarray(scores, dtype=float)
                        y_pred_arr = (scores >= threshold).astype(int)
                        case_specs = []
                        tp_idx = np.where((y_true_arr == 1) & (y_pred_arr == 1))[0]
                        if len(tp_idx):
                            case_specs.append(("true_positive",
                                               tp_idx[np.argsort(scores[tp_idx])[::-1]][:1]))
                        fp_idx = np.where((y_true_arr == 0) & (y_pred_arr == 1))[0]
                        if len(fp_idx):
                            case_specs.append(("false_positive",
                                               fp_idx[np.argsort(scores[fp_idx])[::-1]][:1]))
                        fn_idx = np.where((y_true_arr == 1) & (y_pred_arr == 0))[0]
                        if len(fn_idx):
                            case_specs.append(("false_negative",
                                               fn_idx[np.argsort(scores[fn_idx])[::-1]][:1]))
                        for err_type, idx in case_specs:
                            ridx = int(np.asarray(idx).item())
                            row = X_val.iloc[[ridx]]
                            score = float(scores[ridx])
                            y_true = int(y_true_arr[ridx])
                            y_pred = int(y_pred_arr[ridx])
                            subject = subject_generator(
                                f"{persona[0].upper()}-{ridx:06d}",
                                seed=42, persona=persona)
                            sv, _ = _shap_row(explainer, row)
                            tf = _top_features_from_shap(row, fitted_cols, sv, 5, persona)
                            provenance = dict(provenance_base)
                            provenance["row_id"] = ridx
                            casefile = build_casefile(
                                persona, model, row, y_true, y_pred, score,
                                threshold, group, subject.to_dict(), tf,
                                provenance, lambda cf: recommend(persona, cf))
                            png_path = out_dir / f"{name}__{group}__{persona}__case_{subject.subject_id}.png"
                            json_path = out_dir / f"{name}__{group}__{persona}__case_{subject.subject_id}.json"
                            render_case_waterfall(explainer, row, casefile, png_path, top_k=10)
                            atomic_write_json(json_path, casefile)
                        casefiles.append({
                            "json": str(json_path),
                            "png": str(png_path),
                            "subject_id": subject.subject_id,
                            "subject_name": subject.full_name,
                            "case_type": "empirical",
                            "narrative": casefile.get("narrative", ""),
                            "recommendation": casefile.get("recommendation", {}),
                            # Fields needed by render_case_summary:
                            "subject": casefile.get("subject", {}),
                            "score": casefile.get("anomaly_score") or casefile.get("score"),
                            "threshold": casefile.get("threshold"),
                            "disagreement_type": casefile.get("disagreement_type"),
                            "top_features": casefile.get("top_features", []),
                            "provenance": casefile.get("provenance", {}),
                            "persona": "anomaly",
                            "anomaly_score": casefile.get("anomaly_score"),
                            "supervised_score": casefile.get("supervised_score"),
                            "raw_score": casefile.get("raw_score"),
                        })

                # ── hypothetical cases ──────────────────────────────────
                for scenario in scenario_pool.get(persona, []):
                    try:
                        h_casefile, h_row = build_hypothetical_casefile(
                            persona, model, explainer, scenario,
                            fitted_cols, {}, None,
                            dict(provenance_base),
                            lambda sid, seed=42, persona=None: subject_generator(
                                sid, seed=seed, persona=persona),
                            lambda cf: recommend(persona, cf),
                            threshold,
                            top_k_features=5,
                        )
                        subj_id = h_casefile["subject"]["subject_id"]
                        png_path = out_dir / f"{name}__{group}__{persona}__hypo_{scenario['scenario_id']}.png"
                        json_path = out_dir / f"{name}__{group}__{persona}__hypo_{scenario['scenario_id']}.json"
                        render_hypothetical_case_waterfall(explainer, h_row, h_casefile, png_path, top_k=10)
                        atomic_write_json(json_path, h_casefile)
                        casefiles.append({
                            "json": str(json_path),
                            "png": str(png_path),
                            "subject_id": subj_id,
                            "subject_name": h_casefile["subject"]["full_name"],
                            "case_type": "hypothetical",
                            "scenario_id": scenario["scenario_id"],
                            "narrative": h_casefile.get("narrative", ""),
                            "recommendation": h_casefile.get("recommendation", {}),
                            # Fields needed by render_case_summary:
                            "subject": h_casefile.get("subject", {}),
                            "score": h_casefile.get("score") or h_casefile.get("anomaly_score"),
                            "threshold": h_casefile.get("threshold"),
                            "top_features": h_casefile.get("top_features", []) or h_casefile.get("top_tokens", []),
                            "provenance": h_casefile.get("provenance", {}),
                            "persona": persona,
                            "anomaly_score": h_casefile.get("anomaly_score"),
                            "supervised_score": h_casefile.get("supervised_score"),
                            "raw_score": h_casefile.get("raw_score"),
                            "disagreement_type": h_casefile.get("disagreement_type"),
                        })
                    except Exception as exc:
                        print(f"[explain] {name}__{group} hypo {scenario['scenario_id']} FAILED: {exc}")
                        failed += 1

                summary_path = (casefiles[0]["png"] if casefiles
                                else str(out_dir / f"{name}__{group}__{persona}__summary.png"))
                store.record_explainability_direct(
                    name=name, group=group,
                    paths={
                        "backend": "shap",
                        "summary_path": summary_path,
                        "waterfall_paths": [],
                        "background_hash": background_hash,
                        "feature_names_hash": feature_names_hash,
                    },
                    persona=persona,
                    casefiles=casefiles,
                )
                explained += 1
                print(f"[explain] {name}__{group} → {len(casefiles)} case files")
                print(f"    (written to reports/{domain}/docs/explainability/)")
                print()
                for cf in casefiles:
                    try:
                        print(render_case_summary(cf))
                    except Exception as render_exc:
                        import traceback as _tb
                        print(f"[RENDER FAIL] {cf.get('subject_id', '?')}: {render_exc}")
                        _tb.print_exc()
                    print()

            except Exception as exc:
                print(f"[explain] {name}__{group} FAILED: {exc}")
                failed += 1

        print(f"Explain-only run complete. {explained} explained, {skipped} skipped, {failed} failed.")

    def _run_stats_only(self) -> None:
        """Run A/B statistical analysis only, without retraining models.

        Loads existing bundles and predictions, runs StatsEngine.compute(),
        writes ab_test.json + summary.md to outputs/stats/<domain>/,
        and injects the summary into the domain doc.
        """
        from pathlib import Path
        from spam_fraud_detector.core.stats import StatsEngine

        domain = f"{self.task}_detection"
        st = Path("outputs")

        if not (st / "models" / domain / "all").exists():
            print(f"[stats-only] No bundles at outputs/models/{domain}/all. Run `make train` first.")
            return

        engine = StatsEngine(st, domain)
        ab = engine.compute(store=self.store)

        # Inject stats into domain doc
        import importlib.util
        from pathlib import Path

        scripts_dir = Path("scripts")
        inject_stats_path = scripts_dir / "inject_stats.py"
        if inject_stats_path.exists():
            spec = importlib.util.spec_from_file_location("inject_stats", inject_stats_path)
            inject_stats_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(inject_stats_mod)
            inject_stats_mod.inject_stats(domain)
        else:
            print(f"[stats-only] inject_stats.py not found at {inject_stats_path}")

        n_comp = len(ab.get("comparisons", []))
        n_sig = sum(1 for c in ab.get("comparisons", []) if c.get("significant_holm_0.05"))
        print(f"\n[stats-only] {domain}: {n_comp} comparisons, {n_sig} significant (Holm 0.05)")
        print(f"[stats-only] Written to outputs/stats/{domain}/ab_test.json")

    def _run_stats_tournament(self) -> None:
        """Run sequential pairwise A/B tournament only, without retraining models.

        Runs a champion-challenger tournament: model[0] vs model[1], then the
        winner vs model[2], etc., until all models have been evaluated.
        The final surviving model is the overall champion.

        Writes ab_test_tournament.json and summary_tournament.md to
        outputs/stats/<domain>/ and injects the summary into the domain doc.
        """
        from pathlib import Path
        from spam_fraud_detector.core.stats import StatsEngine

        domain = f"{self.task}_detection"
        st = Path("outputs")

        if not (st / "models" / domain / "all").exists():
            print(
                f"[stats-tournament] No bundles at outputs/models/{domain}/all. "
                f"Run `make train` first."
            )
            return

        engine = StatsEngine(st, domain)
        tournament = engine.compute_tournament(store=self.store)

        # Inject stats into domain doc
        import importlib.util
        from pathlib import Path

        scripts_dir = Path("scripts")
        inject_stats_path = scripts_dir / "inject_stats.py"
        if inject_stats_path.exists():
            spec = importlib.util.spec_from_file_location("inject_stats", inject_stats_path)
            inject_stats_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(inject_stats_mod)
            inject_stats_mod.inject_stats(domain)
        else:
            print(f"[stats-tournament] inject_stats.py not found at {inject_stats_path}")

        n_rounds = len(tournament.get("rounds", []))
        champion = tournament.get("champion", "unknown")
        champion_group = tournament.get("champion_group", "unknown")
        print(
            f"\n[stats-tournament] {domain}: {n_rounds} rounds, "
            f"champion={champion} ({champion_group})"
        )
        print(
            f"[stats-tournament] Written to outputs/stats/{domain}/ab_test_tournament.json"
        )

    # ------------------------------------------------------------------ #
    # Main dispatcher
    # ------------------------------------------------------------------ #

    def run(self):
        """Execute the pipeline. Dispatches on self.explain_only."""
        print(f"\n{'='*20} Starting benchmark for: {self.persona.upper()} detection {'='*20}")

        if self.explain_only:
            try:
                self._run_explain_only()
            except Exception as e:
                print(f"Error during explain-only execution: {e}")
                import traceback
                traceback.print_exc()
                sys.exit(1)
            print("\n" + "=" * 75)
            print(f"Explain-only run successfully completed for: {self.persona.upper()} detection")
            print("=" * 75)
            return

        if self.stats_only:
            try:
                self._run_stats_only()
            except Exception as e:
                print(f"Error during stats-only execution: {e}")
                import traceback
                traceback.print_exc()
                sys.exit(1)
            print("\n" + "=" * 75)
            print(f"Stats-only run successfully completed for: {self.persona.upper()} detection")
            print("=" * 75)
            return

        if self.stats_tournament:
            try:
                self._run_stats_tournament()
            except Exception as e:
                print(f"Error during stats-tournament execution: {e}")
                import traceback
                traceback.print_exc()
                sys.exit(1)
            print("\n" + "=" * 75)
            print(f"Stats-tournament run successfully completed for: {self.persona.upper()} detection")
            print("=" * 75)
            return

        try:
            df = self._load_data()
            self.results = self._compare_classifiers(df)
            self._display_results(self.results)
            self._visualize_and_export(self.results)

            # Run explainability on the trained bundles (staging) before finalize.
            if self.store is not None and self.explainer != "off":
                try:
                    self._run_explainability()
                except Exception as e:
                    print(f"Warning: _run_explainability() failed: {e}")

            # Finalize the ResultStore (writes dashboard + bundles atomically).
            if self.store is not None:
                try:
                    self.store.finalize()
                except Exception as e:
                    print(f"Warning: store.finalize() failed: {e}")

            print("\n" + "=" * 75)
            print(f"Benchmark successfully completed for: {self.task.upper()} detection")
            print("=" * 75)

        except Exception as e:
            print(f"Error during benchmark execution: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    def _run_explainability(self):
        """Run explainability for trained bundles before finalize."""
        try:
            import json
            from pathlib import Path
            import joblib
            from spam_fraud_detector.core.explainer import make_explainer

            domain = f"{self.task}_detection"
            out_dir = Path(f"reports/{domain}/docs/explainability")
            out_dir.mkdir(parents=True, exist_ok=True)

            # Load val split for SHAP background/sample.
            import pandas as pd
            val_path = Path("data/processed") / self.task / "01_split" / "val.parquet"
            val_df = None
            if val_path.exists():
                val_df = pd.read_parquet(val_path)
                label_col = "Class" if self.task == "fraud" else "Prediction"
                drop_cols = ["Time"] if self.task == "fraud" else ["Email No."]
                feature_cols = [c for c in val_df.columns if c not in (label_col, "_row_id") and c not in drop_cols]

            # Find top tree model bundle in staging area
            bundle_dir = None
            if self.store is not None and getattr(self.store, "staging", None) is not None:
                cand = Path(self.store.staging) / "models" / domain / "all"
                if cand.exists():
                    bundle_dir = cand
            if bundle_dir is None:
                bundle_dir = Path(f"outputs/models/{domain}/all")
            if not bundle_dir.exists():
                return

            for bundle_path in sorted(bundle_dir.glob("*.joblib")):
                meta_path = bundle_path.with_suffix(".meta.json")
                if not meta_path.exists():
                    continue
                meta = json.loads(meta_path.read_text())
                group = meta.get("group", "")
                name = meta.get("model_name") or meta.get("model", "")
                # Only tree models
                if group == "anomaly":
                    persona = "anomaly"
                else:
                    persona = "fraud" if self.task == "fraud" else "spam"

                # Check persona filter
                if self.persona != "all" and self.persona != persona:
                    continue

                fitted_cols = meta.get("feature_columns")
                if fitted_cols is None:
                    fitted_cols = feature_cols
                if val_df is not None and fitted_cols:
                    X_val = val_df[[c for c in fitted_cols if c in val_df.columns]]
                else:
                    X_val = None

                # Load model
                try:
                    model = joblib.load(bundle_path)
                except Exception:
                    continue

                # Determine explainer backend
                backend = self.explainer if self.explainer != "off" else "off"
                if backend == "off":
                    if self.store is not None:
                        self.store.record_explainability(name, group, {"backend": "off"}, persona=persona)
                    continue

                # Get vocabulary for spam
                vocabulary = None
                if persona == "spam":
                    try:
                        enc_path = Path(f"data/processed/{self.task}/06_features/feature_encoder.joblib")
                        if enc_path.exists():
                            enc = joblib.load(enc_path)
                            if hasattr(enc, "vocabulary_"):
                                vocabulary = list(enc.vocabulary_.keys())
                    except Exception:
                        pass

                # Create explainer
                try:
                    if persona == "spam" and vocabulary:
                        explainer = make_explainer(persona="spam", model=model,
                                                   X_background=None, vocabulary=vocabulary,
                                                   backend=backend)
                    elif persona == "anomaly":
                        explainer = make_explainer(persona="anomaly", model=model,
                                                   X_background=None,
                                                   feature_names=fitted_cols,
                                                   backend=backend)
                    else:
                        explainer = make_explainer(persona="fraud", model=model,
                                                   X_background=None,
                                                   feature_names=fitted_cols or [],
                                                   backend=backend)
                except Exception:
                    continue

                # Write summary
                summary_path = out_dir / f"{name}__{group}__{persona}__summary.png"
                try:
                    explainer.explain_summary(X_val, summary_path)
                except Exception as exc:
                    logging.getLogger(__name__).debug("explain_summary failed: %s", exc)

                # Record explainability
                if self.store is not None:
                    self.store.record_explainability(name, group, {
                        "summary_path": str(summary_path),
                        "waterfall_paths": [],
                        "background_hash": "sha256:0" * 8,
                        "feature_names_hash": "sha256:0" * 8,
                        "backend": backend,
                        "persona": persona,
                    }, persona=persona)

        except Exception as e:
            print(f"Explainability failed: {e}")


__version__ = "0.1.0"


# ---------------------------------------------------------------------- #
# CLI
# ---------------------------------------------------------------------- #

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run the binary classification benchmark.")
    parser.add_argument("--dataset", "--task", dest="task", choices=["spam", "fraud"], required=True,
                        help="Dataset/task to benchmark.")
    parser.add_argument("--report", dest="report", default=None,
                        help="Report name (defaults to '<task>_detection').")
    parser.add_argument("--save-format", dest="save_format", default="csv",
                        help="Data save format (csv).")
    parser.add_argument("--seed", dest="seed", type=int, default=42,
                        help="Random seed.")
    parser.add_argument("--verbose", dest="verbose", action="store_true",
                        help="Enable debug-level training output per model.")
    parser.add_argument("--explainer", dest="explainer", choices=["shap", "lime", "off"], default="shap",
                        help="Explainer backend (default: shap).")
    parser.add_argument("--persona", dest="persona",
                        choices=["fraud", "spam", "anomaly", "all"], default="all",
                        help="Persona filter for --explain-only (default: all).")
    parser.add_argument("--explain-only", dest="explain_only", action="store_true",
                        help="Run explainability only, skip training.")
    parser.add_argument("--stats-only", dest="stats_only", action="store_true",
                        help="Run A/B statistical analysis only, skip training and explainability.")
    parser.add_argument("--stats-tournament", dest="stats_tournament", action="store_true",
                        help="Run sequential pairwise A/B tournament only (champion-challenger), skip training.")
    parser.add_argument("--strict-anomaly", dest="strict_anomaly", action="store_true",
                        help="Fail on anomaly errors.")
    parser.add_argument("--anomaly-only", dest="anomaly_only", action="store_true",
                        help="Run anomaly detectors only.")
    parser.add_argument("--no-anomaly", dest="no_anomaly", action="store_true",
                        help="Skip anomaly detectors.")
    parser.add_argument("--threshold-policy", dest="threshold_policy",
                        choices=["auto", "fixed"], default="auto",
                        help="Threshold policy (default: auto).")
    parser.add_argument("--selection-policy", dest="selection_policy",
                        choices=["auto", "pr_auc", "f1", "expected_cost"], default="auto",
                        help="Selection policy (default: auto).")
    parser.add_argument("--skip-train", dest="skip_train", action="store_true",
                        help="Skip training (use existing).")
    parser.add_argument("--skip-stats", dest="skip_stats", action="store_true",
                        help="Skip stats.")
    args = parser.parse_args()

    report_name = args.report or f"{args.task}_detection"

    detector = SpamFraudDetector(
        task=args.task,
        save_format=args.save_format,
        report_name=report_name,
        seed=args.seed,
        skip_train=args.skip_train,
        skip_stats=args.skip_stats,
        with_anomaly=not args.no_anomaly and not args.anomaly_only,
        explainer=args.explainer,
        persona=args.persona,
        threshold_policy=args.threshold_policy,
        selection_policy=args.selection_policy,
        strict_anomaly=args.strict_anomaly,
        stats_only=args.stats_only,
        stats_tournament=args.stats_tournament,
        explain_only=args.explain_only,
        verbose=args.verbose,
    )
    detector.run()


if __name__ == "__main__":
    main()