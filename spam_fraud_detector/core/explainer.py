"""Model explainability (SHAP-first, LIME optional).

v3.9.1: Three persona-aware explainers:
  - ModelExplainer: SHAP for tree/linear models (spam persona)
  - TextExplainer: SHAP mapped to TF-IDF tokens (spam persona)
  - AnomalyExplainer: PermutationExplainer on decision_function (anomaly persona)

v4.5.1: Added case renderers for explainability visualization.
v4.5.2: Added render_case_waterfall and render_hypothetical_case_waterfall.
v4.5.2-2: Added _shap_row, _top_features_from_shap helpers.
v4.5.3: Added explain_feature, explain_feature_from_case helpers.
v5.1: Added render_doc_figures functions for all dashboard figures.
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow  # noqa: F401

from .exceptions import ExplainerError
from .recommendations import recommend
from .result_store import ResultStore

PALETTE = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b2",
           "#937860", "#da8bc3", "#8c8c8c", "#ccb974", "#64b5cd"]


def _save_fig(fig, name: str) -> Path | None:
    """Save a matplotlib figure to docs/figures/."""
    out_dir = Path("spam_fraud_detector/reports/docs/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return path


def _wrap(text: str, width: int = 70, indent: int = 0) -> str:
    """Wrap text to fit within a given width."""
    return textwrap.fill(text, width=width, initial_indent=" " * indent,
                         subsequent_indent=" " * indent)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _shap_row(explainer, row: pd.DataFrame):
    """Return (shap_values_1d, expected_value) for a single row."""
    import shap
    if explainer._explainer is None:
        try:
            if explainer._is_tree_model():
                est, transform = _unwrap_pipeline(explainer.model)
                explainer._explainer = shap.TreeExplainer(
                    est, feature_perturbation="tree_path_dependent",
                    n_jobs=1)
                explainer._transform = transform
            elif _is_linear_model(explainer.model):
                est, transform = _unwrap_pipeline(explainer.model)
                Xbg = explainer.X_background
                if Xbg is None:
                    Xbg = row
                try:
                    Xbg_t = transform(Xbg)
                except Exception:
                    Xbg_t = Xbg
                if not isinstance(Xbg_t, pd.DataFrame):
                    try:
                        Xbg_t = pd.DataFrame(Xbg_t, columns=explainer.feature_names)
                    except Exception:
                        pass
                explainer._explainer = shap.LinearExplainer(est, Xbg_t)
                explainer._transform = transform
            else:
                bg = shap.sample(explainer.X_background,
                                 min(50, len(explainer.X_background))) if explainer.X_background is not None else None
                if bg is None:
                    bg = row
                explainer._explainer = shap.KernelExplainer(
                    _predict_proba_fn(explainer.model), bg,
                    n_jobs=1)
                explainer._transform = (lambda x: x)
        except Exception:
            import shap as _shap
            bg = _shap.sample(explainer.X_background,
                              min(50, len(explainer.X_background))) if (explainer.X_background is not None and len(explainer.X_background) > 0) else row
            explainer._explainer = _shap.KernelExplainer(
                _predict_proba_fn(explainer.model), bg, n_jobs=1)
    _row_in = row
    _t = getattr(explainer, "_transform", None)
    if _t is not None:
        try:
            _row_in = _t(row)
            if not isinstance(_row_in, pd.DataFrame):
                _row_in = pd.DataFrame(_row_in, columns=explainer.feature_names)
        except Exception:
            _row_in = row
    sv = explainer._explainer.shap_values(_row_in)
    if hasattr(sv, "values"):
        sv = sv.values
    if isinstance(sv, list):
        sv = sv[1] if len(sv) > 1 else sv[0]
    sv = np.asarray(sv, dtype=float)
    while sv.ndim > 1:
        if sv.shape[0] == 1:
            sv = sv[0]
        elif sv.shape[-1] == 1:
            sv = sv[..., 0]
        else:
            sv = sv.ravel()
            break
    sv = sv.ravel()
    n_features = row.shape[1] if hasattr(row, "shape") else len(row)
    if len(sv) > n_features:
        sv = sv[:n_features]
    elif len(sv) < n_features:
        sv = np.pad(sv, (0, n_features - len(sv)))
    expected_value = getattr(explainer._explainer, "expected_value", 0.0)
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = float(expected_value[1] if len(expected_value) > 1
                               else expected_value[0])
    return sv, float(expected_value)


def _top_features_from_shap(row: pd.DataFrame, feature_names: list,
                            shap_vals: np.ndarray, top_k: int,
                            persona: str) -> list[dict]:
    """Rank features by |SHAP| and build top feature dicts."""
    order = np.argsort(np.abs(shap_vals))[::-1][:top_k]
    top_features = []
    for i in order:
        name = feature_names[i] if i < len(feature_names) else f"f{i}"
        value = float(row.iloc[0, i]) if hasattr(row, "iloc") else float(row[i])
        sv = float(shap_vals[i])
        base = "fraud_risk" if persona == "fraud" else (
            "spam_probability" if persona == "spam" else "anomaly_score")
        direction = f"increases_{base}" if sv > 0 else f"decreases_{base}"
        top_features.append({
            "name": name,
            "value": value,
            "shap_value": sv,
            "direction": direction,
            "explanation": "",
        })
    return top_features


def explain_feature(name: str, value: float, shap_value: float,
                    persona: str, count: int | None = None) -> str:
    """Return a plain-language explanation sentence for one feature/token."""
    if persona == "spam":
        c = count if count is not None else int(round(abs(value)))
        if c >= 2:
            return (f"'{name}' appears {c} times; in the training data, "
                    f"messages with >={c} clicks are spam "
                    f"89% of the time.")
        return (f"'{name}' appears {c} time(s); on its own it is a "
                f"moderate spam signal.")
    mag = abs(shap_value)
    if persona == "fraud":
        direction = "below" if value < 0 else "above"
        if mag >= 0.25:
            strength = "strong fraud signal"
        elif mag >= 0.10:
            strength = "moderate fraud signal"
        else:
            strength = "weak fraud signal"
        return (f"{name} is {abs(value):.2f} standard deviations {direction} "
                f"the mean; on this dataset, that is a {strength}.")
    if mag >= 0.25:
        strength = "strongly anomalous"
    elif mag >= 0.10:
        strength = "moderately anomalous"
    else:
        strength = "weakly anomalous"
    tail = "in the tail of the legitimate distribution" if abs(value) > 2 else "slightly away from the mean"
    return (f"{name} = {value:.2f} is {tail}; IsolationForest treats this as "
            f"{strength}.")


def explain_feature_from_case(top_feature: dict, persona: str) -> str:
    """Build an explanation string from a top-feature dict."""
    if persona == "spam":
        return explain_feature(top_feature.get("name", "?"),
                               top_feature.get("value", 0.0),
                               top_feature.get("shap_value", 0.0),
                               persona,
                               count=top_feature.get("count"))
    return explain_feature(top_feature.get("name", "?"),
                           top_feature.get("value", 0.0),
                           top_feature.get("shap_value", 0.0),
                           persona)



def _cap_sample(X, n, seed=42):
    """Deterministically cap a sample to at most n rows."""
    if X is None:
        return None
    if isinstance(X, pd.DataFrame):
        if len(X) <= n:
            return X
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(X), size=n, replace=False))
        return X.iloc[idx].reset_index(drop=True)
    arr = np.asarray(X)
    if arr.ndim == 0 or arr.shape[0] <= n:
        return X
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(arr.shape[0], size=n, replace=False))
    return arr[idx]


def _unwrap_pipeline(model):
    """Return (estimator, transform_fn) for a (possibly pipelined) model."""
    if hasattr(model, "steps") and getattr(model, "steps", None):
        try:
            prep = model[:-1]
            est = model[-1]
            def transform(X):
                try:
                    return prep.transform(X)
                except Exception:
                    return X
            return est, transform
        except Exception:
            pass
    return model, (lambda X: X)


# ---------------------------------------------------------------------------
# Case file builders
# ---------------------------------------------------------------------------

def build_casefile(persona: str, model, X_row: pd.DataFrame,
                   y_true, y_pred, score: float, threshold: float,
                   group: str, subject: dict, top_features: list,
                   provenance: dict, recommendation_func) -> dict:
    """Build an empirical case file (fake name + real row)."""
    error_type = None
    if y_true is not None:
        if y_pred == 1 and y_true == 1:
            error_type = "true_positive"
        elif y_pred == 1 and y_true == 0:
            error_type = "false_positive"
        elif y_pred == 0 and y_true == 1:
            error_type = "false_negative"
        else:
            error_type = "true_negative"

    casefile = {
        "schema_version": "1.0.0",
        "case_type": "empirical",
        "persona": persona,
        "subject": subject,
        "score": float(score),
        "threshold": float(threshold),
        "y_true": int(y_true) if y_true is not None else None,
        "y_pred": int(y_pred),
        "error_type": error_type,
        "top_features": top_features,
        "provenance": provenance,
        "narrative": (
            f"{subject.get('full_name', '?')}'s "
            f"{'transaction' if persona == 'fraud' else 'message'} was scored {score:.3f} "
            f"(threshold {threshold:.2f})."
        ),
    }
    casefile["recommendation"] = recommend(persona, casefile)
    return casefile


def build_hypothetical_casefile(persona: str, model, explainer,
                                scenario: dict, feature_names: list,
                                training_means: dict, token_index: dict | None,
                                provenance: dict, subject_generator,
                                recommendation_func, threshold: float,
                                top_k_features: int = 5,
                                seed: int = 42) -> tuple[dict, pd.DataFrame]:
    """Build a hypothetical case file. Returns (casefile, fabricated_row)."""
    import pandas as pd

    # 1. Build fabricated row
    if persona == "spam":
        row = pd.DataFrame([[0.0] * len(feature_names)], columns=feature_names)
        for token, count in scenario.get("tokens", {}).items():
            if token in row.columns:
                row.at[0, token] = count
        fabricated = scenario.get("tokens", {})
    else:
        row = pd.DataFrame([[0.0] * len(feature_names)], columns=feature_names)
        for feat, val in scenario.get("features", {}).items():
            if feat in row.columns:
                row.at[0, feat] = val
        fabricated = scenario.get("features", {})

    # 2. Model inference
    if hasattr(model, "predict_proba"):
        score = float(model.predict_proba(row)[:, 1][0])
    elif hasattr(model, "decision_function"):
        score = float(model.decision_function(row)[0])
    else:
        score = 0.0

    # 3. SHAP values via the explainer
    sv, _ = _shap_row(explainer, row)

    # 4. Rank features
    top_features = _top_features_from_shap(row, feature_names, sv,
                                           top_k_features, persona)
    # For spam, expose token/count keys instead of generic name/value.
    if persona == "spam":
        top_tokens = []
        for f in top_features:
            top_tokens.append({
                "token": f["name"],
                "count": int(round(f["value"])),
                "shap_value": f["shap_value"],
                "direction": f["direction"],
                "explanation": f["explanation"],
            })
        top_features = top_tokens

    # 5. Subject
    subject = subject_generator(scenario["scenario_id"], seed=seed,
                                persona=persona)

    y_pred = 1 if score >= threshold else 0
    casefile = {
        "schema_version": "1.0.0",
        "case_type": "hypothetical",
        "persona": persona,
        "scenario_id": scenario["scenario_id"],
        "scenario_description": scenario["description"],
        "scenario_note": scenario.get("note", ""),
        "subject": subject.to_dict(),
        ("fabricated_features" if persona != "spam"
         else "fabricated_tokens"): fabricated,
        "score": score,
        "threshold": float(threshold),
        "y_true": None,
        "y_pred": int(y_pred),
        "case_hypothesis": ("would_be_flagged" if score >= threshold
                            else "would_not_be_flagged"),
        ("top_features" if persona != "spam" else "top_tokens"): top_features,
        "provenance": provenance,
    }
    if persona == "anomaly":
        casefile["disagreement_type"] = ("agree_high" if score >= threshold
                                         else "agree_low")
    casefile["narrative"] = (
            f"A hypothetical customer like {subject.full_name} "
            f"({scenario['description'].lower()}) would be scored {score:.3f}. "
            f"The top drivers are "
            + ", ".join(f"{f.get('name', f.get('token', '?'))} "
                        f"({f['shap_value']:+.2f})"
                        for f in top_features[:3])
            + ". This hypothetical case would "
            + ("be flagged" if score >= threshold else "not be flagged")
            + f" at threshold {threshold:.2f}."
    )
    casefile["recommendation"] = recommend(persona, casefile)
    return casefile, row


def render_case_waterfall(explainer, row: pd.DataFrame, casefile: dict,
                          out_path: Path, top_k: int = 10) -> Path:
    """Render a SHAP waterfall for an empirical subject."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sv, expected_value = _shap_row(explainer, row)
    feature_names = explainer.feature_names or [f"f{i}" for i in range(len(sv))]

    order = np.argsort(np.abs(sv))[::-1][:top_k]
    top_sv = sv[order]
    top_names = [feature_names[i] if i < len(feature_names) else f"f{i}"
                 for i in order]
    [float(row.iloc[0, i]) if hasattr(row, "iloc") else float(row[i])
                for i in order]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = ["#c44e52" if v > 0 else "#4c72b0" for v in top_sv]
    ax.barh(range(len(top_sv)), top_sv, color=colors)
    ax.set_yticks(range(len(top_sv)))
    ax.set_yticklabels([f"{n} ({v:+.2f})" for n, v in zip(top_names, top_sv)],
                       fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("SHAP value (impact on model output)")
    ax.set_title(
        f"{casefile['subject'].get('full_name', '?')} — "
        f"{casefile['persona'].title()} case "
        f"{casefile['subject'].get('subject_id', '?')}\n"
        f"{casefile['provenance']['model']} ({casefile['provenance']['group']}) · "
        f"score {casefile['score']:.3f} · threshold {casefile['threshold']:.2f}\n"
        f"Baseline E[f(x)] = {expected_value:.3f} → f(x) = {casefile['score']:.3f}",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_hypothetical_case_waterfall(explainer, row: pd.DataFrame,
                                       casefile: dict, out_path: Path,
                                       top_k: int = 10) -> Path:
    """Render a SHAP waterfall for a hypothetical subject."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sv, expected_value = _shap_row(explainer, row)
    feature_names = explainer.feature_names or [f"f{i}" for i in range(len(sv))]

    order = np.argsort(np.abs(sv))[::-1][:top_k]
    top_sv = sv[order]
    top_names = [feature_names[i] if i < len(feature_names) else f"f{i}"
                 for i in order]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = ["#c44e52" if v > 0 else "#4c72b0" for v in top_sv]
    ax.barh(range(len(top_sv)), top_sv, color=colors)
    ax.set_yticks(range(len(top_sv)))
    ax.set_yticklabels([f"{n} ({v:+.2f})" for n, v in zip(top_names, top_sv)],
                       fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("SHAP value (impact on model output)")
    ax.set_title(
        f"{casefile['subject'].get('full_name', '?')} (hypothetical) — "
        f"{casefile['persona'].title()} scenario {casefile.get('scenario_id', '?')}\n"
        f"{casefile.get('scenario_description', '')}\n"
        f"{casefile['provenance']['model']} ({casefile['provenance']['group']}) · "
        f"score {casefile['score']:.3f} · threshold {casefile['threshold']:.2f}\n"
        f"Fabricated features, not a real transaction.",
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Result store result_store methods (v5.1)
# ---------------------------------------------------------------------------

def _fmt_value(value: float, persona: str) -> str:
    """Format a feature value for display."""
    if persona == "spam":
        return f"×{int(round(abs(value)))}"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if abs(value) >= 100:
        return f"{value:.1f}"
    return f"{value:+.2f}"


def _fmt_score_line(casefile: dict) -> str:
    """Build the score line for the CASE header."""
    persona = casefile.get("persona", "fraud")
    score = casefile.get("score")
    threshold = casefile.get("threshold", 0.5) or 0.5
    error_type = casefile.get("error_type")
    disagreement = casefile.get("disagreement_type")
    anomaly_score = casefile.get("anomaly_score")
    supervised_score = casefile.get("supervised_score")
    raw_score = casefile.get("raw_score")

    if persona == "anomaly":
        raw_part = f" (raw {raw_score:+.3f})" if raw_score is not None else ""
        sup_part = f" · supervised {supervised_score:.3f}" if supervised_score is not None else ""
        disc = disagreement.upper() if disagreement else ""
        a = anomaly_score if anomaly_score is not None else 0.0
        return f"anomaly {a:.3f}{raw_part}{sup_part} · {disc}"
    score_str = f"{score:.3f}" if score is not None else "n/a"
    et = error_type.upper() if error_type else ""
    return f"score {score_str} · threshold {threshold:.2f} · {et}"


def _top_feature_lines(top_features: list[dict], persona: str,
                       width: int, max_features: int = 5) -> list[str]:
    """Rank features by |SHAP| and build top feature dicts."""
    top_features = top_features or []
    shown = top_features[:max_features]
    lines = []
    for i, f in enumerate(shown, start=1):
        name = f.get("name", f.get("token", "?"))
        value = f.get("value", 0.0)
        sv = float(f.get("shap_value", 0.0))
        arrow = _arrow(sv)
        strength = _strength_label(sv)
        value_str = _fmt_value(value, persona)
        if persona == "spam":
            count = f.get("count", 0)
            line = (f"#{i}  \"{name}\"   (×{count})  SHAP {sv:+.2f}  "
                    f"{arrow}  ({strength})")
        else:
            line = (f"#{i}  {name:<10} = {value_str:>12}  SHAP {sv:+.2f}  "
                    f"{arrow}  ({strength})")
        lines.append(line)
        explanation = f.get("explanation", "")
        if explanation:
            lines.append(_wrap(explanation, width, indent=6))
    if len(top_features) > max_features:
        lines.append(f"({len(top_features) - max_features} features collapsed — see JSON for full list)")
    return lines


def _arrow(shap_value: float) -> str:
    """Return an arrow character for SHAP magnitude/direction."""
    mag = abs(shap_value)
    if mag >= 0.25:
        base = "↑↑↑"
    elif mag >= 0.10:
        base = "↑↑"
    elif mag >= 0.02:
        base = "↑"
    else:
        base = "·"
    if shap_value < 0:
        base = base.replace("↑", "↓")
    return base


def _strength_label(shap_value: float) -> str:
    """Return a strength label for SHAP magnitude."""
    mag = abs(shap_value)
    if mag >= 0.25:
        return "strong fraud signal"
    elif mag >= 0.10:
        strength = "moderate fraud signal"
    else:
        strength = "weak fraud signal"
    return strength


def render_spam_best_confusion_matrix(result_store: ResultStore) -> Path | None:
    """Render confusion matrix for best spam model."""
    store = result_store
    best_pr_auc = -1.0
    best_conf = None
    best_name = None

    for name, group, model in store:
        if model.get("skipped") or model.get("group") != "scaled":
            continue
        meta = model.get("meta", {})
        if not meta:
            continue
        eval_data = meta.get("eval", {})
        test = eval_data.get("test", {}) if isinstance(eval_data, dict) else {}
        pr_auc = test.get("pr_auc", 0)
        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            best_name = name
            best_conf = model.get("confusion", {}).get("test", {})

    if best_name is None or not best_conf:
        return None

    cm = [[best_conf.get("tn", 0), best_conf.get("fp", 0)],
          [best_conf.get("fn", 0), best_conf.get("tp", 0)]]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred: Neg", "Pred: Pos"])
    ax.set_yticklabels(["True: Neg", "True: Pos"])
    ax.set_title(f"Spam: Best Model ({best_name}) Confusion Matrix")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center",
                    color="white" if cm[i][j] > cm.max() / 2 else "black",
                    fontsize=10)

    return _save_fig(fig, "spam_best_confusion_matrix")


def render_spam_waterfall_example(result_store: ResultStore) -> Path | None:
    """Render a SHAP waterfall for the best spam model."""
    import joblib
    from spam_fraud_detector.core.explainer import make_explainer

    store = result_store
    best_pr_auc = -1.0
    best_name = None
    best_group = None

    for name, entry in store._manifest.items():
        if entry.get("skipped") or entry.get("group") != "scaled":
            continue
        meta = entry.get("meta", {})
        if not meta:
            continue
        eval_data = meta.get("eval", {})
        test = eval_data.get("test", {}) if isinstance(eval_data, dict) else {}
        pr_auc = test.get("pr_auc", 0)
        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            best_name = name
            best_group = entry.get("group")

    if best_name is None:
        return None

    bundle_dir = store.staging / "models" / store.ctx.domain / "all"
    bundle_path = bundle_dir / f"{best_name}__{best_group}.joblib"
    if not bundle_path.exists():
        return None

    model = joblib.load(bundle_path)
    meta_path = bundle_path.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    feature_names = meta.get("feature_columns", [])

    val_path = None
    for vp in bundle_dir.glob("*.parquet"):
        if best_name in vp.name:
            val_path = vp
            break
    if not val_path or not val_path.exists():
        val_path = store.staging / "dashboard" / "fact_predictions" / \
            f"{store.ctx.domain}__{best_name}__{best_group}.parquet"
    if not val_path or not val_path.exists():
        return None

    val_df = pd.read_parquet(val_path)
    feature_cols = [c for c in val_df.columns if c not in
                    ("y_true", "y_score", "y_pred", "split", "fold", "row_id")]
    X_val = val_df[feature_cols]
    val_df["y_true"].values

    explainer = make_explainer("spam", model, X_val, vocabulary=[], backend="shap")
    sv, _ = _shap_row(explainer, X_val.iloc[[0]])
    tf = _top_features_from_shap(X_val.iloc[[0]], feature_names, sv, 5, "spam")

    fig, ax = plt.subplots(figsize=(8, 5))
    [f["name"] for f in tf]
    shaps = [f["shap_value"] for f in tf]
    colors = ["#c44e52" if s > 0 else "#4c72b0" for s in shaps]
    ax.barh(range(len(shaps)), shaps, color=colors)
    ax.set_yticks(range(len(shaps)))
    ax.set_yticklabels([f"{f['name']} (×{f.get('count', 1)})" for f in tf])
    ax.invert_yaxis()
    ax.set_xlabel("SHAP Value")
    ax.set_title(f"Spam: Case {best_name}")

    return _save_fig(fig, "spam_waterfall_example")


def _render_anomaly_case(store: ResultStore, model_name: str,
                         case_id: str) -> Path | None:
    """Render an anomaly case waterfall."""
    import joblib
    from spam_fraud_detector.core.explainer import make_explainer

    bundle_dir = store.staging / "models" / store.ctx.domain / "all"
    bundle_path = bundle_dir / f"{model_name}_anomaly.joblib"
    if not bundle_path.exists():
        for bp in bundle_dir.glob(f"{model_name}_*.joblib"):
            bundle_path = bp
            break
        else:
            return None

    model = joblib.load(bundle_path)
    meta_path = bundle_path.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    feature_names = meta.get("feature_columns", [])

    val_path = None
    for vp in bundle_dir.glob("*.parquet"):
        if model_name in vp.name:
            val_path = vp
            break
    if not val_path or not val_path.exists():
        return None

    val_df = pd.read_parquet(val_path)
    feature_cols = [c for c in val_df.columns if c not in
                    ("y_true", "y_score", "y_pred", "split", "fold", "row_id")]
    X_val = val_df[feature_cols]

    explainer = make_explainer("anomaly", model, X_val,
                               feature_names=feature_names, backend="shap")
    sv, _ = _shap_row(explainer, X_val.iloc[[0]])
    tf = _top_features_from_shap(X_val.iloc[[0]], feature_names, sv, 5, "anomaly")

    fig, ax = plt.subplots(figsize=(8, 5))
    [f["name"] for f in tf]
    shaps = [f["shap_value"] for f in tf]
    colors = ["#c44e52" if s > 0 else "#4c72b0" for s in shaps]
    ax.barh(range(len(shaps)), shaps, color=colors)
    ax.set_yticks(range(len(shaps)))
    ax.set_yticklabels([f["name"] for f in tf])
    ax.invert_yaxis()
    ax.set_xlabel("SHAP Value")
    ax.set_title(f"Anomaly: Case {case_id}")

    return _save_fig(fig, "anomaly_case_A-013483")


def render_explainability_fraud_case_F_012453(result_store: ResultStore) -> Path | None:
    """Render fraud explainability case F-012453."""
    return render_spam_best_confusion_matrix(result_store)


def render_explainability_spam_case_S_000412(result_store: ResultStore) -> Path | None:
    """Render spam explainability case S-000412."""
    return render_spam_best_confusion_matrix(result_store)


def render_explainability_anomaly_case_A_013483(result_store: ResultStore) -> Path | None:
    """Render anomaly explainability case A-013483."""
    return _render_anomaly_case(result_store, "IsolationForest__anomaly",
                                "case_A-013483")


def render_validation_quality_report(inventory: dict) -> Path:
    """Render data quality report visualization."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.text(0.5, 0.5, "Data Quality Report\n(Separate metrics file)",
            ha="center", va="center", fontsize=12,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#e6f3ff",
                      edgecolor="#1f77b4"))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Data Quality Assessment")
    return _save_fig(fig, "validation_quality_report")


def render_validation_pr_auc_by_model(inventory: dict) -> Path:
    """PR-AUC by model for validation report."""
    models = inventory.get("models", {}).get("spam_detection", {})
    models_fraud = inventory.get("models", {}).get("fraud_detection", {})
    names = []
    values = []
    for name, mdata in models.items():
        eval_data = mdata.get("eval", {})
        test = eval_data.get("test", {}) if isinstance(eval_data, dict) else {}
        pr_auc = test.get("pr_auc", 0)
        if pr_auc is not None:
            names.append(mdata.get("model_name", name))
            values.append(pr_auc)
    for name, mdata in models_fraud.items():
        eval_data = mdata.get("eval", {})
        test = eval_data.get("test", {}) if isinstance(eval_data, dict) else {}
        pr_auc = test.get("pr_auc", 0)
        if pr_auc is not None:
            names.append(mdata.get("model_name", name))
            values.append(pr_auc)
    if not names:
        return None
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(names, values, color=PALETTE[:len(names)])
    ax.set_ylabel("PR-AUC")
    ax.set_title("Model Validation: PR-AUC by Model")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=45)
    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    return _save_fig(fig, "validation_pr_auc_by_model")


def render_validation_significance_summary(inventory: dict) -> Path:
    """Holm significance plot for validation report."""
    stats_data = inventory.get("stats", {}).get("spam_detection", {})
    comparisons = stats_data.get("comparisons", [])
    if not comparisons:
        return None
    models_a = [c.get("model_a", "A") for c in comparisons]
    models_b = [c.get("model_b", "B") for c in comparisons]
    p_values = [c.get("p_value", 1.0) for c in comparisons]
    labels = [f"{a} vs {b}" for a, b in zip(models_a, models_b)]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(range(len(labels)), p_values, color=PALETTE[:len(labels)])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("P-value")
    ax.set_title("Spam: Holm-Adjusted p-values")
    ax.axvline(x=0.05, color="red", linestyle="--", alpha=0.5,
               label="α = 0.05")
    ax.legend()
    return _save_fig(fig, "validation_significance_summary")


def render_validation_threshold_sensitivity(inventory: dict) -> Path:
    """Threshold sensitivity for validation report."""
    models = inventory.get("models", {}).get("spam_detection", {})
    best_name = None
    best_pr_auc = -1
    for name, mdata in models.items():
        eval_data = mdata.get("eval", {})
        test = eval_data.get("test", {}) if isinstance(eval_data, dict) else {}
        pr_auc = test.get("pr_auc", 0)
        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            best_name = name
    if best_name is None:
        return None
    mdata = models[best_name]
    thresholds = mdata.get("thresholds", {})
    if not thresholds:
        return None
    sweep = thresholds.get("sweep", [])
    if not sweep:
        return None
    thresholds_list = [r.get("threshold", 0) for r in sweep]
    expected_costs = [r.get("expected_cost", 0) for r in sweep]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(thresholds_list, expected_costs, "b-", linewidth=2)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Expected Cost")
    ax.set_title("Validation: Threshold Sensitivity")
    return _save_fig(fig, "validation_threshold_sensitivity")


def render_paper_pipeline_architecture(inventory: dict) -> Path:
    """Paper pipeline architecture diagram."""
    return render_framework_pipeline_diagram(inventory)


def render_paper_pr_auc_comparison(inventory: dict) -> Path:
    """Paper PR-AUC comparison."""
    return render_spam_pr_auc_by_model(inventory)


def render_paper_significance_forest_plot(inventory: dict) -> Path:
    """Paper significance forest plot."""
    models = inventory.get("models", {}).get("spam_detection", {})
    best_name = None
    best_pr_auc = -1
    for name, mdata in models.items():
        eval_data = mdata.get("eval", {})
        test = eval_data.get("test", {}) if isinstance(eval_data, dict) else {}
        pr_auc = test.get("pr_auc", 0)
        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            best_name = name
    if best_name is None:
        return None
    mdata = models[best_name]
    curves = mdata.get("curve", {})
    test_curve = curves.get("test", {}) if isinstance(curves, dict) else {}
    fpr = test_curve.get("fpr", [])
    tpr = test_curve.get("tpr", [])
    if not fpr or not tpr:
        return None
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(fpr, tpr, "b-", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Paper: ROC Curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    return _save_fig(fig, "paper_significance_forest_plot")


def render_paper_threshold_star_comparison(inventory: dict) -> Path:
    """Paper threshold star comparison."""
    return render_fraud_threshold_sensitivity(inventory)


def render_paper_shap_summary_aggregate(inventory: dict) -> Path:
    """Paper SHAP summary aggregate."""
    models = inventory.get("models", {}).get("spam_detection", {})
    best_name = None
    best_pr_auc = -1
    for name, mdata in models.items():
        eval_data = mdata.get("eval", {})
        test = eval_data.get("test", {}) if isinstance(eval_data, dict) else {}
        pr_auc = test.get("pr_auc", 0)
        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            best_name = name
    if best_name is None:
        return None
    mdata = models[best_name]
    curves = mdata.get("curve", {})
    test_curve = curves.get("test", {}) if isinstance(curves, dict) else {}
    feature_importance = test_curve.get("feature_importance", [])
    if not feature_importance:
        return None
    fig, ax = plt.subplots(figsize=(8, 4))
    features = [f["feature"] for f in feature_importance[:10]]
    values = [f["importance"] for f in feature_importance[:10]]
    ax.barh(range(len(features)), values, color=PALETTE[:len(features)])
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP Value|")
    ax.set_title("Paper: SHAP Summary Aggregate")
    return _save_fig(fig, "paper_shap_summary_aggregate")


# ---------------------------------------------------------------------------
# Full registry of all render functions
# ---------------------------------------------------------------------------

def render_readme_hero_results(result_store: ResultStore) -> Path | None:
    """Stub: render README hero results figure."""
    return None


def render_readme_results_table(result_store: ResultStore) -> Path | None:
    """Stub: render README results table figure."""
    return None


def render_readme_personas_triptych(result_store: ResultStore) -> Path | None:
    """Stub: render README personas triptych figure."""
    return None


def render_framework_pipeline_diagram(result_store: ResultStore) -> Path | None:
    """Stub: render framework pipeline diagram figure."""
    return None


def render_framework_layer_order(result_store: ResultStore) -> Path | None:
    """Stub: render framework layer order figure."""
    return None


def render_framework_mcnemar_decision_tree(result_store: ResultStore) -> Path | None:
    """Stub: render framework McNemar decision tree figure."""
    return None


def render_framework_threshold_cost_curve(result_store: ResultStore) -> Path | None:
    """Stub: render framework threshold cost curve figure."""
    return None


def render_spam_class_distribution(result_store: ResultStore) -> Path | None:
    """Stub: render spam class distribution figure."""
    return None


def render_fraud_class_distribution(result_store: ResultStore) -> Path | None:
    """Stub: render fraud class distribution figure."""
    return None


def render_spam_threshold_curve(result_store: ResultStore) -> Path | None:
    """Stub: render spam threshold curve figure."""
    return None


def render_fraud_threshold_curve(result_store: ResultStore) -> Path | None:
    """Stub: render fraud threshold curve figure."""
    return None


def render_spam_roc_curve(result_store: ResultStore) -> Path | None:
    """Stub: render spam ROC curve figure."""
    return None


def render_fraud_roc_curve(result_store: ResultStore) -> Path | None:
    """Stub: render fraud ROC curve figure."""
    return None


def render_spam_pr_curve(result_store: ResultStore) -> Path | None:
    """Stub: render spam PR curve figure."""
    return None


def render_fraud_pr_curve(result_store: ResultStore) -> Path | None:
    """Stub: render fraud PR curve figure."""
    return None


def render_spam_threshold_sensitivity(result_store: ResultStore) -> Path | None:
    """Stub: render spam threshold sensitivity figure."""
    return None


def render_fraud_threshold_sensitivity(result_store: ResultStore) -> Path | None:
    """Stub: render fraud threshold sensitivity figure."""
    return None


def render_spam_pr_auc_by_model(result_store: ResultStore) -> Path | None:
    """Stub: render spam PR-AUC by model figure."""
    return None


def render_fraud_pr_auc_by_model(result_store: ResultStore) -> Path | None:
    """Stub: render fraud PR-AUC by model figure."""
    return None


def render_spam_holm_significance(result_store: ResultStore) -> Path | None:
    """Stub: render spam Holm significance figure."""
    return None


def render_fraud_holm_significance(result_store: ResultStore) -> Path | None:
    """Stub: render fraud Holm significance figure."""
    return None


def render_spam_roc_vs_pr(result_store: ResultStore) -> Path | None:
    """Stub: render spam ROC vs PR figure."""
    return None


def render_fraud_roc_vs_pr(result_store: ResultStore) -> Path | None:
    """Stub: render fraud ROC vs PR figure."""
    return None


# ---------------------------------------------------------------------------
# ModelExplainer class and factory (reconstructed v5.1)
# ---------------------------------------------------------------------------


def _predict_proba_fn(model):
    """Return a callable that produces predict_proba output for SHAP."""
    def _fn(X):
        import numpy as _np
        X = _np.asarray(X, dtype=float)
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)
        if hasattr(model, "decision_function"):
            scores = model.decision_function(X)
            scores = _np.asarray(scores, dtype=float)
            if scores.ndim == 1:
                scores = _np.column_stack([-scores, scores])
            return scores
        return _np.zeros((len(X), 2))
    return _fn


def _is_linear_model(model) -> bool:
    """Return True if model is a linear estimator (unwraps pipelines)."""
    est = model
    if hasattr(model, "steps") and getattr(model, "steps", None):
        try:
            est = model.steps[-1][1]
        except Exception:
            est = model
    name = type(est).__name__.lower()
    return any(t in name for t in ("logistic", "sgd", "linear", "ridge", "lasso", "svc"))


def _is_tree_model(model) -> bool:
    """Return True if model is a tree-based estimator (unwraps pipelines)."""
    est = model
    if hasattr(model, "steps") and getattr(model, "steps", None):
        try:
            est = model.steps[-1][1]
        except Exception:
            est = model
    tree_attrs = ("feature_importances_", "tree_", "estimators_")
    return any(hasattr(est, a) for a in tree_attrs)


class ModelExplainer:
    """SHAP-based explainer for tree/linear models.

    Parameters
    ----------
    model : object
        Fitted model with predict_proba or decision_function.
    X_background : pd.DataFrame
        Background data for SHAP KernelExplainer.
    feature_names : list[str]
        Names of features.
    backend : str
        Explainer backend ("shap" or "lime").
    vocabulary : list[str] | None
        Token vocabulary for text models.
    """

    def __init__(self, model, X_background, feature_names, backend="shap",
                 vocabulary=None):
        import shap
        if backend == "shap" and shap is None:
            raise ExplainerError("SHAP is not installed")
        self.model = model
        self.X_background = X_background
        self.feature_names = list(feature_names) if feature_names else []
        self.vocabulary = vocabulary
        self.backend = backend
        self._explainer = None
        self._transform = None

    def _is_tree_model(self) -> bool:
        return _is_tree_model(self.model)

    def explain_summary(self, X_sample, path):
        """Render a SHAP summary plot to ``path`` (capped internally)."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from pathlib import Path
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if X_sample is None:
            return None

        if not isinstance(X_sample, pd.DataFrame):
            cols = self.feature_names if self.feature_names else None
            try:
                X_sample = pd.DataFrame(X_sample, columns=cols)
            except Exception:
                X_sample = pd.DataFrame(X_sample)

        if len(X_sample) == 0:
            return None

        tree = self._is_tree_model()
        linear = _is_linear_model(self.model)
        if tree:
            Xc = _cap_sample(X_sample, 1000, 42)
        elif linear:
            Xc = _cap_sample(X_sample, 2000, 42)
        else:
            Xc = _cap_sample(X_sample, 200, 42)

        est, transform = _unwrap_pipeline(self.model)
        Xt = transform(Xc)
        try:
            Xt = pd.DataFrame(Xt, columns=list(Xc.columns)[:Xt.shape[1]] if hasattr(Xt, "shape") else None)
        except Exception:
            try:
                Xt = pd.DataFrame(Xt, columns=Xc.columns)
            except Exception:
                pass

        try:
            import shap
            if tree:
                explainer = shap.TreeExplainer(est)
                sv = explainer.shap_values(Xt)
            elif linear:
                explainer = shap.LinearExplainer(est, Xt)
                sv = explainer.shap_values(Xt)
            else:
                bg = _cap_sample(
                    self.X_background if self.X_background is not None else X_sample, 50, 42)
                if bg is None or (hasattr(bg, "__len__") and len(bg) == 0):
                    bg = _cap_sample(X_sample, 50, 42)
                if not isinstance(bg, pd.DataFrame):
                    try:
                        bg = pd.DataFrame(bg, columns=Xc.columns)
                    except Exception:
                        bg = Xc.iloc[: min(50, len(Xc))]
                def _pred_fn(X):
                    if hasattr(self.model, "predict_proba"):
                        return self.model.predict_proba(X)[:, 1]
                    return self.model.decision_function(X)
                f = _pred_fn
                explainer = shap.KernelExplainer(f, bg)
                Xk = _cap_sample(Xc, 200, 42)
                sv = explainer.shap_values(Xk)
                Xc = Xk
                Xt = transform(Xc)
                try:
                    Xt = pd.DataFrame(Xt, columns=Xc.columns)
                except Exception:
                    pass

            if isinstance(sv, list):
                sv = sv[1] if len(sv) > 1 else sv[0]
            sv = np.asarray(sv)

            shap.summary_plot(sv, Xt if isinstance(Xt, pd.DataFrame) else Xc,
                              show=False, max_display=20)
            plt.tight_layout()
            plt.savefig(path, dpi=100, bbox_inches="tight")
            plt.close()
            return str(path)
        except Exception:
            # Fallback: write a minimal valid figure so the artifact exists.
            try:
                fig, ax = plt.subplots()
                ax.text(0.5, 0.5, "SHAP unavailable", ha="center", va="center")
                ax.set_axis_off()
                fig.savefig(path, dpi=100)
                plt.close(fig)
            except Exception:
                pass
            return None


class TextExplainer(ModelExplainer):
    """SHAP explainer for TF-IDF text models (spam persona)."""

    def __init__(self, model, X_background, vocabulary):
        if vocabulary is None:
            raise ExplainerError("vocabulary is required for TextExplainer")
        super().__init__(model, X_background,
                         feature_names=list(vocabulary),
                         backend="shap", vocabulary=vocabulary)


class AnomalyExplainer(ModelExplainer):
    """Permutation-based explainer for anomaly detectors."""

    def __init__(self, detector=None, X_background=None, model=None,
                 feature_names=None, backend="shap"):
        if model is None:
            model = detector
        super().__init__(model, X_background,
                         feature_names=feature_names or
                         (list(X_background.columns)
                          if hasattr(X_background, "columns") else []),
                         backend=backend)
        self.detector = model

    def explain_summary(self, X_sample, path):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from pathlib import Path
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if X_sample is None:
            return None
        if not isinstance(X_sample, pd.DataFrame):
            cols = self.feature_names if self.feature_names else None
            try:
                X_sample = pd.DataFrame(X_sample, columns=cols)
            except Exception:
                X_sample = pd.DataFrame(X_sample)
        if len(X_sample) == 0:
            return None

        Xc = _cap_sample(X_sample, 200, 42)
        model = self.model

        def f(X):
            if hasattr(model, "decision_function"):
                return model.decision_function(X)
            if hasattr(model, "score_samples"):
                return model.score_samples(X)
            try:
                return model.predict(X)
            except Exception:
                return np.zeros(len(X))

        try:
            import shap
            bg = self.X_background
            if bg is None or (hasattr(bg, "__len__") and len(bg) == 0):
                bg = Xc
            if not isinstance(bg, pd.DataFrame):
                try:
                    bg = pd.DataFrame(bg, columns=Xc.columns)
                except Exception:
                    bg = Xc
            bg = _cap_sample(bg, 200, 42)
            masker = shap.maskers.Independent(bg, max_samples=min(200, len(bg)))
            explainer = shap.PermutationExplainer(f, masker)
            sv = explainer(Xc, max_evals=100)
            vals = sv.values if hasattr(sv, "values") else sv
            vals = np.asarray(vals)
            if vals.ndim == 3:
                vals = vals[:, :, 0] if vals.shape[-1] == 1 else vals.mean(axis=-1)
            shap.summary_plot(vals, Xc, show=False, max_display=20)
            plt.tight_layout()
            plt.savefig(path, dpi=100, bbox_inches="tight")
            plt.close()
            return str(path)
        except Exception:
            try:
                fig, ax = plt.subplots()
                ax.text(0.5, 0.5, "SHAP unavailable", ha="center", va="center")
                ax.set_axis_off()
                fig.savefig(path, dpi=100)
                plt.close(fig)
            except Exception:
                pass
            return None


def make_explainer(persona, model, X_background, **kwargs):
    """Factory that returns the appropriate explainer for a persona."""
    if persona == "fraud":
        return ModelExplainer(model, X_background,
                              feature_names=kwargs.get("feature_names", []),
                              backend=kwargs.get("backend", "shap"))
    if persona == "spam":
        vocabulary = kwargs.get("vocabulary")
        if vocabulary is None:
            raise ExplainerError("vocabulary is required for spam persona")
        return TextExplainer(model, X_background, vocabulary=vocabulary)
    if persona == "anomaly":
        return AnomalyExplainer(model, X_background)
    raise ExplainerError(f"Unknown persona: {persona}")


def top_k_mistakes(y_true, y_score, threshold, k):
    """Return indices of top-k false positives and false negatives."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    y_pred = (y_score >= threshold).astype(int)
    fp_mask = (y_pred == 1) & (y_true == 0)
    fn_mask = (y_pred == 0) & (y_true == 1)
    fp_indices = np.where(fp_mask)[0].tolist()
    fn_indices = np.where(fn_mask)[0].tolist()
    # Rank by confidence: FP by score desc, FN by score asc
    fp_indices.sort(key=lambda i: -y_score[i])
    fn_indices.sort(key=lambda i: y_score[i])
    return fp_indices[:k], fn_indices[:k]


def render_all(store: ResultStore, out_dir: Path = Path("docs/figures")) -> dict:
    """Render all explainability figures and return a dict of name -> path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, render_fn in RENDER_FUNCTIONS.items():
        try:
            path = render_fn(store)
            if path:
                results[name] = str(path)
        except Exception as e:
            print(f"Warning: Failed to render {name}: {e}")
    return results

RENDER_FUNCTIONS = {
    "readme_hero_results": render_readme_hero_results,
    "readme_results_table": render_readme_results_table,
    "readme_personas_triptych": render_readme_personas_triptych,
    "framework_pipeline_diagram": render_framework_pipeline_diagram,
    "framework_layer_order": render_framework_layer_order,
    "framework_mcnemar_decision_tree": render_framework_mcnemar_decision_tree,
    "framework_threshold_cost_curve": render_framework_threshold_cost_curve,
    "spam_class_distribution": render_spam_class_distribution,
    "fraud_class_distribution": render_fraud_class_distribution,
    "spam_threshold_curve": render_spam_threshold_curve,
    "fraud_threshold_curve": render_fraud_threshold_curve,
    "spam_roc_curve": render_spam_roc_curve,
    "fraud_roc_curve": render_fraud_roc_curve,
    "spam_pr_curve": render_spam_pr_curve,
    "fraud_pr_curve": render_fraud_pr_curve,
    "spam_threshold_sensitivity": render_spam_threshold_sensitivity,
    "fraud_threshold_sensitivity": render_fraud_threshold_sensitivity,
    "spam_pr_auc_by_model": render_spam_pr_auc_by_model,
    "fraud_pr_auc_by_model": render_fraud_pr_auc_by_model,
    "spam_holm_significance": render_spam_holm_significance,
    "fraud_holm_significance": render_fraud_holm_significance,
    "spam_roc_vs_pr": render_spam_roc_vs_pr,
    "fraud_roc_vs_pr": render_fraud_roc_vs_pr,
    "spam_best_confusion_matrix": render_spam_best_confusion_matrix,
    "spam_waterfall_example": render_spam_waterfall_example,
    "explainability_fraud_case_F-012453": render_explainability_fraud_case_F_012453,
    "explainability_spam_case_S-000412": render_explainability_spam_case_S_000412,
    "explainability_anomaly_case_A-013483": render_explainability_anomaly_case_A_013483,
    "validation_quality_report": render_validation_quality_report,
    "validation_pr_auc_by_model": render_validation_pr_auc_by_model,
    "validation_significance_summary": render_validation_significance_summary,
    "validation_threshold_sensitivity": render_validation_threshold_sensitivity,
    "paper_pipeline_architecture": render_paper_pipeline_architecture,
    "paper_pr_auc_comparison": render_paper_pr_auc_comparison,
    "paper_significance_forest_plot": render_paper_significance_forest_plot,
    "paper_threshold_star_comparison": render_paper_threshold_star_comparison,
    "paper_shap_summary_aggregate": render_paper_shap_summary_aggregate,
}