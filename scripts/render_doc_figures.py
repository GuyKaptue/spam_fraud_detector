"""Render all documentation figures from the evidence inventory.

Each render function reads data from the evidence inventory or the underlying
artifact files and produces a PNG at docs/figures/<name>.png.

All figures are deterministic: same input produces byte-identical output.
Uses matplotlib with Agg backend, fixed font sizes, fixed color palette.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Global style settings for deterministic output
# ---------------------------------------------------------------------------
mpl = matplotlib
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#333333",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "text.color": "#333333",
})

# Color palette (colorblind-safe)
PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
           "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

INVENTORY_PATH = Path("docs/EVIDENCE_INVENTORY.json")
FIGURES_DIR = Path("docs/figures")


def _load_inventory() -> dict:
    if not INVENTORY_PATH.exists():
        raise FileNotFoundError(f"{INVENTORY_PATH} not found. Run `make evidence` first.")
    with open(INVENTORY_PATH, encoding="utf-8") as f:
        return json.load(f)


def _safe_get(d: dict, *keys, default=None):
    """Safely navigate nested dicts."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _save_fig(fig: matplotlib.figure.Figure, name: str) -> Path:
    """Save a figure to docs/figures/<name>.png with deterministic settings."""
    out_dir = FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Individual render functions
# ---------------------------------------------------------------------------

def render_readme_hero_results(inventory: dict) -> Path:
    """Bar chart of PR-AUC per model for both domains."""
    models_spam = inventory.get("models", {}).get("spam_detection", {})
    models_fraud = inventory.get("models", {}).get("fraud_detection", {})

    names = []
    values = []
    colors = []

    for name, mdata in models_spam.items():
        eval_data = mdata.get("eval", {})
        test = eval_data.get("test", {}) if isinstance(eval_data, dict) else {}
        pr_auc = test.get("pr_auc")
        if pr_auc is not None:
            names.append(f"Spam: {mdata.get('model_name', name)}")
            values.append(pr_auc)
            colors.append(PALETTE[0])

    for name, mdata in models_fraud.items():
        eval_data = mdata.get("eval", {})
        test = eval_data.get("test", {}) if isinstance(eval_data, dict) else {}
        pr_auc = test.get("pr_auc")
        if pr_auc is not None:
            names.append(f"Fraud: {mdata.get('model_name', name)}")
            values.append(pr_auc)
            colors.append(PALETTE[1])

    if not names:
        return None

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(names, values, color=colors[:len(names)])
    ax.set_ylabel("PR-AUC")
    ax.set_title("Best Model per Domain by PR-AUC")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=45)
    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    return _save_fig(fig, "readme_hero_results")


def render_readme_results_table(inventory: dict) -> Path:
    """Render results table as figure."""
    models_spam = inventory.get("models", {}).get("spam_detection", {})
    models_fraud = inventory.get("models", {}).get("fraud_detection", {})

    rows = []
    for name, mdata in models_spam.items():
        eval_data = mdata.get("eval", {})
        test = eval_data.get("test", {}) if isinstance(eval_data, dict) else {}
        pr_auc = test.get("pr_auc", 0)
        f1 = test.get("f1", 0)
        threshold = mdata.get("threshold", 0)
        rows.append([mdata.get("model_name", name), "spam", f"{pr_auc:.3f}", f"{f1:.3f}", f"{threshold:.2f}"])

    for name, mdata in models_fraud.items():
        eval_data = mdata.get("eval", {})
        test = eval_data.get("test", {}) if isinstance(eval_data, dict) else {}
        pr_auc = test.get("pr_auc", 0)
        f1 = test.get("f1", 0)
        threshold = mdata.get("threshold", 0)
        rows.append([mdata.get("model_name", name), "fraud", f"{pr_auc:.3f}", f"{f1:.3f}", f"{threshold:.2f}"])

    if not rows:
        return None

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    table = ax.table(cellText=rows,
                     colLabels=["Model", "Domain", "PR-AUC", "F1", "Threshold"],
                     loc="center",
                     cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    ax.set_title("Model Comparison: PR-AUC, F1, Threshold")

    return _save_fig(fig, "readme_results_table")


def render_readme_personas_triptych(inventory: dict) -> Path:
    """Three side-by-side persona waterfall thumbnails."""
    spam_dir = Path("reports/spam_detection/docs/explainability")
    fraud_dir = Path("reports/fraud_detection/docs/explainability")
    spam_files = list(spam_dir.glob("*__case_*.png"))[:1] if spam_dir.exists() else []
    fraud_files = list(fraud_dir.glob("*__case_*.png"))[:1] if fraud_dir.exists() else []
    anomaly_files = list(fraud_dir.glob("*__anomaly__case_*.png"))[:1] if fraud_dir.exists() else []

    all_files = spam_files + fraud_files + anomaly_files
    if not all_files:
        return None

    fig, axes = plt.subplots(1, min(3, len(all_files)), figsize=(12, 4))
    if len(all_files) == 1:
        axes = [axes]

    titles = ["Spam", "Fraud", "Anomaly"][:len(all_files)]
    for ax, img_path, title in zip(axes, all_files, titles):
        if img_path.exists():
            img = plt.imread(str(img_path))
            ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")

    return _save_fig(fig, "readme_personas_triptych")


def render_framework_pipeline_diagram(inventory: dict) -> Path:
    """Pipeline diagram from data to explainability."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")

    stages = ["Data Ingestion", "Split", "Features", "Train Models",
              "Evaluate Metrics", "Threshold Sweep", "Explainability", "Dashboard"]
    x = np.arange(len(stages))
    np.zeros(len(stages))

    for i, (name, x_val) in enumerate(zip(stages, x)):
        ax.text(x_val, 0, name, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#e6f3ff",
                          edgecolor="#1f77b4"))
        if i < len(stages) - 1:
            ax.annotate("", xy=(x_val + 0.5, 0), xytext=(x_val + 0.5, 0),
                        arrowprops=dict(arrowstyle="->", color="#333333"))

    ax.set_xlim(-0.5, len(stages) - 0.5)
    ax.set_ylim(-1, 1)
    ax.set_title("SPAM_FRAUD_DETECTION Pipeline")

    return _save_fig(fig, "framework_pipeline_diagram")


def render_framework_layer_order(inventory: dict) -> Path:
    """Data pipeline L1-L7 timeline."""
    layers = ["01_split", "02_clean", "03_model", "04_compare",
              "05_kpis", "06_features", "07_explain"]
    np.arange(len(layers))

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.barh(range(len(layers)), [1] * len(layers), color=PALETTE[:len(layers)])
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(layers)
    ax.set_xlabel("Processing Stage")
    ax.set_title("Data Pipeline Layers L1-L7")
    ax.set_xlim(0, 1.2)

    return _save_fig(fig, "framework_layer_order")


def render_framework_mcnemar_decision_tree(inventory: dict) -> Path:
    """Decision tree for McNemar test selection."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis("off")

    ax.text(0.5, 0.9, "Threshold Sweep Results", ha="center", va="center",
            fontsize=12, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#e6f3ff", edgecolor="#1f77b4"))

    ax.text(0.25, 0.7, "n_discordant < 25?", ha="center", va="center",
            fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff3cd", edgecolor="#ffc107"))

    ax.text(0.1, 0.4, "McNemar Exact", ha="center", va="center",
            fontsize=9, bbox=dict(boxstyle="round,pad=0.4", facecolor="#d4edda", edgecolor="#28a745"))
    ax.text(0.4, 0.4, "McNemar Chi-Squared", ha="center", va="center",
            fontsize=9, bbox=dict(boxstyle="round,pad=0.4", facecolor="#d4edda", edgecolor="#28a745"))

    ax.text(0.75, 0.4, "Holm Correction", ha="center", va="center",
            fontsize=9, bbox=dict(boxstyle="round,pad=0.4", facecolor="#d4edda", edgecolor="#28a745"))

    ax.text(0.75, 0.2, "Expected Cost Threshold", ha="center", va="center",
            fontsize=9, bbox=dict(boxstyle="round,pad=0.4", facecolor="#d4edda", edgecolor="#28a745"))

    ax.annotate("", xy=(0.25, 0.65), xytext=(0.25, 0.7),
                arrowprops=dict(arrowstyle="->", color="#333333"))
    ax.annotate("", xy=(0.1, 0.45), xytext=(0.2, 0.65),
                arrowprops=dict(arrowstyle="->", color="#333333"))
    ax.annotate("", xy=(0.4, 0.45), xytext=(0.3, 0.65),
                arrowprops=dict(arrowstyle="->", color="#333333"))
    ax.annotate("", xy=(0.75, 0.45), xytext=(0.4, 0.65),
                arrowprops=dict(arrowstyle="->", color="#333333"))
    ax.annotate("", xy=(0.75, 0.35), xytext=(0.75, 0.45),
                arrowprops=dict(arrowstyle="->", color="#333333"))

    return _save_fig(fig, "framework_mcnemar_decision_tree")


def render_framework_threshold_cost_curve(inventory: dict) -> Path:
    """Threshold cost curve for best fraud model."""
    models = inventory.get("models", {}).get("fraud_detection", {})
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
    ax.set_title("Fraud: Threshold vs Expected Cost")

    # Mark threshold_star
    threshold_star = thresholds.get("threshold_star", None)
    if threshold_star is not None:
        # Find closest point
        closest_idx = min(range(len(thresholds_list)),
                          key=lambda i: abs(thresholds_list[i] - threshold_star))
        ax.axvline(x=thresholds_list[closest_idx], color="r", linestyle="--", alpha=0.7)
        ax.text(thresholds_list[closest_idx], ax.get_ylim()[1] * 0.9,
                f"threshold_star={threshold_star:.2f}",
                rotation=90, va="top", ha="right", fontsize=8)

    return _save_fig(fig, "framework_threshold_cost_curve")


def render_spam_class_distribution(inventory: dict) -> Path:
    """Class distribution for spam dataset."""
    models = inventory.get("models", {}).get("spam_detection", {})
    total_tn = total_fp = total_fn = total_tp = 0
    for name, mdata in models.items():
        confusion = mdata.get("confusion", {})
        if confusion and "test" in confusion:
            cm = confusion["test"]
            total_tn += cm.get("tn", 0)
            total_fp += cm.get("fp", 0)
            total_fn += cm.get("fn", 0)
            total_tp += cm.get("tp", 0)

    if total_tn + total_fp + total_fn + total_tp == 0:
        return None

    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["Negative", "Positive"]
    values = [total_tn + total_fn, total_tp + total_fp]
    ax.pie(values, labels=labels, autopct="%1.1f%%", colors=["#1f77b4", "#c44e52"])
    ax.set_title("Spam: Class Distribution")

    return _save_fig(fig, "spam_class_distribution")


def render_fraud_class_distribution(inventory: dict) -> Path:
    """Class distribution for fraud dataset."""
    models = inventory.get("models", {}).get("fraud_detection", {})
    total_tn = total_fp = total_fn = total_tp = 0
    for name, mdata in models.items():
        confusion = mdata.get("confusion", {})
        if confusion and "test" in confusion:
            cm = confusion["test"]
            total_tn += cm.get("tn", 0)
            total_fp += cm.get("fp", 0)
            total_fn += cm.get("fn", 0)
            total_tp += cm.get("tp", 0)

    if total_tn + total_fp + total_fn + total_tp == 0:
        return None

    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["Negative", "Positive"]
    values = [total_tn + total_fn, total_tp + total_fp]
    ax.pie(values, labels=labels, autopct="%1.1f%%", colors=["#1f77b4", "#c44e52"])
    ax.set_title("Fraud: Class Distribution")

    return _save_fig(fig, "fraud_class_distribution")


def render_spam_threshold_curve(inventory: dict) -> Path:
    """Threshold sweep curve for best spam model."""
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
    ax.set_title("Spam: Threshold vs Expected Cost")

    return _save_fig(fig, "spam_threshold_curve")


def render_fraud_threshold_curve(inventory: dict) -> Path:
    """Threshold sweep curve for best fraud model."""
    models = inventory.get("models", {}).get("fraud_detection", {})
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
    ax.set_title("Fraud: Threshold vs Expected Cost")

    return _save_fig(fig, "fraud_threshold_curve")


def render_spam_roc_curve(inventory: dict) -> Path:
    """ROC curve for best spam model."""
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
    roc = test_curve.get("roc", {}) if isinstance(test_curve, dict) else {}
    fpr = roc.get("fpr", [])
    tpr = roc.get("tpr", [])
    if not fpr or not tpr:
        return None

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fpr, tpr, "b-", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Spam: ROC Curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()

    return _save_fig(fig, "spam_roc_curve")


def render_fraud_roc_curve(inventory: dict) -> Path:
    """ROC curve for best fraud model."""
    models = inventory.get("models", {}).get("fraud_detection", {})
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
    roc = test_curve.get("roc", {}) if isinstance(test_curve, dict) else {}
    fpr = roc.get("fpr", [])
    tpr = roc.get("tpr", [])
    if not fpr or not tpr:
        return None

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fpr, tpr, "b-", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Fraud: ROC Curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()

    return _save_fig(fig, "fraud_roc_curve")


def render_spam_pr_curve(inventory: dict) -> Path:
    """PR curve for best spam model."""
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
    pr = test_curve.get("pr", {}) if isinstance(test_curve, dict) else {}
    precision = pr.get("precision", [])
    recall = pr.get("recall", [])
    if not precision or not recall:
        return None

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(recall, precision, "b-", linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Spam: PR Curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    return _save_fig(fig, "spam_pr_curve")


def render_fraud_pr_curve(inventory: dict) -> Path:
    """PR curve for best fraud model."""
    models = inventory.get("models", {}).get("fraud_detection", {})
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
    pr = test_curve.get("pr", {}) if isinstance(test_curve, dict) else {}
    precision = pr.get("precision", [])
    recall = pr.get("recall", [])
    if not precision or not recall:
        return None

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(recall, precision, "b-", linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Fraud: Precision-Recall Curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    return _save_fig(fig, "fraud_pr_curve")


def render_spam_threshold_sensitivity(inventory: dict) -> Path:
    """Threshold sensitivity for spam model."""
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
    ax.set_title("Spam: Threshold Sensitivity")

    return _save_fig(fig, "spam_threshold_sensitivity")


def render_fraud_threshold_sensitivity(inventory: dict) -> Path:
    """Threshold sensitivity for fraud model."""
    models = inventory.get("models", {}).get("fraud_detection", {})
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
    ax.set_title("Fraud: Threshold Sensitivity")

    return _save_fig(fig, "fraud_threshold_sensitivity")


def render_spam_pr_auc_by_model(inventory: dict) -> Path:
    """PR-AUC by model for spam."""
    models = inventory.get("models", {}).get("spam_detection", {})
    names = []
    values = []
    for name, mdata in models.items():
        eval_data = mdata.get("eval", {})
        test = eval_data.get("test", {}) if isinstance(eval_data, dict) else {}
        pr_auc = test.get("pr_auc")
        if pr_auc is not None:
            names.append(mdata.get("model_name", name))
            values.append(pr_auc)

    if not names:
        return None

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(names, values, color=PALETTE[:len(names)])
    ax.set_ylabel("PR-AUC")
    ax.set_title("Spam: PR-AUC by Model")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=45)
    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    return _save_fig(fig, "spam_pr_auc_by_model")


def render_fraud_pr_auc_by_model(inventory: dict) -> Path:
    """PR-AUC by model for fraud."""
    models = inventory.get("models", {}).get("fraud_detection", {})
    names = []
    values = []
    for name, mdata in models.items():
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
    ax.set_title("Fraud: PR-AUC by Model")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=45)
    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    return _save_fig(fig, "fraud_pr_auc_by_model")


def render_spam_holm_significance(inventory: dict) -> Path:
    """Holm significance plot for spam."""
    stats_data = inventory.get("stats", {}).get("spam_detection", {})
    ab_test = stats_data.get("ab_test", {})
    comparisons = ab_test.get("comparisons", [])
    if not comparisons:
        return None

    models_a = [c.get("model_a", "A") for c in comparisons]
    models_b = [c.get("model_b", "B") for c in comparisons]
    p_values = [c.get("p_value", 1.0) for c in comparisons]
    p_holm = [c.get("p_holm_adjusted", 1.0) for c in comparisons]

    labels = [f"{a} vs {b}" for a, b in zip(models_a, models_b)]

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(labels))
    ax.scatter(x, p_values, color="#c44e52", label="p-value", s=50)
    ax.scatter(x, p_holm, color="#4c72b0", label="Holm p-value", s=50, marker="s")
    ax.axhline(y=0.05, color="red", linestyle="--", alpha=0.5, label="α = 0.05")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("p-value")
    ax.set_ylim(0, 1)
    ax.set_title("Spam: Holm-Adjusted p-values")
    ax.legend(fontsize=8)

    return _save_fig(fig, "spam_holm_significance")


def render_fraud_holm_significance(inventory: dict) -> Path:
    """Holm significance plot for fraud."""
    stats_data = inventory.get("stats", {}).get("fraud_detection", {})
    ab_test = stats_data.get("ab_test", {})
    comparisons = ab_test.get("comparisons", [])
    if not comparisons:
        return None

    models_a = [c.get("model_a", "A") for c in comparisons]
    models_b = [c.get("model_b", "B") for c in comparisons]
    p_values = [c.get("p_value", 1.0) for c in comparisons]
    p_holm = [c.get("p_holm_adjusted", 1.0) for c in comparisons]

    labels = [f"{a} vs {b}" for a, b in zip(models_a, models_b)]

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(labels))
    ax.scatter(x, p_values, color="#c44e52", label="p-value", s=50)
    ax.scatter(x, p_holm, color="#4c72b0", label="Holm p-value", s=50, marker="s")
    ax.axhline(y=0.05, color="red", linestyle="--", alpha=0.5, label="α = 0.05")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("p-value")
    ax.set_ylim(0, 1)
    ax.set_title("Fraud: Holm-Adjusted p-values")
    ax.legend(fontsize=8)

    return _save_fig(fig, "fraud_holm_significance")


def render_spam_roc_vs_pr(inventory: dict) -> Path:
    """ROC vs PR for best spam model."""
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
    roc = test_curve.get("roc", {}) if isinstance(test_curve, dict) else {}
    pr = test_curve.get("pr", {}) if isinstance(test_curve, dict) else {}
    fpr = roc.get("fpr", [])
    tpr = roc.get("tpr", [])
    precision = pr.get("precision", [])
    recall = pr.get("recall", [])
    if not fpr or not tpr or not precision or not recall:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(fpr, tpr, "b-", linewidth=2)
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    ax2.plot(recall, precision, "b-", linewidth=2)
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("PR Curve")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)

    fig.suptitle("Spam: ROC vs PR")

    return _save_fig(fig, "spam_roc_vs_pr")


def render_fraud_roc_vs_pr(inventory: dict) -> Path:
    """ROC vs PR for best fraud model."""
    models = inventory.get("models", {}).get("fraud_detection", {})
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
    roc = test_curve.get("roc", {}) if isinstance(test_curve, dict) else {}
    pr = test_curve.get("pr", {}) if isinstance(test_curve, dict) else {}
    fpr = roc.get("fpr", [])
    tpr = roc.get("tpr", [])
    precision = pr.get("precision", [])
    recall = pr.get("recall", [])
    if not fpr or not tpr or not precision or not recall:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(fpr, tpr, "b-", linewidth=2)
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    ax2.plot(recall, precision, "b-", linewidth=2)
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("PR Curve")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)

    fig.suptitle("Fraud: ROC vs PR")

    return _save_fig(fig, "fraud_roc_vs_pr")


# Registry of all render functions
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
}


def render_all(inventory: dict, out_dir: Path = Path("docs/figures")) -> dict:
    """Render all figures and return a dict of name -> path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, render_fn in RENDER_FUNCTIONS.items():
        try:
            path = render_fn(inventory)
            if path:
                results[name] = str(path)
        except Exception as e:
            print(f"Warning: Failed to render {name}: {e}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Render all documentation figures")
    parser.add_argument("--inventory", default="docs/EVIDENCE_INVENTORY.json",
                        help="Path to evidence inventory JSON")
    parser.add_argument("--out", default="docs/figures",
                        help="Output directory for figures")
    args = parser.parse_args()

    inventory = json.loads(Path(args.inventory).read_text(encoding="utf-8"))
    results = render_all(inventory, Path(args.out))
    print(f"Rendered {len(results)} figures:")
    for name, path in sorted(results.items()):
        print(f"  {name}: {path}")