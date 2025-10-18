# spam_fraud_detector/core/visualizer.py
# type: ignore
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from matplotlib.backends.backend_pdf import PdfPages

from spam_fraud_detector.core.utils import ensure_dir

class ModelVisualizer:
    def __init__(self, results: dict, report_name: str = "spam_detection", yticklabels: list = None):
        self.report_name = report_name
        self.figures_path = os.path.join("reports", report_name, "figures")
        self.docs_path = os.path.join("reports", report_name, "docs")

        self.scaled_metrics_df = results["scaled_metrics"]
        self.unscaled_metrics_df = results["unscaled_metrics"]
        self.group_comparison_df = results["group_comparison"]
        self.scaled_conf_matrices = results["scaled_confusion_matrices"]
        self.unscaled_conf_matrices = results["unscaled_confusion_matrices"]
        self.scaled_probs = {k: v["probs"] for k, v in results["scaled_raw"].items() if v["probs"] is not None}
        self.unscaled_probs = {k: v["probs"] for k, v in results["unscaled_raw"].items() if v["probs"] is not None}
        self.scaled_precision_df = results.get("scaled_precision", pd.DataFrame())
        self.unscaled_precision_df = results.get("unscaled_precision", pd.DataFrame())
        self.y_true = results["scaled_raw"][next(iter(results["scaled_raw"]))]["true"]
        self.yticklabels = yticklabels if yticklabels else ["Spam", "Not Spam"]

    def _save_plot(self, plot_func, filename: str, figsize=(10, 6)):
        path = os.path.join(self.figures_path, filename)
        ensure_dir(path)
        fig, ax = plt.subplots(figsize=figsize)
        plot_func(ax)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)

    def plot_f1_scores(self, ax):
        combined = pd.concat([self.scaled_metrics_df, self.unscaled_metrics_df])
        sorted_f1 = combined["f1"].sort_values(ascending=False).reset_index()
        sorted_f1.columns = ["Model", "F1"]
        sns.barplot(data=sorted_f1, x="F1", y="Model", ax=ax, palette="viridis", hue="Model", dodge=False, legend=False)
        ax.set_title("Model Comparison by F1 Score")
        ax.set_xlabel("F1 Score")
        ax.set_ylabel("Model")

    def save_f1_scores(self):
        self._save_plot(self.plot_f1_scores, f"{self.report_name}_f1_scores.png")

    def plot_avg_precision_scores(self, ax):
        combined = pd.concat([self.scaled_precision_df, self.unscaled_precision_df])
        combined = combined.reset_index().rename(columns={"index": "Model", "Average Precision": "Precision"})
        sns.barplot(data=combined, x="Precision", y="Model", ax=ax, palette="mako", hue="Model", dodge=False, legend=False)
        ax.set_title("Model Comparison by Average Precision")
        ax.set_xlabel("Average Precision Score")
        ax.set_ylabel("Model")

    def save_avg_precision_scores(self):
        self._save_plot(self.plot_avg_precision_scores, f"{self.report_name}_avg_precision_scores.png")

    def plot_group_comparison(self, ax):
        df = self.group_comparison_df.T.reset_index().melt(id_vars="index", var_name="Group", value_name="Score")
        sns.barplot(data=df, x="index", y="Score", hue="Group", ax=ax, palette="Set2")
        ax.set_title("Scaled vs. Unscaled Group Averages")
        ax.set_ylabel("Score")
        ax.set_xlabel("Metric")
        ax.legend(loc="lower right")

    def save_group_comparison(self):
        self._save_plot(self.plot_group_comparison, f"{self.report_name}_group_comparison.png")

    def plot_confusion_matrices(self, axes):
        all_matrices = {**self.scaled_conf_matrices, **self.unscaled_conf_matrices}
        for ax, (model_name, matrix) in zip(axes, all_matrices.items()):
            sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues",
                        xticklabels=["Pred 0", "Pred 1"],
                        yticklabels=self.yticklabels,
                        ax=ax)
            ax.set_title(model_name)

    def save_confusion_matrices(self):
        path = os.path.join(self.figures_path, f"{self.report_name}_confusion_matrices.png")
        ensure_dir(path)
        all_matrices = {**self.scaled_conf_matrices, **self.unscaled_conf_matrices}
        n = len(all_matrices)
        cols = 3
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
        axes = axes.flatten()
        self.plot_confusion_matrices(axes)
        for i in range(n, len(axes)):
            axes[i].axis("off")
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)

    def plot_roc_curves(self, ax):
        for model_name, probs in {**self.scaled_probs, **self.unscaled_probs}.items():
            fpr, tpr, _ = roc_curve(self.y_true, probs)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"{model_name} (AUC={roc_auc:.2f})")
        ax.plot([0, 1], [0, 1], "k--", label="Random")
        ax.set_title("ROC Curves")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right")

    def save_roc_curves(self):
        self._save_plot(self.plot_roc_curves, f"{self.report_name}_roc_curves.png")

    def plot_precision_recall_curves(self, ax):
        for model_name, probs in {**self.scaled_probs, **self.unscaled_probs}.items():
            precision, recall, _ = precision_recall_curve(self.y_true, probs)
            ax.plot(recall, precision, label=model_name)
        ax.set_title("Precision-Recall Curves")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend(loc="lower left")

    def save_precision_recall_curves(self):
        self._save_plot(self.plot_precision_recall_curves, f"{self.report_name}_precision_recall_curves.png")

    def export_metrics_to_excel(self):
        path = os.path.join(self.docs_path, f"{self.report_name}_model_metrics.xlsx")
        ensure_dir(path)
        with pd.ExcelWriter(path) as writer:
            self.scaled_metrics_df.to_excel(writer, sheet_name="Scaled Models")
            self.unscaled_metrics_df.to_excel(writer, sheet_name="Unscaled Models")
            self.group_comparison_df.to_excel(writer, sheet_name="Group Comparison")
            self.scaled_precision_df.to_excel(writer, sheet_name="Scaled Precision")
            self.unscaled_precision_df.to_excel(writer, sheet_name="Unscaled Precision")
        print(f"Metrics exported to {path}")

    def export_plots_to_pdf(self):
        path = os.path.join(self.docs_path, f"{self.report_name}_model_plots.pdf")
        ensure_dir(path)
        with PdfPages(path) as pdf:
            for plot_func in [
                self.plot_f1_scores,
                self.plot_avg_precision_scores,
                self.plot_group_comparison,
                self.plot_roc_curves,
                self.plot_precision_recall_curves
            ]:
                fig, ax = plt.subplots(figsize=(10, 6))
                plot_func(ax)
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

            all_matrices = {**self.scaled_conf_matrices, **self.unscaled_conf_matrices}
            n = len(all_matrices)
            cols = 3
            rows = (n + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
            axes = axes.flatten()
            self.plot_confusion_matrices(axes)
            for i in range(n, len(axes)):
                axes[i].axis("off")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        print(f"All plots exported to {path}")
