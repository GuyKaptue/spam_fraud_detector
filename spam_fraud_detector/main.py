# spam_fraud_detector/main.py

import sys

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



class SpamFraudDetector:
    """
    BenchmarkRunner Class
    ---------------------
    Handles the complete benchmarking pipeline for a specified task ('spam' or 'fraud').

    Steps:
        1. Load and preprocess data
        2. Compare classifiers and display metrics
        3. Visualize and export results
    """

    def __init__(self, task: str, save_format: str = "csv"):
        self.task = task.lower()
        self.save_format = save_format
        self.results = None

        if self.task not in ["spam", "fraud"]:
            raise ValueError(f"Task '{self.task}' not recognized. Choose between 'spam' and 'fraud'.")

    def _load_data(self):
        """Load and save the dataset."""
        print(f"  [1/3] Downloading/Loading data for {self.task}...")
        loader = KaggleDataLoader(task=self.task)
        df = loader.download_load_and_save(save_as=self.save_format)
        print("  Data loaded successfully.")
        return df

    def _compare_classifiers(self, df):
        """Prepare, train, and evaluate classifiers."""
        print("\n  [2/3] Preparing and comparing classifiers...")
        comparer = ClassifierGroupComparer(task=self.task, df=df)
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
            report_name=f"{self.task}_detection",
            yticklabels=yticklabels
        )

        visualizer.save_f1_scores()
        visualizer.save_group_comparison()
        visualizer.save_confusion_matrices()
        visualizer.save_roc_curves()
        visualizer.save_precision_recall_curves()
        visualizer.export_metrics_to_excel()
        visualizer.export_plots_to_pdf()
        print("  Visualizations and exports completed.")

    def run(self):
        """Execute the full benchmark pipeline."""
        print(f"\n{'='*20} Starting benchmark for: {self.task.upper()} detection {'='*20}")

        try:
            df = self._load_data()
            self.results = self._compare_classifiers(df)
            self._display_results(self.results)
            self._visualize_and_export(self.results)

            print("\n" + "=" * 75)
            print(f"Benchmark successfully completed for: {self.task.upper()} detection")
            print("=" * 75)

        except Exception as e:
            print(f"Error during benchmark execution: {e}")
            sys.exit(1)



