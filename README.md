# Modular Benchmarking Framework for Binary Classification

## Overview
        
This project provides a scalable, reusable benchmarking suite for evaluating binary classification models across diverse datasets. It supports automated training, evaluation, visualization, and reporting for both scaled and unscaled classifiers. The framework is designed to be dataset-agnostic, with robust error handling, dynamic reporting paths, and clear interpretability.

## Why It Matters

Binary classification problems — such as spam detection and fraud detection — are critical in real-world applications. However, model performance varies widely depending on data characteristics, preprocessing, and scaling. This framework helps:

- Identify the best-performing models for a given dataset
- Understand precision-recall tradeoffs in imbalanced settings
- Visualize and compare model behavior using confusion matrices, ROC and PR curves
- Support reproducible, extensible experimentation across domains

---

## How It Works

The pipeline is organized into modular components:

- `ClassifierGroupComparer`: Loads data, splits it, and benchmarks multiple classifiers (scaled vs. unscaled)
- `ModelVisualizer`: Generates plots and exports metrics for interpretation and reporting
- `KaggleDataLoader`: Downloads and prepares datasets from Kaggle
- MLflow integration for experiment tracking

Each model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- ROC and Precision-Recall curves

Skipped models (e.g., MultinomialNB on negative inputs) are tracked and reported.

---

## Supported Datasets

### 1. Email Spam Detection

**Source**: [Kaggle - Email Spam Classification](https://www.kaggle.com/datasets/venky73/spam-mails-dataset)

**Description**:
- 5,172 emails
- 3,002 columns (Email ID + 3,000 word counts + label)
- Target: `Prediction` (1 = spam, 0 = not spam)
- Moderate class imbalance (29% spam)

**Structure**:
- Each row represents an email
- Columns represent word frequencies
- Labels indicate spam or not spam

### 2. Credit Card Fraud Detection

**Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

**Description**:
- 284,807 transactions
- PCA-transformed features: `V1` to `V28`
- Additional features: `Time`, `Amount`
- Target: `Class` (1 = fraud, 0 = legitimate)
- Severe class imbalance (0.172% fraud)

**Note**: Due to the imbalance, AUPRC is preferred over accuracy. Confusion matrix accuracy alone is misleading.

---

## Results

### Spam Detection

- **Best Scaled Models**: LogisticRegression, MLPClassifier
- **Best Unscaled Models**: LightGBM, XGBoost
- **Precision-Recall Tradeoffs**: SVC is cautious (high precision, low recall); KNN is aggressive (high recall, low precision)

### Fraud Detection

- **Best Scaled Models**: KNN, MLPClassifier
- **Best Unscaled Models**: RandomForest, XGBoost
- **Precision-Recall Tradeoffs**: SVC avoids false alarms but misses fraud; GradientBoosting struggles with recall

Group-level comparisons and confusion matrix interpretations are included in the visual reports.

---

## File Structure

```bash
classifier-bench/
│
├── data/                         # Raw and processed datasets
│   ├── email_spam.csv
│   └── creditcard.csv
│
├── reports/                      # Generated reports and visualizations
│   ├── spam_detection/
│   │   ├── figures/              # Plots (F1 scores, ROC, PR curves, confusion matrices)
│   │   └── docs/                 # Excel and PDF exports
│   └── fraud_detection/
│       ├── figures/
│       └── docs/
│
├── src/                          # Core benchmarking and visualization modules
│   ├── comparer.py              # ClassifierGroupComparer: model evaluation and comparison
│   ├── visualizer.py            # ModelVisualizer: plotting and exporting results
│   ├── data_loader.py           # KaggleDataLoader: dataset download and preparation
│   └── utils.py                 # Utility functions (optional)
│
├── notebooks/                   # Jupyter notebooks for experimentation and analysis
│   ├── spam_analysis.ipynb
│   └── fraud_analysis.ipynb
│
├── mlruns/                      # MLflow experiment tracking (auto-generated)
│
├── README.md                    # Project overview and usage instructions
├── requirements.txt             # Python dependencies
└── .gitignore 
```
---

## How to Use

1. Clone the repository  
2. Install dependencies (`requirements.txt`)  
3. Download datasets using `KaggleDataLoader`  
4. Run benchmarking with `ClassifierGroupComparer`  
5. Visualize results with `ModelVisualizer`  
6. Track experiments using MLflow

```python
from src.comparer import ClassifierGroupComparer
from src.visualizer import ModelVisualizer

comparer = ClassifierGroupComparer(data_path="data/creditcard.csv", target_col="Class", drop_cols=["Time"])
comparer.load_data()
comparer.prepare_data()
results = comparer.compare_groups()

viz = ModelVisualizer(results, report_name="fraud_detection", yticklabels=["Legitimate", "Fraud"])
viz.save_f1_scores()
viz.save_group_comparison()
viz.save_confusion_matrices()
viz.save_roc_curves()
viz.save_precision_recall_curves()
viz.export_metrics_to_excel()
viz.export_plots_to_pdf()
# classifier-bench
# classifier-bench
# classifier-bench
