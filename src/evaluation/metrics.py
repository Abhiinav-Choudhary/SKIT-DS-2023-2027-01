"""Comprehensive Classification Evaluation and Visualization Module.

Computes clinical and statistical metrics:
  - Accuracy
  - Precision (Positive Predictive Value)
  - Recall (Sensitivity)
  - Specificity (True Negative Rate)
  - F1-Score
  - Balanced Accuracy
  - ROC-AUC
  - PR-AUC (Average Precision)
  - Confusion Matrix

Generates publication-quality confusion matrices, ROC curves, and PR curves.

Project: AI-Powered Health Risk Prediction and Monitoring System
Author: Abhiram (Lead AI/ML Engineer)
Date: 8 October 2026
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Calculate comprehensive classification metrics.

    Args:
      y_true: Ground truth binary labels (0 or 1).
      y_pred: Predicted binary labels (0 or 1).
      y_prob: Predicted positive-class probabilities (optional).

    Returns:
      Dictionary of performance metrics.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    metrics = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "specificity": round(float(specificity), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }

    if y_prob is not None:
        metrics["roc_auc"] = round(float(roc_auc_score(y_true, y_prob)), 4)
        metrics["pr_auc"] = round(float(average_precision_score(y_true, y_prob)), 4)
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None

    return metrics


def plot_confusion_matrices(
    results: Dict[str, Dict[str, Any]],
    output_path: Path,
):
    """Plot a 2x2 grid of confusion matrices for the baseline models."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    axes = axes.ravel()

    model_names = list(results.keys())
    for idx, name in enumerate(model_names):
        ax = axes[idx]
        cm_dict = results[name]["metrics"]["confusion_matrix"]
        cm = np.array([
            [cm_dict["tn"], cm_dict["fp"]],
            [cm_dict["fn"], cm_dict["tp"]],
        ])
        total = cm.sum()
        pcts = cm / total * 100

        annot = np.array([
            [f"TN: {cm[0, 0]:,}\n({pcts[0, 0]:.1f}%)", f"FP: {cm[0, 1]:,}\n({pcts[0, 1]:.1f}%)"],
            [f"FN: {cm[1, 0]:,}\n({pcts[1, 0]:.1f}%)", f"TP: {cm[1, 1]:,}\n({pcts[1, 1]:.1f}%)"],
        ])

        sns.heatmap(
            cm,
            annot=annot,
            fmt="",
            cmap="Blues",
            cbar=False,
            ax=ax,
            linewidths=1.5,
            linecolor="gray",
            annot_kws={"size": 11, "fontweight": "bold"},
        )

        acc = results[name]["metrics"]["accuracy"]
        rec = results[name]["metrics"]["recall"]
        prec = results[name]["metrics"]["precision"]
        f1 = results[name]["metrics"]["f1"]
        auc = results[name]["metrics"].get("roc_auc", 0.0)

        ax.set_title(
            f"{name}\nAcc: {acc*100:.1f}% | Rec: {rec*100:.1f}% | Prec: {prec*100:.1f}% | AUC: {auc:.4f}",
            fontsize=11.5,
            pad=10,
        )
        ax.set_xlabel("Predicted Diagnosis Label", fontsize=10.5)
        ax.set_ylabel("Actual Diagnosis Label", fontsize=10.5)
        ax.set_xticklabels(["0: No Diabetes", "1: Diabetes"], fontsize=10)
        ax.set_yticklabels(["0: No Diabetes", "1: Diabetes"], fontsize=10)

    plt.suptitle("Validation Confusion Matrices for Baseline Health Risk Classifiers", fontsize=14, y=0.99)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_roc_and_pr_curves(
    results: Dict[str, Dict[str, Any]],
    y_val: np.ndarray,
    output_path: Path,
):
    """Plot side-by-side ROC Curves and Precision-Recall Curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    baseline_prev = float(y_val.mean())

    for idx, (name, res) in enumerate(results.items()):
        y_prob = res["y_prob"]
        color = colors[idx % len(colors)]

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_val, y_prob)
        auc_val = res["metrics"]["roc_auc"]
        axes[0].plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.4f})", color=color, linewidth=2)

        # PR Curve
        precision, recall, _ = precision_recall_curve(y_val, y_prob)
        pr_auc_val = res["metrics"]["pr_auc"]
        axes[1].plot(recall, precision, label=f"{name} (PR-AUC = {pr_auc_val:.4f})", color=color, linewidth=2)

    # Reference lines
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.2, label="Random Guess (AUC=0.50)")
    axes[0].set_title("A. Receiver Operating Characteristic (ROC) Curves", fontsize=12, pad=12)
    axes[0].set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    axes[0].set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    axes[0].legend(loc="lower right", fontsize=9.5)
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.02])

    axes[1].axhline(baseline_prev, linestyle="--", color="gray", linewidth=1.2, label=f"Prevalence Baseline ({baseline_prev*100:.1f}%)")
    axes[1].set_title("B. Precision-Recall (PR) Curves (Imbalanced Metric)", fontsize=12, pad=12)
    axes[1].set_xlabel("Recall (Sensitivity)", fontsize=11)
    axes[1].set_ylabel("Precision (Positive Predictive Value)", fontsize=11)
    axes[1].legend(loc="upper right", fontsize=9.5)
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.02])

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
