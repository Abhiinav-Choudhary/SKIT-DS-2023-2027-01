"""Hyperparameter Tuning & Final Evaluation Script for CDC Diabetes.

Tunes Gradient Boosting and Random Forest using RandomizedSearchCV strictly on the
training partition. Preprocessing is nested inside each CV fold via sklearn Pipeline.
The test partition is evaluated once after model selection by train CV score.

Project: AI-Powered Health Risk Prediction and Monitoring System
Author: Abhiram (Lead AI/ML Engineer)
Date: 8 October 2026
"""

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)

from src.models.hyperparameter_tuning import (
    tune_gradient_boosting,
    tune_random_forest,
    select_champion_by_cv_score,
)
from src.evaluation.metrics import evaluate_predictions

TARGET_COL = "Diabetes_binary"


def plot_test_confusion_matrix(
    cm_dict: dict,
    metrics: dict,
    model_name: str,
    output_path: Path,
):
    """Plot publication-quality confusion matrix for final test evaluation."""
    cm = np.array(
        [
            [cm_dict["tn"], cm_dict["fp"]],
            [cm_dict["fn"], cm_dict["tp"]],
        ]
    )
    total = cm.sum()
    pcts = cm / total * 100

    annot = np.array(
        [
            [
                f"True Negative (TN)\n{cm[0, 0]:,} ({pcts[0, 0]:.1f}%)",
                f"False Positive (FP)\n{cm[0, 1]:,} ({pcts[0, 1]:.1f}%)",
            ],
            [
                f"False Negative (FN)\n{cm[1, 0]:,} ({pcts[1, 0]:.1f}%)",
                f"True Positive (TP)\n{cm[1, 1]:,} ({pcts[1, 1]:.1f}%)",
            ],
        ]
    )

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        cbar=False,
        linewidths=1.5,
        linecolor="gray",
        annot_kws={"size": 11, "fontweight": "bold"},
    )
    plt.title(
        f"Final Test Evaluation: {model_name} (N = {total:,})\n"
        f"Accuracy: {metrics['accuracy'] * 100:.1f}% | Recall: {metrics['recall'] * 100:.1f}% | "
        f"F1: {metrics['f1']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f}",
        fontsize=11.5,
        pad=12,
    )
    plt.xlabel("Predicted Risk Category", fontsize=11)
    plt.ylabel("Actual Clinical Diagnosis", fontsize=11)
    plt.xticks([0.5, 1.5], ["0: No Diabetes", "1: Diabetes"], fontsize=10.5)
    plt.yticks([0.5, 1.5], ["0: No Diabetes", "1: Diabetes"], fontsize=10.5)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_test_roc_and_pr(
    y_test: np.ndarray,
    y_prob: np.ndarray,
    metrics: dict,
    model_name: str,
    output_path: Path,
):
    """Plot side-by-side ROC and PR curves for the final test set."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    baseline_prev = float(y_test.mean())

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    axes[0].plot(
        fpr,
        tpr,
        color="#d95f02",
        linewidth=2.5,
        label=f"{model_name} (AUC = {metrics['roc_auc']:.4f})",
    )
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.2, label="Random (AUC = 0.50)")
    axes[0].set_title(f"A. Test Set ROC Curve (N = {len(y_test):,})", fontsize=12, pad=10)
    axes[0].set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    axes[0].set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    axes[0].legend(loc="lower right", fontsize=10)
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.02])

    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    axes[1].plot(
        rec,
        prec,
        color="#2b5c8f",
        linewidth=2.5,
        label=f"{model_name} (PR-AUC = {metrics['pr_auc']:.4f})",
    )
    axes[1].axhline(
        baseline_prev,
        linestyle="--",
        color="gray",
        linewidth=1.2,
        label=f"Baseline Prevalence ({baseline_prev * 100:.1f}%)",
    )
    axes[1].set_title(
        f"B. Test Set Precision-Recall Curve (PR-AUC: {metrics['pr_auc']:.4f})",
        fontsize=12,
        pad=10,
    )
    axes[1].set_xlabel("Recall (Sensitivity)", fontsize=11)
    axes[1].set_ylabel("Precision (Positive Predictive Value)", fontsize=11)
    axes[1].legend(loc="upper right", fontsize=10)
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.02])

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def run_tuning_and_final_evaluation():
    print("=" * 80)
    print("STEP 10: NESTED-PIPELINE HYPERPARAMETER TUNING & FINAL TEST EVALUATION")
    print("=" * 80)

    print("\n[STEP 1] Loading partitions (raw features, no pre-transform)...")
    train_path = PROJECT_ROOT / "data" / "processed" / "diabetes" / "diabetes_train.csv"
    test_path = PROJECT_ROOT / "data" / "processed" / "diabetes" / "diabetes_test.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]
    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL].values

    print(f"Train cohort: {len(train_df):,} samples (tuning only)")
    print(f"Test cohort : {len(test_df):,} samples (held out until final evaluation)")

    print("\n[STEP 2] RandomizedSearchCV on train (StratifiedKFold, preprocessing inside Pipeline)...")
    t0 = time.time()
    gb_pipeline, gb_summary = tune_gradient_boosting(
        X_train, y_train, n_iter=15, cv_splits=5, random_state=42
    )
    gb_dur = time.time() - t0
    gb_summary["training_time_seconds"] = round(gb_dur, 2)
    print(f"  Gradient Boosting best CV ROC-AUC: {gb_summary['best_score_cv_roc_auc']:.4f} ({gb_dur:.1f}s)")
    print(f"  best_params_: {gb_summary['best_params']}")

    t0 = time.time()
    rf_pipeline, rf_summary = tune_random_forest(
        X_train, y_train, n_iter=8, cv_splits=5, random_state=42
    )
    rf_dur = time.time() - t0
    rf_summary["training_time_seconds"] = round(rf_dur, 2)
    print(f"  Random Forest best CV ROC-AUC: {rf_summary['best_score_cv_roc_auc']:.4f} ({rf_dur:.1f}s)")
    print(f"  best_params_: {rf_summary['best_params']}")

    print("\n[STEP 3] Selecting champion by highest train CV ROC-AUC...")
    champion_name, champion_pipeline, champion_summary = select_champion_by_cv_score(
        {
            "Gradient Boosting": (gb_pipeline, gb_summary),
            "Random Forest": (rf_pipeline, rf_summary),
        }
    )
    print(f"  Champion: {champion_name}")
    print(f"  CV ROC-AUC: {champion_summary['best_score_cv_roc_auc']:.4f}")

    print("\n[STEP 4] Final evaluation on untouched test set...")
    y_test_pred = champion_pipeline.predict(X_test)
    y_test_prob = champion_pipeline.predict_proba(X_test)[:, 1]
    test_metrics = evaluate_predictions(y_test, y_test_pred, y_test_prob)

    print("\n" + "=" * 80)
    print(f"FINAL TEST SET PERFORMANCE — {champion_name} (N = {len(test_df):,})")
    print("=" * 80)
    print(f"Accuracy  : {test_metrics['accuracy']:.4f}")
    print(f"Precision : {test_metrics['precision']:.4f}")
    print(f"Recall    : {test_metrics['recall']:.4f}")
    print(f"F1        : {test_metrics['f1']:.4f}")
    print(f"ROC-AUC   : {test_metrics['roc_auc']:.4f}")
    cm = test_metrics["confusion_matrix"]
    print("\nConfusion matrix:")
    print(f"  TN={cm['tn']:,}  FP={cm['fp']:,}")
    print(f"  FN={cm['fn']:,}  TP={cm['tp']:,}")
    print("=" * 80)

    figures_dir = PROJECT_ROOT / "reports" / "figures"
    plot_test_confusion_matrix(
        cm,
        test_metrics,
        champion_name,
        figures_dir / "08_final_test_confusion_matrix.png",
    )
    plot_test_roc_and_pr(
        y_test,
        y_test_prob,
        test_metrics,
        champion_name,
        figures_dir / "09_final_test_roc_and_pr_curves.png",
    )

    models_dir = PROJECT_ROOT / "models" / "diabetes"
    models_dir.mkdir(parents=True, exist_ok=True)
    pipeline_save_path = models_dir / "diabetes_risk_pipeline.joblib"
    joblib.dump(champion_pipeline, pipeline_save_path)

    preprocess_step = champion_pipeline.named_steps["preprocess"]
    feature_names = preprocess_step.named_steps["preprocessor"].get_feature_names_out()

    metadata = {
        "model_name": champion_name,
        "dataset": "CDC Diabetes Health Indicators (BRFSS 2015)",
        "training_date": "2026-10-08",
        "training_samples": len(train_df),
        "test_samples": len(test_df),
        "tuning_method": "RandomizedSearchCV + StratifiedKFold (preprocess nested in Pipeline)",
        "champion_selection": "Highest train CV ROC-AUC between tuned Gradient Boosting and Random Forest",
        "best_hyperparameters": champion_summary["best_params_classifier_only"],
        "best_score_cv_roc_auc": champion_summary["best_score_cv_roc_auc"],
        "final_test_metrics": test_metrics,
        "features": list(feature_names),
    }
    with open(models_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    report_data = {
        "methodology": {
            "search": "RandomizedSearchCV",
            "cv": "StratifiedKFold (5 splits)",
            "scoring": "roc_auc",
            "train_only_tuning": True,
            "preprocessing_in_pipeline": True,
            "test_set_used_for": "single final evaluation only",
        },
        "gb_tuning": gb_summary,
        "rf_tuning": rf_summary,
        "champion": {
            "name": champion_name,
            "best_score_cv_roc_auc": champion_summary["best_score_cv_roc_auc"],
            "best_params": champion_summary["best_params"],
        },
        "final_test_evaluation": test_metrics,
    }
    report_path = PROJECT_ROOT / "reports" / "hyperparameter_tuning_and_final_evaluation.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print(f"\nSaved pipeline: {pipeline_save_path.relative_to(PROJECT_ROOT)}")
    print(f"Saved report  : {report_path.relative_to(PROJECT_ROOT)}")
    print("=" * 80)


if __name__ == "__main__":
    run_tuning_and_final_evaluation()
