"""Baseline Classical ML Training, Cross-Validation & Model Comparison.

Trains and compares the 4 canonical baseline models:
  1. Logistic Regression
  2. Decision Tree
  3. Random Forest
  4. Gradient Boosting (HistGradientBoostingClassifier)

Methodology:
  - Stratified 5-Fold Cross-Validation on the training partition (N = 160,631).
  - Validation evaluation on untouched validation partition (N = 34,421).
  - The final test partition (N = 34,422) remains strictly untouched.
  - Comprehensive metrics: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Confusion Matrices.

Project: AI-Powered Health Risk Prediction and Monitoring System
Author: Abhiram (Lead AI/ML Engineer)
Date: 8 October 2026
"""

import json
import shutil
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.models.baseline_models import get_baseline_models
from src.evaluation.metrics import evaluate_predictions, plot_confusion_matrices, plot_roc_and_pr_curves


def run_baseline_training():
    print("=" * 80)
    print("PHASE 8-11: BASELINE CLASSICAL ML MODELS & STRATIFIED CROSS-VALIDATION")
    print("=" * 80)

    # 1. Load Data
    print("\n[STEP 1] Ingesting Preprocessed Partitions...")
    train_path = PROJECT_ROOT / "data" / "processed" / "diabetes" / "diabetes_train.csv"
    val_path = PROJECT_ROOT / "data" / "processed" / "diabetes" / "diabetes_val.csv"
    test_path = PROJECT_ROOT / "data" / "processed" / "diabetes" / "diabetes_test.csv"

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    print(f"Train Cohort : {len(train_df):,} samples (Prevalence: {train_df['Diabetes_binary'].mean()*100:.2f}%)")
    print(f"Val Cohort   : {len(val_df):,} samples (Prevalence: {val_df['Diabetes_binary'].mean()*100:.2f}%)")
    print(f"Test Cohort  : {len(test_df):,} samples [Untouched for Final Evaluation]")

    # 2. Transform Features via Preprocessor Pipeline
    print("\n[STEP 2] Transforming Features using Fitted Scikit-Learn Pipeline...")
    pipeline_path = PROJECT_ROOT / "models" / "diabetes" / "diabetes_full_preprocessor.joblib"
    pipeline = joblib.load(pipeline_path)

    X_train_raw = train_df.drop(columns=["Diabetes_binary"])
    y_train = train_df["Diabetes_binary"].values

    X_val_raw = val_df.drop(columns=["Diabetes_binary"])
    y_val = val_df["Diabetes_binary"].values

    X_train = pipeline.transform(X_train_raw)
    X_val = pipeline.transform(X_val_raw)

    print(f"Transformed X_train: {X_train.shape} (27 features)")
    print(f"Transformed X_val  : {X_val.shape} (27 features)")

    # 3. Stratified 5-Fold Cross-Validation on Training Set
    print("\n[STEP 3] Executing Stratified 5-Fold Cross-Validation (Training Partition)...")
    models = get_baseline_models(random_state=42, class_weight="balanced")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    cv_results = {}
    for name, model in models.items():
        print(f"  Training {name} with 5-Fold Stratified CV...")
        t0 = time.time()
        scores = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        dur = time.time() - t0

        cv_results[name] = {
            "cv_accuracy_mean": float(scores["test_accuracy"].mean()),
            "cv_accuracy_std": float(scores["test_accuracy"].std()),
            "cv_precision_mean": float(scores["test_precision"].mean()),
            "cv_precision_std": float(scores["test_precision"].std()),
            "cv_recall_mean": float(scores["test_recall"].mean()),
            "cv_recall_std": float(scores["test_recall"].std()),
            "cv_f1_mean": float(scores["test_f1"].mean()),
            "cv_f1_std": float(scores["test_f1"].std()),
            "cv_roc_auc_mean": float(scores["test_roc_auc"].mean()),
            "cv_roc_auc_std": float(scores["test_roc_auc"].std()),
            "training_time_seconds": round(dur, 2),
        }
        res = cv_results[name]
        print(f"    -> ROC-AUC: {res['cv_roc_auc_mean']:.4f} ± {res['cv_roc_auc_std']:.4f} | "
              f"Recall: {res['cv_recall_mean']:.4f} ± {res['cv_recall_std']:.4f} | "
              f"F1: {res['cv_f1_mean']:.4f} ± {res['cv_f1_std']:.4f} | "
              f"Acc: {res['cv_accuracy_mean']:.4f} ({dur:.1f}s)")

    # 4. Train Models on Full Training Set and Evaluate on Validation Set
    print("\n[STEP 4] Fitting Models on Full Training Set & Evaluating on Validation Partition...")
    val_evaluation = {}
    fitted_models = {}

    baselines_dir = PROJECT_ROOT / "models" / "diabetes" / "baselines"
    baselines_dir.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        print(f"  Fitting {name}...")
        model.fit(X_train, y_train)
        fitted_models[name] = model

        # Predict classes and probabilities
        y_val_pred = model.predict(X_val)
        y_val_prob = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else None

        metrics = evaluate_predictions(y_val, y_val_pred, y_val_prob)
        val_evaluation[name] = {
            "metrics": metrics,
            "y_prob": y_val_prob,
        }

        # Save model artifact
        slug = name.lower().replace(" ", "_")
        model_save_path = baselines_dir / f"{slug}_baseline.joblib"
        joblib.dump(model, model_save_path)

    # 5. Generate Figures (Confusion Matrices, ROC Curves, PR Curves)
    print("\n[STEP 5] Generating Validation Confusion Matrices and ROC/PR Curves...")
    figures_dir = PROJECT_ROOT / "reports" / "figures"
    cm_plot_path = figures_dir / "06_baseline_confusion_matrices.png"
    roc_plot_path = figures_dir / "07_baseline_roc_and_pr_curves.png"

    plot_confusion_matrices(val_evaluation, cm_plot_path)
    plot_roc_and_pr_curves(val_evaluation, y_val, roc_plot_path)

    # Copy to artifact directory if present
    artifact_fig_dir = Path(r"C:\Users\Admin\.gemini\antigravity-ide\brain\3a9de9f1-a070-4784-8aa5-9d00085d28a1\figures")
    if artifact_fig_dir.exists():
        shutil.copy(cm_plot_path, artifact_fig_dir / cm_plot_path.name)
        shutil.copy(roc_plot_path, artifact_fig_dir / roc_plot_path.name)
        print(f"  -> Exported figures to artifact directory: {artifact_fig_dir}")

    # 6. Print Formatted Comparison Tables
    print("\n" + "=" * 95)
    print("STRATIFIED 5-FOLD CROSS-VALIDATION RESULTS (TRAIN SET: N = 160,631)")
    print("=" * 95)
    print(f"{'Model':<22} | {'ROC-AUC':<18} | {'Recall':<18} | {'Precision':<18} | {'F1-Score':<18} | {'Accuracy':<10}")
    print("-" * 115)
    for name, r in cv_results.items():
        auc_str = f"{r['cv_roc_auc_mean']:.4f} ± {r['cv_roc_auc_std']:.4f}"
        rec_str = f"{r['cv_recall_mean']:.4f} ± {r['cv_recall_std']:.4f}"
        prec_str = f"{r['cv_precision_mean']:.4f} ± {r['cv_precision_std']:.4f}"
        f1_str = f"{r['cv_f1_mean']:.4f} ± {r['cv_f1_std']:.4f}"
        acc_str = f"{r['cv_accuracy_mean']:.4f}"
        print(f"{name:<22} | {auc_str:<18} | {rec_str:<18} | {prec_str:<18} | {f1_str:<18} | {acc_str:<10}")
    print("=" * 115)

    print("\n" + "=" * 115)
    print("VALIDATION SET EVALUATION RESULTS (N = 34,421, UNTOILED DURING TRAINING)")
    print("=" * 115)
    print(f"{'Model':<22} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1':<10} | {'ROC-AUC':<10} | {'PR-AUC':<10} | {'TN':<7} | {'FP':<7} | {'FN':<7} | {'TP':<7}")
    print("-" * 115)
    for name, r in val_evaluation.items():
        m = r["metrics"]
        cm = m["confusion_matrix"]
        print(f"{name:<22} | {m['accuracy']:<10.4f} | {m['precision']:<10.4f} | {m['recall']:<10.4f} | {m['f1']:<10.4f} | {m['roc_auc']:<10.4f} | {m['pr_auc']:<10.4f} | {cm['tn']:<7} | {cm['fp']:<7} | {cm['fn']:<7} | {cm['tp']:<7}")
    print("=" * 115)

    # 7. Persist JSON Comparison Report
    report_data = {
        "evaluation_partition": "Stratified 5-Fold CV on Train (N=160,631) & Validation (N=34,421)",
        "cross_validation_results": cv_results,
        "validation_results": {
            name: r["metrics"] for name, r in val_evaluation.items()
        },
        "figures": [
            "06_baseline_confusion_matrices.png",
            "07_baseline_roc_and_pr_curves.png",
        ],
    }
    report_path = PROJECT_ROOT / "reports" / "baseline_model_comparison.json"
    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=2)
    print(f"\n[OK] Model comparison report saved to: {report_path.relative_to(PROJECT_ROOT)}")
    print("=" * 80)


if __name__ == "__main__":
    run_baseline_training()
