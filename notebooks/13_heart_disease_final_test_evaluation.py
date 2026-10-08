"""Final Heart Disease test evaluation.

Loads the selected tuned Logistic Regression pipeline from Step 7 and evaluates
it once on the untouched Heart Disease test split. This script does not retrain
or retune the model.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import joblib  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.preprocessing.heart_disease import HEART_DISEASE_TARGET, SOURCE_COLUMN  # noqa: E402


PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "heart_disease"
MODEL_PATH = PROJECT_ROOT / "models" / "heart" / "tuned" / "heart_disease_selected_tuned_pipeline.joblib"
TUNING_REPORT_PATH = PROJECT_ROOT / "reports" / "heart_disease_hyperparameter_tuning_report.json"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
CLASS_LABELS = [0, 1, 2, 3, 4]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)
    if pd.isna(value):
        return None
    return value


def _load_test_split() -> pd.DataFrame:
    path = PROCESSED_DIR / "heart_disease_test.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing Heart Disease test split: {path}")
    return pd.read_csv(path)


def _split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    X = df.drop(columns=[HEART_DISEASE_TARGET, SOURCE_COLUMN], errors="ignore")
    y = df[HEART_DISEASE_TARGET].to_numpy()
    return X, y


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision_macro": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "precision_weighted": round(float(precision_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "recall_macro": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "recall_weighted": round(float(recall_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "f1_weighted": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=CLASS_LABELS).astype(int).tolist(),
        "per_class": classification_report(
            y_true,
            y_pred,
            labels=CLASS_LABELS,
            target_names=[str(label) for label in CLASS_LABELS],
            output_dict=True,
            zero_division=0,
        ),
    }

    if y_prob is not None:
        try:
            metrics["roc_auc_ovr_macro"] = round(
                float(roc_auc_score(y_true, y_prob, labels=CLASS_LABELS, multi_class="ovr", average="macro")),
                4,
            )
            metrics["roc_auc_ovr_weighted"] = round(
                float(roc_auc_score(y_true, y_prob, labels=CLASS_LABELS, multi_class="ovr", average="weighted")),
                4,
            )
        except ValueError as exc:
            metrics["roc_auc_ovr_macro"] = None
            metrics["roc_auc_ovr_weighted"] = None
            metrics["roc_auc_error"] = str(exc)
    else:
        metrics["roc_auc_ovr_macro"] = None
        metrics["roc_auc_ovr_weighted"] = None
        metrics["roc_auc_error"] = "predict_proba unavailable"

    return _json_safe(metrics)


def _align_probabilities(model: Any, probabilities: np.ndarray) -> np.ndarray:
    classes = list(model.named_steps["clf"].classes_)
    if classes == CLASS_LABELS:
        return probabilities

    aligned = np.zeros((probabilities.shape[0], len(CLASS_LABELS)), dtype=float)
    for column_index, class_label in enumerate(classes):
        if int(class_label) in CLASS_LABELS:
            aligned[:, CLASS_LABELS.index(int(class_label))] = probabilities[:, column_index]
    return aligned


def _compare_with_cv(test_metrics: dict[str, Any], cv_metrics: dict[str, Any]) -> dict[str, Any]:
    pairs = {
        "accuracy": "best_cv_accuracy",
        "precision_macro": "best_cv_precision_macro",
        "precision_weighted": "best_cv_precision_weighted",
        "recall_macro": "best_cv_recall_macro",
        "recall_weighted": "best_cv_recall_weighted",
        "f1_macro": "best_cv_f1_macro",
        "f1_weighted": "best_cv_f1_weighted",
        "roc_auc_ovr_macro": "best_cv_roc_auc_ovr_macro",
        "roc_auc_ovr_weighted": "best_cv_roc_auc_ovr_weighted",
    }
    comparison = {}
    for test_key, cv_key in pairs.items():
        test_value = test_metrics.get(test_key)
        cv_value = cv_metrics.get(cv_key)
        if test_value is None or cv_value is None:
            comparison[test_key] = {"training_cv": cv_value, "test": test_value, "delta_test_minus_cv": None}
        else:
            comparison[test_key] = {
                "training_cv": cv_value,
                "test": test_value,
                "delta_test_minus_cv": round(float(test_value) - float(cv_value), 4),
            }
    return comparison


def _plot_confusion_matrix(metrics: dict[str, Any], output_path: Path) -> None:
    cm = np.array(metrics["confusion_matrix"])
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        linewidths=0.5,
        linecolor="gray",
        xticklabels=CLASS_LABELS,
        yticklabels=CLASS_LABELS,
    )
    plt.title(
        "Final Heart Disease Test Confusion Matrix\n"
        f"Accuracy={metrics['accuracy']:.4f} | F1 weighted={metrics['f1_weighted']:.4f} | "
        f"AUC weighted={metrics['roc_auc_ovr_weighted']}",
        fontsize=11,
    )
    plt.xlabel("Predicted num")
    plt.ylabel("Actual num")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def _write_markdown_report(path: Path, report: dict[str, Any]) -> None:
    metrics = report["test_metrics"]
    lines = [
        "# Heart Disease Final Test Evaluation Report",
        "",
        f"Target: {report['target_definition']}",
        "",
        "The selected tuned pipeline was loaded from disk and evaluated once on the test split. No retraining or retuning was performed.",
        "",
        "## Test Metrics",
        "",
        f"- Accuracy: {metrics['accuracy']:.4f}",
        f"- Precision macro: {metrics['precision_macro']:.4f}",
        f"- Precision weighted: {metrics['precision_weighted']:.4f}",
        f"- Recall macro: {metrics['recall_macro']:.4f}",
        f"- Recall weighted: {metrics['recall_weighted']:.4f}",
        f"- F1 macro: {metrics['f1_macro']:.4f}",
        f"- F1 weighted: {metrics['f1_weighted']:.4f}",
        f"- ROC-AUC OVR macro: {metrics['roc_auc_ovr_macro']}",
        f"- ROC-AUC OVR weighted: {metrics['roc_auc_ovr_weighted']}",
        "",
        "## Confusion Matrix",
        "",
        "Class order: `[0, 1, 2, 3, 4]`",
        "",
        "```text",
        str(metrics["confusion_matrix"]),
        "```",
        "",
        "## Training CV vs Test",
        "",
        "| Metric | Training CV | Test | Delta test-CV |",
        "| --- | ---: | ---: | ---: |",
    ]
    for metric, values in report["training_cv_vs_test"].items():
        lines.append(
            f"| {metric} | {values['training_cv']} | {values['test']} | {values['delta_test_minus_cv']} |"
        )

    lines.extend(["", "## Per-Class Results", ""])
    lines.append("| Class | Precision | Recall | F1 | Support |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for label in [str(value) for value in CLASS_LABELS]:
        class_metrics = metrics["per_class"][label]
        lines.append(
            f"| {label} | {class_metrics['precision']:.4f} | {class_metrics['recall']:.4f} | "
            f"{class_metrics['f1-score']:.4f} | {int(class_metrics['support'])} |"
        )

    lines.extend(["", "## Files", ""])
    for output in report["files_created"]:
        lines.append(f"- `{output}`")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def run_final_test_evaluation() -> None:
    print("=" * 80)
    print("STEP 8: HEART DISEASE FINAL TEST EVALUATION")
    print("=" * 80)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Selected tuned Heart pipeline not found: {MODEL_PATH}")
    if not TUNING_REPORT_PATH.exists():
        raise FileNotFoundError(f"Heart tuning report not found: {TUNING_REPORT_PATH}")

    with open(TUNING_REPORT_PATH, encoding="utf-8") as f:
        tuning_report = json.load(f)

    selected_name = tuning_report["selected_model"]["model"]
    if selected_name != "Logistic Regression":
        raise ValueError(f"Expected selected Logistic Regression pipeline, found: {selected_name}")

    model = joblib.load(MODEL_PATH)
    test_df = _load_test_split()
    X_test, y_test = _split_xy(test_df)

    print(f"Loaded selected model: {MODEL_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Selected model from tuning report: {selected_name}")
    print(f"Loaded test split: {test_df.shape}; target counts={pd.Series(y_test).value_counts().sort_index().to_dict()}")
    print("No fit/refit/search is called in this script.")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
    if y_prob is not None:
        y_prob = _align_probabilities(model, y_prob)

    metrics = _classification_metrics(y_test, y_pred, y_prob)
    cv_metrics = tuning_report["selected_model"]["cv_metrics"]
    cv_comparison = _compare_with_cv(metrics, cv_metrics)

    figure_path = FIGURES_DIR / "17_heart_final_test_confusion_matrix.png"
    _plot_confusion_matrix(metrics, figure_path)

    report = {
        "target_definition": "Existing multiclass `num` target with classes [0, 1, 2, 3, 4].",
        "model": {
            "selected_model": selected_name,
            "artifact": str(MODEL_PATH.relative_to(PROJECT_ROOT)),
            "loaded_pipeline_steps": list(model.named_steps.keys()) if hasattr(model, "named_steps") else None,
            "classifier_classes": [int(value) for value in model.named_steps["clf"].classes_],
        },
        "data": {
            "test_shape": [int(test_df.shape[0]), int(test_df.shape[1])],
            "test_target_counts": {int(k): int(v) for k, v in pd.Series(y_test).value_counts().sort_index().items()},
        },
        "test_metrics": metrics,
        "training_cv_metrics_from_tuning_report": cv_metrics,
        "training_cv_vs_test": cv_comparison,
        "model_update_from_test_results": "None. Test results were used for final evaluation only.",
        "files_created": [
            str(figure_path.relative_to(PROJECT_ROOT)),
            "reports/heart_disease_final_test_evaluation_report.json",
            "reports/heart_disease_final_test_evaluation_report.md",
        ],
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORTS_DIR / "heart_disease_final_test_evaluation_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(report), f, indent=2)

    markdown_path = REPORTS_DIR / "heart_disease_final_test_evaluation_report.md"
    _write_markdown_report(markdown_path, _json_safe(report))

    print("\n" + "=" * 96)
    print("FINAL TEST RESULTS")
    print("=" * 96)
    print(f"Accuracy             : {metrics['accuracy']:.4f}")
    print(f"Precision macro      : {metrics['precision_macro']:.4f}")
    print(f"Precision weighted   : {metrics['precision_weighted']:.4f}")
    print(f"Recall macro         : {metrics['recall_macro']:.4f}")
    print(f"Recall weighted      : {metrics['recall_weighted']:.4f}")
    print(f"F1 macro             : {metrics['f1_macro']:.4f}")
    print(f"F1 weighted          : {metrics['f1_weighted']:.4f}")
    print(f"ROC-AUC OVR macro    : {metrics['roc_auc_ovr_macro']}")
    print(f"ROC-AUC OVR weighted : {metrics['roc_auc_ovr_weighted']}")
    print(f"Confusion matrix     : {metrics['confusion_matrix']}")
    print("\nPer-class results:")
    for label in [str(value) for value in CLASS_LABELS]:
        item = metrics["per_class"][label]
        print(
            f"  class {label}: precision={item['precision']:.4f}, recall={item['recall']:.4f}, "
            f"f1={item['f1-score']:.4f}, support={int(item['support'])}"
        )
    print("\nTraining CV vs test:")
    for metric_name, values in cv_comparison.items():
        print(
            f"  {metric_name}: cv={values['training_cv']}, test={values['test']}, "
            f"delta={values['delta_test_minus_cv']}"
        )
    print(f"\n[OK] JSON report saved to: {json_path.relative_to(PROJECT_ROOT)}")
    print(f"[OK] Markdown report saved to: {markdown_path.relative_to(PROJECT_ROOT)}")
    print(f"[OK] Confusion matrix figure saved to: {figure_path.relative_to(PROJECT_ROOT)}")
    print("No retraining or retuning was performed.")
    print("=" * 80)


if __name__ == "__main__":
    run_final_test_evaluation()
