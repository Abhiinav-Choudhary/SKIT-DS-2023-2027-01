"""Heart Disease baseline model training and validation.

Trains four baseline classifiers for the existing multiclass `num` target using
the Heart Disease feature-engineering/preprocessing pipeline. The test split is
not loaded or evaluated in this step.
"""

from __future__ import annotations

import json
import os
import sys
import time
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
from sklearn.base import clone  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

from src.preprocessing.heart_disease import (  # noqa: E402
    HEART_DISEASE_TARGET,
    SOURCE_COLUMN,
    create_heart_disease_feature_pipeline,
)


PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "heart_disease"
MODELS_DIR = PROJECT_ROOT / "models" / "heart" / "baselines"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
CLASS_LABELS = [0, 1, 2, 3, 4]


def get_heart_baseline_models(random_state: int = 42) -> Dict[str, Any]:
    """Return the four required baseline classifiers with fixed default settings."""
    return {
        "Logistic Regression": LogisticRegression(
            C=1.0,
            penalty="l2",
            solver="lbfgs",
            max_iter=2000,
            class_weight="balanced",
            random_state=random_state,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=14,
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight="balanced",
            n_jobs=1,
            random_state=random_state,
        ),
        "Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=100,
            max_depth=8,
            learning_rate=0.1,
            class_weight="balanced",
            random_state=random_state,
        ),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if pd.isna(value):
        return None
    return value


def _load_split(name: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"heart_disease_{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing Heart Disease {name} split: {path}")
    return pd.read_csv(path)


def _split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    X = df.drop(columns=[HEART_DISEASE_TARGET, SOURCE_COLUMN], errors="ignore")
    y = df[HEART_DISEASE_TARGET].to_numpy()
    return X, y


def _make_model_pipeline(classifier: Any) -> Pipeline:
    return Pipeline(
        steps=[
            ("features", create_heart_disease_feature_pipeline()),
            ("clf", classifier),
        ]
    )


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

    return metrics


def _summarize_cv(scores: dict[str, np.ndarray], duration_seconds: float) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"training_time_seconds": round(float(duration_seconds), 2)}
    metric_names = [
        "accuracy",
        "precision_macro",
        "precision_weighted",
        "recall_macro",
        "recall_weighted",
        "f1_macro",
        "f1_weighted",
        "roc_auc_ovr_macro",
        "roc_auc_ovr_weighted",
    ]
    for metric in metric_names:
        key = f"test_{metric}"
        values = scores[key]
        summary[f"cv_{metric}_mean"] = round(float(np.nanmean(values)), 4)
        summary[f"cv_{metric}_std"] = round(float(np.nanstd(values)), 4)
    return summary


def _plot_validation_confusion_matrices(validation_results: dict[str, dict[str, Any]], output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    for ax, (name, result) in zip(axes, validation_results.items()):
        cm = np.array(result["metrics"]["confusion_matrix"])
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
            ax=ax,
        )
        metrics = result["metrics"]
        ax.set_title(
            f"{name}\nAcc={metrics['accuracy']:.4f} | F1w={metrics['f1_weighted']:.4f} | "
            f"AUCw={metrics['roc_auc_ovr_weighted']}",
            fontsize=10,
        )
        ax.set_xlabel("Predicted num")
        ax.set_ylabel("Actual num")

    plt.suptitle("Heart Disease Validation Confusion Matrices (`num` target 0-4)", y=0.995)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def _write_markdown_report(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Heart Disease Baseline Model Report",
        "",
        f"Target: {report['target_definition']}",
        "",
        "The test split was not loaded or evaluated in this step.",
        "",
        "## Strongest Candidate",
        "",
        f"`{report['strongest_candidate']['model']}` selected by highest validation weighted F1, "
        "with validation weighted ROC-AUC as the first tie-breaker.",
        "",
        "## Validation Results",
        "",
        "| Model | Accuracy | Precision Macro | Recall Macro | F1 Macro | F1 Weighted | ROC-AUC OVR Weighted |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, result in report["validation_results"].items():
        metrics = result["metrics"]
        lines.append(
            f"| {name} | {metrics['accuracy']:.4f} | {metrics['precision_macro']:.4f} | "
            f"{metrics['recall_macro']:.4f} | {metrics['f1_macro']:.4f} | "
            f"{metrics['f1_weighted']:.4f} | {metrics['roc_auc_ovr_weighted']} |"
        )

    lines.extend(["", "## Cross-Validation Results", ""])
    lines.extend(
        [
            "| Model | Accuracy Mean | F1 Macro Mean | F1 Weighted Mean | ROC-AUC OVR Weighted Mean |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, result in report["cross_validation_results"].items():
        lines.append(
            f"| {name} | {result['cv_accuracy_mean']:.4f} | {result['cv_f1_macro_mean']:.4f} | "
            f"{result['cv_f1_weighted_mean']:.4f} | {result['cv_roc_auc_ovr_weighted_mean']:.4f} |"
        )

    lines.extend(["", "## Files", ""])
    for output in report["files_created"]:
        lines.append(f"- `{output}`")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def run_heart_baselines() -> None:
    print("=" * 80)
    print("STEP 6: HEART DISEASE BASELINE MODELS")
    print("=" * 80)

    train_df = _load_split("train")
    val_df = _load_split("val")
    X_train, y_train = _split_xy(train_df)
    X_val, y_val = _split_xy(val_df)

    print(f"Loaded train split: {train_df.shape}; target counts={pd.Series(y_train).value_counts().sort_index().to_dict()}")
    print(f"Loaded validation split: {val_df.shape}; target counts={pd.Series(y_val).value_counts().sort_index().to_dict()}")
    print("Test split was intentionally not loaded or evaluated.")

    models = get_heart_baseline_models(random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "accuracy": "accuracy",
        "precision_macro": "precision_macro",
        "precision_weighted": "precision_weighted",
        "recall_macro": "recall_macro",
        "recall_weighted": "recall_weighted",
        "f1_macro": "f1_macro",
        "f1_weighted": "f1_weighted",
        "roc_auc_ovr_macro": "roc_auc_ovr",
        "roc_auc_ovr_weighted": "roc_auc_ovr_weighted",
    }

    cv_results: Dict[str, Any] = {}
    validation_results: Dict[str, Any] = {}
    files_created: list[str] = []

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[STEP 1] Stratified 5-fold cross-validation on training split...")
    for name, classifier in models.items():
        print(f"  CV: {name}")
        model_pipeline = _make_model_pipeline(clone(classifier))
        start = time.time()
        scores = cross_validate(
            model_pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            error_score="raise",
        )
        duration = time.time() - start
        cv_results[name] = _summarize_cv(scores, duration)
        print(
            f"    accuracy={cv_results[name]['cv_accuracy_mean']:.4f} "
            f"f1_weighted={cv_results[name]['cv_f1_weighted_mean']:.4f} "
            f"roc_auc_ovr_weighted={cv_results[name]['cv_roc_auc_ovr_weighted_mean']:.4f}"
        )

    print("\n[STEP 2] Fit on full training split and evaluate validation split...")
    for name, classifier in models.items():
        print(f"  Fit/evaluate: {name}")
        model_pipeline = _make_model_pipeline(clone(classifier))
        model_pipeline.fit(X_train, y_train)
        y_pred = model_pipeline.predict(X_val)
        y_prob = model_pipeline.predict_proba(X_val) if hasattr(model_pipeline, "predict_proba") else None
        metrics = _classification_metrics(y_val, y_pred, y_prob)
        validation_results[name] = {"metrics": metrics}

        slug = name.lower().replace(" ", "_")
        model_path = MODELS_DIR / f"{slug}_baseline_pipeline.joblib"
        joblib.dump(model_pipeline, model_path)
        files_created.append(str(model_path.relative_to(PROJECT_ROOT)))
        print(
            f"    accuracy={metrics['accuracy']:.4f} precision_macro={metrics['precision_macro']:.4f} "
            f"recall_macro={metrics['recall_macro']:.4f} f1_weighted={metrics['f1_weighted']:.4f} "
            f"roc_auc_ovr_weighted={metrics['roc_auc_ovr_weighted']}"
        )

    strongest_name = max(
        validation_results,
        key=lambda model_name: (
            validation_results[model_name]["metrics"]["f1_weighted"],
            validation_results[model_name]["metrics"]["roc_auc_ovr_weighted"] or -np.inf,
            cv_results[model_name]["cv_f1_weighted_mean"],
        ),
    )

    cm_plot_path = FIGURES_DIR / "16_heart_baseline_validation_confusion_matrices.png"
    _plot_validation_confusion_matrices(validation_results, cm_plot_path)
    files_created.append(str(cm_plot_path.relative_to(PROJECT_ROOT)))

    report = {
        "target_definition": "Existing multiclass `num` target with classes 0, 1, 2, 3, 4.",
        "test_set_status": "Untouched: test split was not loaded or evaluated.",
        "methodology": {
            "feature_pipeline": "Existing Heart Disease feature-engineering/preprocessing pipeline inside each sklearn Pipeline.",
            "cross_validation": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42) on training split.",
            "validation": "Each baseline fitted on full training split, evaluated once on validation split.",
            "selection_rule": "Highest validation weighted F1; validation weighted ROC-AUC then CV weighted F1 used as tie-breakers.",
            "hyperparameter_tuning": False,
        },
        "data": {
            "train_shape": [int(train_df.shape[0]), int(train_df.shape[1])],
            "validation_shape": [int(val_df.shape[0]), int(val_df.shape[1])],
            "train_target_counts": {int(k): int(v) for k, v in pd.Series(y_train).value_counts().sort_index().items()},
            "validation_target_counts": {int(k): int(v) for k, v in pd.Series(y_val).value_counts().sort_index().items()},
        },
        "cross_validation_results": cv_results,
        "validation_results": validation_results,
        "strongest_candidate": {
            "model": strongest_name,
            "validation_metrics": validation_results[strongest_name]["metrics"],
            "cv_metrics": cv_results[strongest_name],
        },
        "files_created": files_created
        + [
            "reports/heart_disease_baseline_model_report.json",
            "reports/heart_disease_baseline_model_report.md",
        ],
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORTS_DIR / "heart_disease_baseline_model_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(report), f, indent=2)

    markdown_path = REPORTS_DIR / "heart_disease_baseline_model_report.md"
    _write_markdown_report(markdown_path, _json_safe(report))

    print("\n" + "=" * 112)
    print("VALIDATION RESULTS")
    print("=" * 112)
    print(
        f"{'Model':<22} | {'Accuracy':<8} | {'PrecMacro':<9} | {'RecMacro':<8} | "
        f"{'F1Macro':<8} | {'F1Weighted':<10} | {'AUCw':<8}"
    )
    print("-" * 112)
    for name, result in validation_results.items():
        m = result["metrics"]
        print(
            f"{name:<22} | {m['accuracy']:<8.4f} | {m['precision_macro']:<9.4f} | "
            f"{m['recall_macro']:<8.4f} | {m['f1_macro']:<8.4f} | "
            f"{m['f1_weighted']:<10.4f} | {m['roc_auc_ovr_weighted']:<8.4f}"
        )
        print(f"  confusion_matrix={m['confusion_matrix']}")
    print("=" * 112)
    print(f"Strongest candidate: {strongest_name}")
    print(f"[OK] JSON report saved to: {json_path.relative_to(PROJECT_ROOT)}")
    print(f"[OK] Markdown report saved to: {markdown_path.relative_to(PROJECT_ROOT)}")
    print(f"[OK] Confusion matrix figure saved to: {cm_plot_path.relative_to(PROJECT_ROOT)}")
    print("No hyperparameter tuning was performed.")
    print("Test split remained untouched.")
    print("=" * 80)


if __name__ == "__main__":
    run_heart_baselines()
