"""Heart Disease hyperparameter tuning.

Tunes the two baseline candidates implicated by the baseline disagreement:
Logistic Regression (best training CV) and Gradient Boosting (best validation).
Only the training split is loaded. Validation and test splits are not used.
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
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import f1_score, precision_score, recall_score, make_scorer  # noqa: E402
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from src.preprocessing.heart_disease import (  # noqa: E402
    HEART_DISEASE_TARGET,
    SOURCE_COLUMN,
    create_heart_disease_feature_pipeline,
)


PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "heart_disease"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models" / "heart" / "tuned"
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


def _load_train_split() -> pd.DataFrame:
    path = PROCESSED_DIR / "heart_disease_train.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing Heart Disease training split: {path}")
    return pd.read_csv(path)


def _split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    X = df.drop(columns=[HEART_DISEASE_TARGET, SOURCE_COLUMN], errors="ignore")
    y = df[HEART_DISEASE_TARGET].to_numpy()
    return X, y


def _make_pipeline(classifier: Any) -> Pipeline:
    return Pipeline(
        steps=[
            ("features", create_heart_disease_feature_pipeline()),
            ("clf", classifier),
        ]
    )


def _scoring() -> dict[str, Any]:
    return {
        "accuracy": "accuracy",
        "precision_macro": make_scorer(precision_score, average="macro", zero_division=0),
        "precision_weighted": make_scorer(precision_score, average="weighted", zero_division=0),
        "recall_macro": make_scorer(recall_score, average="macro", zero_division=0),
        "recall_weighted": make_scorer(recall_score, average="weighted", zero_division=0),
        "f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
        "f1_weighted": make_scorer(f1_score, average="weighted", zero_division=0),
        "roc_auc_ovr_macro": "roc_auc_ovr",
        "roc_auc_ovr_weighted": "roc_auc_ovr_weighted",
    }


def _best_summary(search: GridSearchCV | RandomizedSearchCV, duration_seconds: float) -> dict[str, Any]:
    best_idx = int(search.best_index_)
    cv = search.cv_results_
    return {
        "search_class": type(search).__name__,
        "best_params": search.best_params_,
        "best_index": best_idx,
        "refit_metric": search.refit,
        "best_cv_f1_weighted": round(float(search.best_score_), 4),
        "best_cv_accuracy": round(float(cv["mean_test_accuracy"][best_idx]), 4),
        "best_cv_precision_macro": round(float(cv["mean_test_precision_macro"][best_idx]), 4),
        "best_cv_precision_weighted": round(float(cv["mean_test_precision_weighted"][best_idx]), 4),
        "best_cv_recall_macro": round(float(cv["mean_test_recall_macro"][best_idx]), 4),
        "best_cv_recall_weighted": round(float(cv["mean_test_recall_weighted"][best_idx]), 4),
        "best_cv_f1_macro": round(float(cv["mean_test_f1_macro"][best_idx]), 4),
        "best_cv_roc_auc_ovr_macro": round(float(cv["mean_test_roc_auc_ovr_macro"][best_idx]), 4),
        "best_cv_roc_auc_ovr_weighted": round(float(cv["mean_test_roc_auc_ovr_weighted"][best_idx]), 4),
        "best_cv_f1_weighted_std": round(float(cv["std_test_f1_weighted"][best_idx]), 4),
        "best_cv_roc_auc_ovr_weighted_std": round(float(cv["std_test_roc_auc_ovr_weighted"][best_idx]), 4),
        "candidates_evaluated": int(len(cv["params"])),
        "total_fit_count": int(len(cv["params"]) * search.cv.get_n_splits()),
        "duration_seconds": round(float(duration_seconds), 2),
    }


def _save_cv_results(search: GridSearchCV | RandomizedSearchCV, filename: str) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / filename
    df = pd.DataFrame(search.cv_results_)
    df = df.sort_values(
        by=["rank_test_f1_weighted", "rank_test_roc_auc_ovr_weighted"],
        ascending=[True, True],
    )
    df.to_csv(path, index=False)
    return path


def _write_markdown_report(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Heart Disease Hyperparameter Tuning Report",
        "",
        f"Target: {report['target_definition']}",
        "",
        "Only the training split was loaded for tuning. Validation and test were not used.",
        "",
        "## Selection",
        "",
        (
            f"Selected `{report['selected_model']['model']}` by highest mean CV weighted F1. "
            "Weighted one-vs-rest ROC-AUC was the first tie-breaker."
        ),
        "",
        "## Tuned Candidate Results",
        "",
        "| Model | Search | Candidates | CV F1 Weighted | CV F1 Macro | CV Accuracy | CV ROC-AUC OVR Weighted | Best Params |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for name, result in report["tuning_results"].items():
        lines.append(
            f"| {name} | {result['search_class']} | {result['candidates_evaluated']} | "
            f"{result['best_cv_f1_weighted']:.4f} | {result['best_cv_f1_macro']:.4f} | "
            f"{result['best_cv_accuracy']:.4f} | {result['best_cv_roc_auc_ovr_weighted']:.4f} | "
            f"`{result['best_params']}` |"
        )

    lines.extend(["", "## Files", ""])
    for output in report["files_created"]:
        lines.append(f"- `{output}`")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def run_tuning() -> None:
    print("=" * 80)
    print("STEP 7: HEART DISEASE HYPERPARAMETER TUNING")
    print("=" * 80)

    train_df = _load_train_split()
    X_train, y_train = _split_xy(train_df)
    print(f"Loaded training split only: {train_df.shape}")
    print(f"Train target counts: {pd.Series(y_train).value_counts().sort_index().to_dict()}")
    print("Validation and test splits were intentionally not loaded.")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = _scoring()

    logistic_pipeline = _make_pipeline(
        LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            max_iter=3000,
            random_state=42,
        )
    )
    logistic_param_grid = {
        "clf__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
        "clf__class_weight": ["balanced", None],
    }
    logistic_search = GridSearchCV(
        estimator=logistic_pipeline,
        param_grid=logistic_param_grid,
        scoring=scoring,
        refit="f1_weighted",
        cv=cv,
        n_jobs=1,
        error_score="raise",
        return_train_score=True,
        verbose=1,
    )

    gradient_pipeline = _make_pipeline(
        HistGradientBoostingClassifier(
            random_state=42,
        )
    )
    gradient_param_distributions = {
        "clf__learning_rate": [0.03, 0.05, 0.08, 0.1, 0.15],
        "clf__max_iter": [50, 75, 100, 125, 150],
        "clf__max_depth": [3, 5, 8, None],
        "clf__max_leaf_nodes": [15, 31],
        "clf__min_samples_leaf": [10, 20, 30],
        "clf__l2_regularization": [0.0, 0.1, 1.0, 10.0],
        "clf__class_weight": ["balanced", None],
    }
    gradient_search = RandomizedSearchCV(
        estimator=gradient_pipeline,
        param_distributions=gradient_param_distributions,
        n_iter=16,
        scoring=scoring,
        refit="f1_weighted",
        cv=cv,
        n_jobs=1,
        random_state=42,
        error_score="raise",
        return_train_score=True,
        verbose=1,
    )

    print("\n[STEP 1] Tuning Logistic Regression with GridSearchCV...")
    t0 = time.time()
    logistic_search.fit(X_train, y_train)
    logistic_duration = time.time() - t0
    logistic_summary = _best_summary(logistic_search, logistic_duration)
    logistic_cv_path = _save_cv_results(
        logistic_search,
        "heart_disease_logistic_regression_tuning_cv_results.csv",
    )
    print(
        "  Logistic Regression best CV: "
        f"f1_weighted={logistic_summary['best_cv_f1_weighted']:.4f}, "
        f"roc_auc_ovr_weighted={logistic_summary['best_cv_roc_auc_ovr_weighted']:.4f}"
    )
    print(f"  best_params={logistic_summary['best_params']}")

    print("\n[STEP 2] Tuning Gradient Boosting with RandomizedSearchCV...")
    t0 = time.time()
    gradient_search.fit(X_train, y_train)
    gradient_duration = time.time() - t0
    gradient_summary = _best_summary(gradient_search, gradient_duration)
    gradient_cv_path = _save_cv_results(
        gradient_search,
        "heart_disease_gradient_boosting_tuning_cv_results.csv",
    )
    print(
        "  Gradient Boosting best CV: "
        f"f1_weighted={gradient_summary['best_cv_f1_weighted']:.4f}, "
        f"roc_auc_ovr_weighted={gradient_summary['best_cv_roc_auc_ovr_weighted']:.4f}"
    )
    print(f"  best_params={gradient_summary['best_params']}")

    summaries = {
        "Logistic Regression": logistic_summary,
        "Gradient Boosting": gradient_summary,
    }
    selected_name = max(
        summaries,
        key=lambda name: (
            summaries[name]["best_cv_f1_weighted"],
            summaries[name]["best_cv_roc_auc_ovr_weighted"],
            summaries[name]["best_cv_f1_macro"],
        ),
    )
    selected_search = logistic_search if selected_name == "Logistic Regression" else gradient_search

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    logistic_model_path = MODELS_DIR / "logistic_regression_tuned_pipeline.joblib"
    gradient_model_path = MODELS_DIR / "gradient_boosting_tuned_pipeline.joblib"
    selected_model_path = MODELS_DIR / "heart_disease_selected_tuned_pipeline.joblib"
    joblib.dump(logistic_search.best_estimator_, logistic_model_path)
    joblib.dump(gradient_search.best_estimator_, gradient_model_path)
    joblib.dump(selected_search.best_estimator_, selected_model_path)

    report = {
        "target_definition": "Existing multiclass `num` target with classes 0, 1, 2, 3, 4.",
        "data_used": {
            "training_split_only": True,
            "train_shape": [int(train_df.shape[0]), int(train_df.shape[1])],
            "train_target_counts": {
                int(k): int(v) for k, v in pd.Series(y_train).value_counts().sort_index().items()
            },
            "validation_used": False,
            "test_used": False,
        },
        "methodology": {
            "candidates": ["Logistic Regression", "Gradient Boosting"],
            "why_these_candidates": (
                "Baseline validation selected Gradient Boosting, while baseline training CV favored "
                "Logistic Regression, so both were tuned."
            ),
            "cv": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
            "searches": {
                "Logistic Regression": "GridSearchCV",
                "Gradient Boosting": "RandomizedSearchCV(n_iter=16, random_state=42)",
            },
            "feature_pipeline": (
                "Heart feature engineering and preprocessing are inside each sklearn Pipeline and refit "
                "inside each CV fold."
            ),
            "refit_metric": "f1_weighted",
            "selection_priority": (
                "Highest mean CV weighted F1; mean CV weighted one-vs-rest ROC-AUC then mean CV macro F1 "
                "used as tie-breakers."
            ),
            "hyperparameter_tuning_only": True,
        },
        "tuning_results": summaries,
        "selected_model": {
            "model": selected_name,
            "selection_reason": (
                f"{selected_name} had the highest mean CV weighted F1 among tuned candidates."
            ),
            "cv_metrics": summaries[selected_name],
            "artifact": str(selected_model_path.relative_to(PROJECT_ROOT)),
        },
        "files_created": [
            str(logistic_cv_path.relative_to(PROJECT_ROOT)),
            str(gradient_cv_path.relative_to(PROJECT_ROOT)),
            str(logistic_model_path.relative_to(PROJECT_ROOT)),
            str(gradient_model_path.relative_to(PROJECT_ROOT)),
            str(selected_model_path.relative_to(PROJECT_ROOT)),
            "reports/heart_disease_hyperparameter_tuning_report.json",
            "reports/heart_disease_hyperparameter_tuning_report.md",
        ],
    }

    json_path = REPORTS_DIR / "heart_disease_hyperparameter_tuning_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(report), f, indent=2)

    markdown_path = REPORTS_DIR / "heart_disease_hyperparameter_tuning_report.md"
    _write_markdown_report(markdown_path, _json_safe(report))

    print("\n" + "=" * 100)
    print("TUNING RESULTS")
    print("=" * 100)
    print(
        f"{'Model':<22} | {'Search':<18} | {'Candidates':<10} | {'F1w':<8} | "
        f"{'F1macro':<8} | {'Acc':<8} | {'AUCw':<8}"
    )
    print("-" * 100)
    for name, result in summaries.items():
        print(
            f"{name:<22} | {result['search_class']:<18} | {result['candidates_evaluated']:<10} | "
            f"{result['best_cv_f1_weighted']:<8.4f} | {result['best_cv_f1_macro']:<8.4f} | "
            f"{result['best_cv_accuracy']:<8.4f} | {result['best_cv_roc_auc_ovr_weighted']:<8.4f}"
        )
    print("=" * 100)
    print(f"Selected final tuned model: {selected_name}")
    print(f"[OK] JSON report saved to: {json_path.relative_to(PROJECT_ROOT)}")
    print(f"[OK] Markdown report saved to: {markdown_path.relative_to(PROJECT_ROOT)}")
    print("Validation and test splits were not used.")
    print("=" * 80)


if __name__ == "__main__":
    run_tuning()
