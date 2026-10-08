"""Production persistence and inference for the CDC diabetes risk pipeline.

Loads the verified Step 10 champion pipeline from disk. Does not retrain models.

Project: AI-Powered Health Risk Prediction and Monitoring System
Author: Abhiram (Lead AI/ML Engineer)
Date: 8 October 2026
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Union

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from src.preprocessing.data_loader import EXPECTED_CDC_COLUMNS

MODEL_VERSION = "1.0.0"
TARGET_VARIABLE = "Diabetes_binary"
DATASET_NAME = "CDC Diabetes Health Indicators (BRFSS 2015)"
MODEL_NAME = "Tuned HistGradientBoostingClassifier (Gradient Boosting champion)"

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DIABETES_MODEL_DIR = REPO_ROOT / "models" / "diabetes"
PIPELINE_FILENAME = "diabetes_risk_pipeline.joblib"
METADATA_FILENAME = "metadata.json"
TUNING_REPORT_PATH = REPO_ROOT / "reports" / "hyperparameter_tuning_and_final_evaluation.json"

INPUT_FEATURE_COLUMNS: List[str] = list(EXPECTED_CDC_COLUMNS)

BINARY_FEATURE_COLUMNS = [
    "HighBP",
    "HighChol",
    "CholCheck",
    "Smoker",
    "Stroke",
    "HeartDiseaseorAttack",
    "PhysActivity",
    "Fruits",
    "Veggies",
    "HvyAlcoholConsump",
    "AnyHealthcare",
    "NoDocbcCost",
    "DiffWalk",
    "Sex",
]

FEATURE_VALUE_RANGES = {
    **{column: (0, 1) for column in BINARY_FEATURE_COLUMNS},
    "BMI": (12, 100),
    "GenHlth": (1, 5),
    "MentHlth": (0, 30),
    "PhysHlth": (0, 30),
    "Age": (1, 13),
    "Education": (1, 6),
    "Income": (1, 8),
}

INTEGER_FEATURE_COLUMNS = [column for column in INPUT_FEATURE_COLUMNS if column != "BMI"]


def get_pipeline_path() -> Path:
    return DIABETES_MODEL_DIR / PIPELINE_FILENAME


def get_metadata_path() -> Path:
    return DIABETES_MODEL_DIR / METADATA_FILENAME


def load_diabetes_risk_pipeline() -> Pipeline:
    """Load the serialized end-to-end diabetes risk pipeline."""
    path = get_pipeline_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"Diabetes risk pipeline not found at {path}. Run Step 10 tuning first."
        )
    pipeline = joblib.load(path)
    if not isinstance(pipeline, Pipeline):
        raise TypeError(f"Expected sklearn Pipeline at {path}, got {type(pipeline)}")
    return pipeline


def _transformed_feature_names(pipeline: Pipeline) -> List[str]:
    preprocess = pipeline.named_steps["preprocess"]
    return list(preprocess.named_steps["preprocessor"].get_feature_names_out())


def build_production_metadata(pipeline: Pipeline) -> Dict[str, Any]:
    """Assemble production metadata from the saved pipeline and tuning report."""
    if not TUNING_REPORT_PATH.is_file():
        raise FileNotFoundError(f"Missing tuning report: {TUNING_REPORT_PATH}")

    with open(TUNING_REPORT_PATH, encoding="utf-8") as f:
        tuning_report = json.load(f)

    champion = tuning_report["champion"]
    methodology = tuning_report["methodology"]
    gb_tuning = tuning_report["gb_tuning"]
    final_test = tuning_report["final_test_evaluation"]

    clf = pipeline.named_steps["clf"]
    return {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "dataset_name": DATASET_NAME,
        "target_variable": TARGET_VARIABLE,
        "input_feature_schema": {
            "columns": INPUT_FEATURE_COLUMNS,
            "column_count": len(INPUT_FEATURE_COLUMNS),
            "description": "Raw BRFSS 2015 survey fields (same order as CDC binary dataset, excluding target).",
        },
        "transformed_feature_names": _transformed_feature_names(pipeline),
        "preprocessing_approach": {
            "pipeline_structure": "preprocess (DiabetesFeatureEngineer + ColumnTransformer) -> clf",
            "feature_engineering": [
                "BMI_Category",
                "CardioMetabolic_Risk_Score",
                "HighBP_x_HighChol",
                "Healthy_Lifestyle_Score",
                "Age_x_GenHlth",
                "Sedentary_Obese",
            ],
            "scaling": "StandardScaler on continuous numerical columns (BMI, MentHlth, PhysHlth, Age_x_GenHlth)",
            "ordinal_and_binary": "passthrough (no encoding)",
            "transformed_feature_count": len(_transformed_feature_names(pipeline)),
        },
        "selected_hyperparameters": gb_tuning["best_params_classifier_only"],
        "cross_validation_methodology": {
            "search_algorithm": methodology["search"],
            "cv_strategy": methodology["cv"],
            "scoring_metric": methodology["scoring"],
            "train_only_tuning": methodology["train_only_tuning"],
            "preprocessing_in_pipeline": methodology["preprocessing_in_pipeline"],
            "champion_selection": "Highest train CV ROC-AUC between tuned Gradient Boosting and Random Forest",
            "best_score_cv_roc_auc": champion["best_score_cv_roc_auc"],
            "candidates_evaluated": ["Gradient Boosting", "Random Forest"],
        },
        "classifier": {
            "library": "scikit-learn",
            "estimator": type(clf).__name__,
            "class_weight": getattr(clf, "class_weight", None),
        },
        "training_date": date.today().isoformat(),
        "training_samples": 160631,
        "final_test_metrics": final_test,
        "artifact": {
            "pipeline_file": PIPELINE_FILENAME,
            "relative_path": f"models/diabetes/{PIPELINE_FILENAME}",
        },
    }


def save_production_metadata(pipeline: Pipeline | None = None) -> Path:
    """Write metadata.json alongside the existing pipeline (no training)."""
    DIABETES_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if pipeline is None:
        pipeline = load_diabetes_risk_pipeline()
    metadata = build_production_metadata(pipeline)
    path = get_metadata_path()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return path


def predict_diabetes_risk(
    features: Union[pd.DataFrame, Dict[str, Any]],
    pipeline: Pipeline | None = None,
) -> Dict[str, Any]:
    """Run inference on one or more survey records (raw 21-feature schema)."""
    if pipeline is None:
        pipeline = load_diabetes_risk_pipeline()

    if isinstance(features, dict):
        X = pd.DataFrame([features])
    else:
        X = features.copy()

    missing = set(INPUT_FEATURE_COLUMNS) - set(X.columns)
    if missing:
        raise ValueError(f"Missing required input columns: {sorted(missing)}")

    X = X[INPUT_FEATURE_COLUMNS]
    X = _validate_inference_features(X)
    labels = pipeline.predict(X)
    probabilities = pipeline.predict_proba(X)[:, 1]

    return {
        "predicted_class": labels.tolist(),
        "diabetes_probability": probabilities.tolist(),
    }


def _validate_inference_features(X: pd.DataFrame) -> pd.DataFrame:
    """Validate raw CDC survey features before model inference."""
    numeric_X = X.apply(pd.to_numeric, errors="coerce")

    invalid_numeric = [
        column
        for column in INPUT_FEATURE_COLUMNS
        if numeric_X[column].isna().any()
    ]
    if invalid_numeric:
        raise ValueError(f"Input columns must contain numeric, non-missing values: {invalid_numeric}")

    range_errors = []
    for column, (minimum, maximum) in FEATURE_VALUE_RANGES.items():
        invalid_mask = ~numeric_X[column].between(minimum, maximum, inclusive="both")
        if invalid_mask.any():
            invalid_values = numeric_X.loc[invalid_mask, column].tolist()
            range_errors.append(f"{column}={invalid_values} outside [{minimum}, {maximum}]")

    integer_errors = []
    for column in INTEGER_FEATURE_COLUMNS:
        invalid_mask = numeric_X[column] % 1 != 0
        if invalid_mask.any():
            invalid_values = numeric_X.loc[invalid_mask, column].tolist()
            integer_errors.append(f"{column}={invalid_values} must be integer-coded")

    if range_errors or integer_errors:
        errors = range_errors + integer_errors
        raise ValueError("Invalid input values: " + "; ".join(errors))

    return numeric_X
