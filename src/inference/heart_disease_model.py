"""Production persistence and inference for the UCI Heart Disease pipeline.

Loads the selected tuned Logistic Regression pipeline from disk. Does not train
or retune models.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.preprocessing.heart_disease import (
    CATEGORICAL_FEATURES,
    HEART_DISEASE_COLUMNS,
    HEART_DISEASE_TARGET,
    NUMERIC_FEATURES,
    ZERO_AS_MISSING_MEASUREMENT_COLUMNS,
)


MODEL_VERSION = "1.0.0"
TARGET_VARIABLE = HEART_DISEASE_TARGET
TARGET_CLASSES = [0, 1, 2, 3, 4]
DATASET_NAME = "UCI Heart Disease"
MODEL_NAME = "Tuned Logistic Regression (multiclass num target)"

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
HEART_MODEL_DIR = REPO_ROOT / "models" / "heart"
PIPELINE_FILENAME = "heart_disease_risk_pipeline.joblib"
METADATA_FILENAME = "metadata.json"
TUNING_REPORT_PATH = REPO_ROOT / "reports" / "heart_disease_hyperparameter_tuning_report.json"
FINAL_TEST_REPORT_PATH = REPO_ROOT / "reports" / "heart_disease_final_test_evaluation_report.json"

INPUT_FEATURE_COLUMNS: List[str] = [
    column for column in HEART_DISEASE_COLUMNS if column != HEART_DISEASE_TARGET
]


def get_pipeline_path() -> Path:
    return HEART_MODEL_DIR / PIPELINE_FILENAME


def get_metadata_path() -> Path:
    return HEART_MODEL_DIR / METADATA_FILENAME


def load_heart_disease_pipeline() -> Pipeline:
    """Load the serialized end-to-end Heart Disease pipeline."""
    path = get_pipeline_path()
    if not path.is_file():
        raise FileNotFoundError(f"Heart Disease pipeline not found at {path}.")

    pipeline = joblib.load(path)
    if not isinstance(pipeline, Pipeline):
        raise TypeError(f"Expected sklearn Pipeline at {path}, got {type(pipeline)}")

    expected_steps = {"features", "clf"}
    if set(pipeline.named_steps) != expected_steps:
        raise TypeError(f"Expected pipeline steps {sorted(expected_steps)}, got {list(pipeline.named_steps)}")

    return pipeline


def _transformed_feature_names(pipeline: Pipeline) -> List[str]:
    features = pipeline.named_steps["features"]
    return list(features.named_steps["preprocessor"].get_feature_names_out())


def _raw_categorical_allowed_values(pipeline: Pipeline) -> Dict[str, List[float]]:
    features = pipeline.named_steps["features"]
    preprocessor = features.named_steps["preprocessor"]
    categorical_columns = list(preprocessor.transformers_[1][2])
    onehot = preprocessor.named_transformers_["cat"].named_steps["onehot"]

    allowed_values: Dict[str, List[float]] = {}
    for column, categories in zip(categorical_columns, onehot.categories_):
        if column in CATEGORICAL_FEATURES:
            allowed_values[column] = [float(value) for value in categories]
    return allowed_values


def build_heart_production_metadata(pipeline: Pipeline) -> Dict[str, Any]:
    """Assemble production metadata from the saved Heart Disease pipeline and reports."""
    if not TUNING_REPORT_PATH.is_file():
        raise FileNotFoundError(f"Missing tuning report: {TUNING_REPORT_PATH}")
    if not FINAL_TEST_REPORT_PATH.is_file():
        raise FileNotFoundError(f"Missing final test report: {FINAL_TEST_REPORT_PATH}")

    with open(TUNING_REPORT_PATH, encoding="utf-8") as f:
        tuning_report = json.load(f)
    with open(FINAL_TEST_REPORT_PATH, encoding="utf-8") as f:
        final_test_report = json.load(f)

    clf = pipeline.named_steps["clf"]
    return {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "dataset_name": DATASET_NAME,
        "target_variable": TARGET_VARIABLE,
        "target_classes": TARGET_CLASSES,
        "input_feature_schema": {
            "columns": INPUT_FEATURE_COLUMNS,
            "column_count": len(INPUT_FEATURE_COLUMNS),
            "numeric_columns": NUMERIC_FEATURES,
            "categorical_columns": CATEGORICAL_FEATURES,
            "categorical_allowed_values": _raw_categorical_allowed_values(pipeline),
            "zero_as_missing_columns_from_existing_preprocessing": ZERO_AS_MISSING_MEASUREMENT_COLUMNS,
            "description": "Raw 13-feature UCI Heart Disease schema, excluding source and target.",
        },
        "transformed_feature_names": _transformed_feature_names(pipeline),
        "transformed_feature_count": len(_transformed_feature_names(pipeline)),
        "pipeline_structure": "features (HeartDiseaseFeatureEngineer + ColumnTransformer) -> clf",
        "classifier": {
            "library": "scikit-learn",
            "estimator": type(clf).__name__,
            "classes": [int(value) for value in clf.classes_],
            "hyperparameters": {
                "C": getattr(clf, "C", None),
                "class_weight": getattr(clf, "class_weight", None),
                "solver": getattr(clf, "solver", None),
                "penalty": getattr(clf, "penalty", None),
                "max_iter": getattr(clf, "max_iter", None),
            },
        },
        "selection": {
            "selected_model": tuning_report["selected_model"]["model"],
            "selection_reason": tuning_report["selected_model"]["selection_reason"],
            "training_cv_metrics": tuning_report["selected_model"]["cv_metrics"],
        },
        "final_test_metrics": final_test_report["test_metrics"],
        "metadata_created_date": date.today().isoformat(),
        "artifact": {
            "pipeline_file": PIPELINE_FILENAME,
            "relative_path": f"models/heart/{PIPELINE_FILENAME}",
        },
    }


def save_heart_production_metadata(pipeline: Pipeline | None = None) -> Path:
    """Write metadata.json beside the existing Heart Disease pipeline."""
    HEART_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if pipeline is None:
        pipeline = load_heart_disease_pipeline()

    metadata = build_heart_production_metadata(pipeline)
    path = get_metadata_path()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return path


def predict_heart_disease(
    features: Union[pd.DataFrame, Dict[str, Any]],
    pipeline: Pipeline | None = None,
) -> Dict[str, Any]:
    """Run inference on one or more raw Heart Disease records."""
    if pipeline is None:
        pipeline = load_heart_disease_pipeline()

    if isinstance(features, dict):
        X = pd.DataFrame([features])
    else:
        X = features.copy()

    missing = set(INPUT_FEATURE_COLUMNS) - set(X.columns)
    if missing:
        raise ValueError(f"Missing required input columns: {sorted(missing)}")

    X = X[INPUT_FEATURE_COLUMNS]
    X = _validate_inference_features(X, pipeline)

    labels = pipeline.predict(X)
    result: Dict[str, Any] = {
        "predicted_class": [int(value) for value in labels.tolist()],
    }

    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(X)
        classes = [int(value) for value in pipeline.named_steps["clf"].classes_]
        result["class_labels"] = classes
        result["class_probabilities"] = probabilities.tolist()

    return result


def _validate_inference_features(X: pd.DataFrame, pipeline: Pipeline) -> pd.DataFrame:
    """Validate raw Heart Disease features before model inference."""
    numeric_X = X.apply(pd.to_numeric, errors="coerce")

    invalid_numeric = [
        column
        for column in INPUT_FEATURE_COLUMNS
        if numeric_X[column].isna().any()
    ]
    if invalid_numeric:
        raise ValueError(f"Input columns must contain numeric, non-missing values: {invalid_numeric}")

    non_finite = [
        column
        for column in INPUT_FEATURE_COLUMNS
        if not np.isfinite(numeric_X[column].to_numpy()).all()
    ]
    if non_finite:
        raise ValueError(f"Input columns must contain finite numeric values: {non_finite}")

    zero_measurement_errors = []
    for column in ZERO_AS_MISSING_MEASUREMENT_COLUMNS:
        invalid_mask = numeric_X[column] == 0
        if invalid_mask.any():
            invalid_values = numeric_X.loc[invalid_mask, column].tolist()
            zero_measurement_errors.append(
                f"{column}={invalid_values} is invalid because existing preprocessing treats zero as missing"
            )

    categorical_errors = []
    for column, allowed_values in _raw_categorical_allowed_values(pipeline).items():
        invalid_mask = ~numeric_X[column].isin(allowed_values)
        if invalid_mask.any():
            invalid_values = numeric_X.loc[invalid_mask, column].tolist()
            categorical_errors.append(f"{column}={invalid_values} not in {allowed_values}")

    if zero_measurement_errors or categorical_errors:
        errors = zero_measurement_errors + categorical_errors
        raise ValueError("Invalid input values: " + "; ".join(errors))

    return numeric_X
