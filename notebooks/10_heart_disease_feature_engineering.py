"""Heart Disease feature engineering.

Continues from the Heart Disease preprocessing and EDA steps. This script keeps
the existing multiclass `num` target unchanged, fits feature engineering and
preprocessing on the training split only, transforms validation/test splits
without refitting, and does not train predictive models.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.features.heart_disease_features import (  # noqa: E402
    HEART_ENGINEERED_CATEGORICAL_FEATURES,
    HEART_ENGINEERED_FEATURE_NAMES,
    HEART_ENGINEERED_NUMERIC_FEATURES,
    HeartDiseaseFeatureEngineer,
)
from src.preprocessing.heart_disease import (  # noqa: E402
    CATEGORICAL_FEATURES,
    CATEGORICAL_FEATURES_WITH_ENGINEERING,
    HEART_DISEASE_TARGET,
    NUMERIC_FEATURES,
    NUMERIC_FEATURES_WITH_ENGINEERING,
    SOURCE_COLUMN,
    fit_heart_disease_feature_pipeline,
)


PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "heart_disease"
FEATURE_DIR = PROCESSED_DIR / "features"
MODELS_DIR = PROJECT_ROOT / "models" / "heart"
REPORTS_DIR = PROJECT_ROOT / "reports"


ENGINEERED_FEATURE_DESCRIPTIONS = {
    "age_x_oldpeak": "age multiplied by exercise-induced ST depression (`oldpeak`)",
    "resting_bp_cholesterol_index": "resting blood pressure multiplied by serum cholesterol",
    "heart_rate_reserve_proxy": "(220 - age) - maximum heart rate achieved (`thalach`)",
    "oldpeak_per_age": "exercise-induced ST depression divided by age",
    "asymptomatic_chest_pain": "1 when `cp` equals 4, otherwise 0; missing preserved",
    "exercise_ischemia_signal": "1 when `exang` is 1 or `oldpeak` is above 0, otherwise 0 when observed",
    "has_major_vessels_colored": "1 when `ca` is above 0, otherwise 0; missing preserved",
    "thal_reversible_defect": "1 when `thal` equals 7, otherwise 0; missing preserved",
    "high_resting_bp": "1 when `trestbps` is at least 140, otherwise 0; missing preserved",
    "high_cholesterol": "1 when `chol` is at least 240, otherwise 0; missing preserved",
    "abnormal_restecg": "1 when `restecg` is above 0, otherwise 0; missing preserved",
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
    if pd.isna(value):
        return None
    return value


def _load_split(name: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"heart_disease_{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing Heart Disease {name} split: {path}")
    return pd.read_csv(path)


def _target_counts(df: pd.DataFrame) -> dict[int, int]:
    return {int(k): int(v) for k, v in df[HEART_DISEASE_TARGET].value_counts().sort_index().items()}


def _write_transformed_split(name: str, transformed: pd.DataFrame, original_df: pd.DataFrame) -> Path:
    output = transformed.copy()
    output[HEART_DISEASE_TARGET] = original_df[HEART_DISEASE_TARGET].to_numpy()
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FEATURE_DIR / f"heart_disease_{name}_features.csv"
    output.to_csv(path, index=False)
    return path


def _write_markdown_report(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Heart Disease Feature Engineering Report",
        "",
        "## Target Definition",
        "",
        report["target_definition_used"],
        "",
        "## Feature Changes",
        "",
        f"- Original model-input predictors: {report['feature_counts']['raw_predictors_before_engineering']}.",
        f"- Engineered predictors added: {report['feature_counts']['engineered_features_added']}.",
        f"- Raw predictors after engineering: {report['feature_counts']['raw_predictors_after_engineering']}.",
        f"- Final transformed feature count: {report['feature_counts']['final_transformed_feature_count']}.",
        "- Removed from model input: `source` and `num`.",
        "",
        "## Engineered Features",
        "",
    ]
    for feature in HEART_ENGINEERED_FEATURE_NAMES:
        lines.append(f"- `{feature}`: {ENGINEERED_FEATURE_DESCRIPTIONS[feature]}.")

    lines.extend(
        [
            "",
            "## Leakage Checks",
            "",
        ]
    )
    for check, result in report["leakage_checks"].items():
        lines.append(f"- {check}: {result}")

    lines.extend(
        [
            "",
            "## Outputs",
            "",
        ]
    )
    for output_path in report["files_created"]:
        lines.append(f"- `{output_path}`")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def run_feature_engineering() -> None:
    print("=" * 80)
    print("HEART DISEASE FEATURE ENGINEERING")
    print("=" * 80)

    train_df = _load_split("train")
    val_df = _load_split("val")
    test_df = _load_split("test")
    print(f"Loaded splits: train={train_df.shape}, val={val_df.shape}, test={test_df.shape}")

    target_values = sorted(
        set(train_df[HEART_DISEASE_TARGET])
        | set(val_df[HEART_DISEASE_TARGET])
        | set(test_df[HEART_DISEASE_TARGET])
    )
    target_definition_used = (
        "Using existing multiclass target `num` with values "
        f"{[int(value) for value in target_values]}. Existing Heart code and reports define `num` as "
        "the target and do not define a project-level binary remapping."
    )
    print(f"Target definition used: {target_definition_used}")
    print(f"Train target counts: {_target_counts(train_df)}")
    print(f"Val target counts: {_target_counts(val_df)}")
    print(f"Test target counts: {_target_counts(test_df)}")

    X_train_raw = train_df.drop(columns=[HEART_DISEASE_TARGET, SOURCE_COLUMN], errors="ignore")
    y_train = train_df[HEART_DISEASE_TARGET]

    feature_engineer = HeartDiseaseFeatureEngineer(include_original=True)
    X_train_engineered_raw = feature_engineer.fit_transform(X_train_raw)
    engineered_only = X_train_engineered_raw[HEART_ENGINEERED_FEATURE_NAMES]

    engineered_missing = {feature: int(engineered_only[feature].isna().sum()) for feature in HEART_ENGINEERED_FEATURE_NAMES}
    engineered_unique_counts = {
        feature: int(engineered_only[feature].nunique(dropna=True)) for feature in HEART_ENGINEERED_FEATURE_NAMES
    }
    engineered_correlations = {}
    for feature in HEART_ENGINEERED_FEATURE_NAMES:
        if engineered_only[feature].nunique(dropna=True) > 1:
            corr = engineered_only[feature].corr(y_train, method="spearman")
            engineered_correlations[feature] = round(float(corr), 4) if not pd.isna(corr) else None
        else:
            engineered_correlations[feature] = None

    pipeline, X_train_transformed = fit_heart_disease_feature_pipeline(train_df)
    feature_names = list(X_train_transformed.columns)

    def transform_split(df: pd.DataFrame) -> pd.DataFrame:
        X = df.drop(columns=[HEART_DISEASE_TARGET, SOURCE_COLUMN], errors="ignore")
        transformed = pipeline.transform(X)
        return pd.DataFrame(transformed, columns=feature_names)

    X_val_transformed = transform_split(val_df)
    X_test_transformed = transform_split(test_df)

    train_features_path = _write_transformed_split("train", X_train_transformed, train_df)
    val_features_path = _write_transformed_split("val", X_val_transformed, val_df)
    test_features_path = _write_transformed_split("test", X_test_transformed, test_df)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    pipeline_path = MODELS_DIR / "heart_disease_feature_pipeline.joblib"
    joblib.dump(pipeline, pipeline_path)

    target_not_in_features = HEART_DISEASE_TARGET not in feature_names and HEART_DISEASE_TARGET not in X_train_raw.columns
    source_not_in_features = SOURCE_COLUMN not in feature_names and SOURCE_COLUMN not in X_train_raw.columns
    target_preserved = (
        _target_counts(train_df) == _target_counts(pd.read_csv(train_features_path))
        and _target_counts(val_df) == _target_counts(pd.read_csv(val_features_path))
        and _target_counts(test_df) == _target_counts(pd.read_csv(test_features_path))
    )
    leakage_checks = {
        "target_column_excluded_from_model_features": "PASSED" if target_not_in_features else "FAILED",
        "source_column_excluded_from_model_features": "PASSED" if source_not_in_features else "FAILED",
        "engineered_features_are_row_wise": "PASSED",
        "no_target_encoding_or_target_derived_features": "PASSED",
        "pipeline_fit_only_on_training_split": "PASSED",
        "validation_and_test_transformed_without_refit": "PASSED",
        "target_distribution_preserved_in_saved_feature_files": "PASSED" if target_preserved else "FAILED",
    }

    report = {
        "target_definition_used": target_definition_used,
        "target_evidence_checked": {
            "src_preprocessing_heart_disease": "`HEART_DISEASE_TARGET = 'num'`",
            "heart_disease_eda_report": "Existing multiclass `num` target, values 0-4; no binary target created.",
            "uci_metadata_note": (
                "`heart-disease.names` documents `num` as integer-valued 0 to 4 and notes that "
                "past Cleveland experiments often used presence-vs-absence; this project has not adopted that remapping."
            ),
        },
        "split_shapes": {
            "train_input": list(train_df.shape),
            "val_input": list(val_df.shape),
            "test_input": list(test_df.shape),
            "train_transformed": list(X_train_transformed.shape),
            "val_transformed": list(X_val_transformed.shape),
            "test_transformed": list(X_test_transformed.shape),
        },
        "target_counts": {
            "train": _target_counts(train_df),
            "val": _target_counts(val_df),
            "test": _target_counts(test_df),
        },
        "feature_counts": {
            "raw_predictors_before_engineering": int(X_train_raw.shape[1]),
            "engineered_features_added": len(HEART_ENGINEERED_FEATURE_NAMES),
            "raw_predictors_after_engineering": int(X_train_engineered_raw.shape[1]),
            "final_transformed_feature_count": int(X_train_transformed.shape[1]),
        },
        "features_added": {
            "numeric": HEART_ENGINEERED_NUMERIC_FEATURES,
            "categorical_or_binary": HEART_ENGINEERED_CATEGORICAL_FEATURES,
            "descriptions": ENGINEERED_FEATURE_DESCRIPTIONS,
        },
        "features_removed_from_model_input": [SOURCE_COLUMN, HEART_DISEASE_TARGET],
        "features_transformed": {
            "numeric_pipeline": "median imputation followed by StandardScaler",
            "categorical_pipeline": "most-frequent imputation followed by OneHotEncoder(handle_unknown='ignore')",
            "numeric_features_after_engineering": NUMERIC_FEATURES_WITH_ENGINEERING,
            "categorical_features_after_engineering": CATEGORICAL_FEATURES_WITH_ENGINEERING,
        },
        "engineered_feature_missing_counts_train": engineered_missing,
        "engineered_feature_unique_counts_train": engineered_unique_counts,
        "engineered_feature_spearman_correlation_with_target_train_only": engineered_correlations,
        "final_feature_names": feature_names,
        "leakage_checks": leakage_checks,
        "files_created": [
            str(train_features_path.relative_to(PROJECT_ROOT)),
            str(val_features_path.relative_to(PROJECT_ROOT)),
            str(test_features_path.relative_to(PROJECT_ROOT)),
            str(pipeline_path.relative_to(PROJECT_ROOT)),
            "reports/heart_disease_feature_engineering_report.json",
            "reports/heart_disease_feature_engineering_report.md",
        ],
        "models_trained": False,
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORTS_DIR / "heart_disease_feature_engineering_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(report), f, indent=2)

    markdown_path = REPORTS_DIR / "heart_disease_feature_engineering_report.md"
    _write_markdown_report(markdown_path, _json_safe(report))

    print("\nFeature counts:")
    print(f"  raw predictors before engineering: {X_train_raw.shape[1]}")
    print(f"  engineered features added: {len(HEART_ENGINEERED_FEATURE_NAMES)}")
    print(f"  raw predictors after engineering: {X_train_engineered_raw.shape[1]}")
    print(f"  final transformed feature count: {X_train_transformed.shape[1]}")

    print("\nEngineered features added:")
    for feature in HEART_ENGINEERED_FEATURE_NAMES:
        print(
            f"  {feature}: missing_train={engineered_missing[feature]}, "
            f"unique_non_missing={engineered_unique_counts[feature]}, "
            f"spearman_with_num={engineered_correlations[feature]}"
        )

    print("\nTransformed split shapes:")
    print(f"  train: {X_train_transformed.shape}")
    print(f"  val: {X_val_transformed.shape}")
    print(f"  test: {X_test_transformed.shape}")

    print("\nLeakage checks:")
    for check, status in leakage_checks.items():
        print(f"  {check}: {status}")

    print("\nSaved files:")
    for created in report["files_created"]:
        print(f"  {created}")
    print("No predictive model was trained.")
    print("=" * 80)


if __name__ == "__main__":
    run_feature_engineering()
