"""Heart Disease data understanding and preprocessing.

Loads the actual UCI Heart Disease processed files in data/raw/heart_disease,
reports data-understanding results, applies preprocessing steps, and persists
reports plus processed train/validation/test partitions. No predictive model is
trained in this script.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.heart_disease import (  # noqa: E402
    HEART_DISEASE_COLUMNS,
    HEART_DISEASE_TARGET,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    audit_heart_disease_data,
    clean_heart_disease_data,
    fit_heart_disease_preprocessor,
    load_heart_disease_data,
    save_heart_disease_processed_datasets,
    stratified_heart_disease_split,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _print_mapping(title: str, mapping: dict) -> None:
    print(title)
    for key, value in mapping.items():
        print(f"  {key}: {value}")


def run_heart_disease_understanding_and_preprocessing() -> None:
    print("=" * 80)
    print("HEART DISEASE DATA UNDERSTANDING AND PREPROCESSING")
    print("=" * 80)

    print("\n[STEP 1] Loading actual processed UCI Heart Disease files...")
    raw_df = load_heart_disease_data(source="all", include_source=True)
    audit = audit_heart_disease_data(raw_df)

    print(f"Dataset shape: {raw_df.shape[0]} rows x {raw_df.shape[1]} columns")
    print(f"Original data columns: {HEART_DISEASE_COLUMNS}")
    print(f"Loaded columns: {list(raw_df.columns)}")
    _print_mapping("Source distribution:", audit["source_distribution"])
    _print_mapping("Data types:", audit["data_types"])
    print(f"Target column: {audit['target']['column']}")
    print(f"Target values: {audit['target']['values']}")
    _print_mapping("Target class counts:", audit["target"]["class_counts"])
    _print_mapping("Target class percentages:", audit["target"]["class_percentages"])
    print(f"Missing values total: {audit['missing_values']['total']}")
    _print_mapping("Missing values by column:", audit["missing_values"]["by_column"])
    _print_mapping("Duplicate rows:", audit["duplicate_rows"])
    _print_mapping("Zero counts in measurement fields:", audit["zero_measurement_counts"])

    print("\nBasic statistics:")
    print(raw_df[HEART_DISEASE_COLUMNS].describe().round(3).to_string())

    understanding_report_path = PROJECT_ROOT / "reports" / "heart_disease_data_understanding_report.json"
    _write_json(understanding_report_path, audit)
    print(f"\n[OK] Data understanding report saved to: {understanding_report_path.relative_to(PROJECT_ROOT)}")

    print("\n[STEP 2] Cleaning Heart Disease data...")
    cleaned_df, cleaning_metadata = clean_heart_disease_data(
        raw_df,
        drop_duplicates=True,
        zero_measurements_as_missing=True,
    )
    cleaned_path = PROJECT_ROOT / "data" / "processed" / "heart_disease" / "heart_disease_cleaned.csv"
    cleaned_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(cleaned_path, index=False)

    print(f"Cleaned shape: {cleaned_df.shape[0]} rows x {cleaned_df.shape[1]} columns")
    _print_mapping("Zero measurements converted to missing:", cleaning_metadata["zero_measurements_converted_to_missing"])
    _print_mapping("Missing after cleaning:", cleaning_metadata["missing_values_after_cleaning"]["by_column"])
    _print_mapping("Final target distribution:", cleaning_metadata["final_target_distribution"])
    print(f"[OK] Cleaned dataset saved to: {cleaned_path.relative_to(PROJECT_ROOT)}")

    print("\n[STEP 3] Creating stratified train/validation/test splits...")
    train_df, val_df, test_df = stratified_heart_disease_split(
        cleaned_df,
        target_col=HEART_DISEASE_TARGET,
        test_size=0.15,
        val_size=0.15,
        random_state=42,
    )
    processed_paths = save_heart_disease_processed_datasets(
        train_df,
        val_df,
        test_df,
        PROJECT_ROOT / "data" / "processed" / "heart_disease",
    )
    for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        counts = {int(k): int(v) for k, v in part[HEART_DISEASE_TARGET].value_counts().sort_index().items()}
        print(f"{name}: shape={part.shape}, target_counts={counts}")
    for name, path in processed_paths.items():
        print(f"[OK] {name} split saved to: {path.relative_to(PROJECT_ROOT)}")

    print("\n[STEP 4] Fitting preprocessing transformer on training data only...")
    preprocessor, transformed_train = fit_heart_disease_preprocessor(
        train_df,
        target_col=HEART_DISEASE_TARGET,
    )
    models_dir = PROJECT_ROOT / "models" / "heart"
    models_dir.mkdir(parents=True, exist_ok=True)
    preprocessor_path = models_dir / "heart_disease_preprocessor.joblib"
    joblib.dump(preprocessor, preprocessor_path)
    print(f"Numeric features: {NUMERIC_FEATURES}")
    print(f"Categorical features: {CATEGORICAL_FEATURES}")
    print(f"Transformed training matrix shape: {transformed_train.shape}")
    print(f"Transformed feature names: {list(transformed_train.columns)}")
    print(f"[OK] Preprocessor saved to: {preprocessor_path.relative_to(PROJECT_ROOT)}")

    preprocessing_report = {
        "cleaning": cleaning_metadata,
        "splits": {
            "train_shape": [int(train_df.shape[0]), int(train_df.shape[1])],
            "val_shape": [int(val_df.shape[0]), int(val_df.shape[1])],
            "test_shape": [int(test_df.shape[0]), int(test_df.shape[1])],
            "train_target_distribution": {
                int(k): int(v) for k, v in train_df[HEART_DISEASE_TARGET].value_counts().sort_index().items()
            },
            "val_target_distribution": {
                int(k): int(v) for k, v in val_df[HEART_DISEASE_TARGET].value_counts().sort_index().items()
            },
            "test_target_distribution": {
                int(k): int(v) for k, v in test_df[HEART_DISEASE_TARGET].value_counts().sort_index().items()
            },
        },
        "preprocessor": {
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "transformed_train_shape": [int(transformed_train.shape[0]), int(transformed_train.shape[1])],
            "transformed_feature_names": list(transformed_train.columns),
            "artifact": str(preprocessor_path.relative_to(PROJECT_ROOT)),
        },
    }
    preprocessing_report_path = PROJECT_ROOT / "reports" / "heart_disease_preprocessing_report.json"
    _write_json(preprocessing_report_path, preprocessing_report)
    print(f"[OK] Preprocessing report saved to: {preprocessing_report_path.relative_to(PROJECT_ROOT)}")

    print("\nNo predictive model was trained.")
    print("=" * 80)


if __name__ == "__main__":
    run_heart_disease_understanding_and_preprocessing()
