"""Data loading, auditing, and preprocessing for the UCI Heart Disease data.

This module uses the four processed UCI Heart Disease files present in
data/raw/heart_disease. It does not train predictive models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Literal, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.heart_disease_features import (
    HEART_ENGINEERED_CATEGORICAL_FEATURES,
    HEART_ENGINEERED_NUMERIC_FEATURES,
    HeartDiseaseFeatureEngineer,
)


SRC_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = SRC_DIR.parent
RAW_HEART_DISEASE_DIR = REPO_ROOT / "data" / "raw" / "heart_disease"

HEART_DISEASE_FILES: Dict[str, str] = {
    "cleveland": "processed.cleveland.data",
    "hungarian": "processed.hungarian.data",
    "switzerland": "processed.switzerland.data",
    "va": "processed.va.data",
}

HEART_DISEASE_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num",
]

HEART_DISEASE_TARGET = "num"
SOURCE_COLUMN = "source"

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
ZERO_AS_MISSING_MEASUREMENT_COLUMNS = ["trestbps", "chol"]
NUMERIC_FEATURES_WITH_ENGINEERING = NUMERIC_FEATURES + HEART_ENGINEERED_NUMERIC_FEATURES
CATEGORICAL_FEATURES_WITH_ENGINEERING = CATEGORICAL_FEATURES + HEART_ENGINEERED_CATEGORICAL_FEATURES

HeartDiseaseSource = Literal["all", "cleveland", "hungarian", "switzerland", "va"]


def load_heart_disease_data(
    source: HeartDiseaseSource = "all",
    include_source: bool = True,
) -> pd.DataFrame:
    """Load one or all processed UCI Heart Disease data files.

    The raw files use '?' as the missing-value marker and have no header row.
    """
    if source == "all":
        selected_sources = list(HEART_DISEASE_FILES)
    elif source in HEART_DISEASE_FILES:
        selected_sources = [source]
    else:
        raise ValueError(f"Unknown heart disease source '{source}'. Choose from {['all'] + list(HEART_DISEASE_FILES)}")

    frames = []
    for source_name in selected_sources:
        file_path = RAW_HEART_DISEASE_DIR / HEART_DISEASE_FILES[source_name]
        if not file_path.exists():
            raise FileNotFoundError(f"Heart Disease data file not found: {file_path}")

        df = pd.read_csv(
            file_path,
            header=None,
            names=HEART_DISEASE_COLUMNS,
            na_values="?",
        )
        for column in HEART_DISEASE_COLUMNS:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        if include_source:
            df.insert(0, SOURCE_COLUMN, source_name)
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def audit_heart_disease_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Return reproducible data-understanding statistics for Heart Disease data."""
    target_values = sorted(df[HEART_DISEASE_TARGET].dropna().unique().tolist())
    target_counts = df[HEART_DISEASE_TARGET].value_counts(dropna=False).sort_index()
    target_percentages = df[HEART_DISEASE_TARGET].value_counts(normalize=True, dropna=False).sort_index()

    duplicate_report = {"exact_duplicate_rows": int(df.duplicated().sum())}
    if SOURCE_COLUMN in df.columns:
        duplicate_report["exact_duplicate_rows_excluding_source"] = int(
            df.drop(columns=[SOURCE_COLUMN]).duplicated().sum()
        )

    source_counts = {}
    if SOURCE_COLUMN in df.columns:
        source_counts = {str(k): int(v) for k, v in df[SOURCE_COLUMN].value_counts().sort_index().items()}

    stats_df = df[HEART_DISEASE_COLUMNS].describe().round(3)

    return {
        "dataset_shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "column_names": list(df.columns),
        "data_types": {column: str(dtype) for column, dtype in df.dtypes.items()},
        "target": {
            "column": HEART_DISEASE_TARGET,
            "values": [int(value) for value in target_values],
            "class_counts": {int(k): int(v) for k, v in target_counts.items()},
            "class_percentages": {int(k): round(float(v) * 100, 2) for k, v in target_percentages.items()},
        },
        "missing_values": {
            "total": int(df.isna().sum().sum()),
            "by_column": {column: int(count) for column, count in df.isna().sum().items()},
        },
        "duplicate_rows": duplicate_report,
        "source_distribution": source_counts,
        "basic_statistics": stats_df.to_dict(),
        "zero_measurement_counts": {
            column: int((df[column] == 0).sum())
            for column in ["trestbps", "chol", "thalach", "oldpeak"]
            if column in df.columns
        },
        "feature_groups": {
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
        },
    }


def clean_heart_disease_data(
    df: pd.DataFrame,
    drop_duplicates: bool = True,
    zero_measurements_as_missing: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Clean the Heart Disease data without fitting a predictive model."""
    cleaned = df.copy()
    initial_shape = cleaned.shape
    initial_target_counts = cleaned[HEART_DISEASE_TARGET].value_counts(dropna=False).sort_index()

    for column in HEART_DISEASE_COLUMNS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    zero_to_missing = {}
    if zero_measurements_as_missing:
        for column in ZERO_AS_MISSING_MEASUREMENT_COLUMNS:
            zero_mask = cleaned[column] == 0
            zero_to_missing[column] = int(zero_mask.sum())
            cleaned.loc[zero_mask, column] = np.nan

    if drop_duplicates:
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    else:
        cleaned = cleaned.reset_index(drop=True)

    if cleaned[HEART_DISEASE_TARGET].isna().any():
        raise ValueError("Heart Disease target column contains missing values after cleaning.")
    cleaned[HEART_DISEASE_TARGET] = cleaned[HEART_DISEASE_TARGET].astype(np.int8)

    final_target_counts = cleaned[HEART_DISEASE_TARGET].value_counts(dropna=False).sort_index()
    metadata = {
        "initial_shape": {"rows": int(initial_shape[0]), "columns": int(initial_shape[1])},
        "final_shape": {"rows": int(cleaned.shape[0]), "columns": int(cleaned.shape[1])},
        "dropped_duplicate_rows": int(initial_shape[0] - cleaned.shape[0]),
        "zero_measurements_converted_to_missing": zero_to_missing,
        "missing_values_after_cleaning": {
            "total": int(cleaned.isna().sum().sum()),
            "by_column": {column: int(count) for column, count in cleaned.isna().sum().items()},
        },
        "initial_target_distribution": {int(k): int(v) for k, v in initial_target_counts.items()},
        "final_target_distribution": {int(k): int(v) for k, v in final_target_counts.items()},
    }
    return cleaned, metadata


def stratified_heart_disease_split(
    df: pd.DataFrame,
    target_col: str = HEART_DISEASE_TARGET,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create train/validation/test partitions stratified by the target column."""
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[target_col],
        random_state=random_state,
        shuffle=True,
    )

    adjusted_val_size = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=adjusted_val_size,
        stratify=train_val_df[target_col],
        random_state=random_state,
        shuffle=True,
    )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def create_heart_disease_preprocessor() -> ColumnTransformer:
    """Create a ColumnTransformer for Heart Disease numeric and categorical fields."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def fit_heart_disease_preprocessor(
    train_df: pd.DataFrame,
    target_col: str = HEART_DISEASE_TARGET,
) -> Tuple[ColumnTransformer, pd.DataFrame]:
    """Fit the preprocessing transformer on training data only."""
    preprocessor = create_heart_disease_preprocessor()
    feature_df = train_df.drop(columns=[target_col, SOURCE_COLUMN], errors="ignore")
    transformed = preprocessor.fit_transform(feature_df)
    feature_names = preprocessor.get_feature_names_out()
    transformed_df = pd.DataFrame(transformed, columns=feature_names)
    return preprocessor, transformed_df


def create_heart_disease_feature_pipeline() -> Pipeline:
    """Create feature engineering plus preprocessing pipeline for Heart Disease data."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES_WITH_ENGINEERING),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES_WITH_ENGINEERING),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline(
        steps=[
            ("feature_engineer", HeartDiseaseFeatureEngineer(include_original=True)),
            ("preprocessor", preprocessor),
        ]
    )


def fit_heart_disease_feature_pipeline(
    train_df: pd.DataFrame,
    target_col: str = HEART_DISEASE_TARGET,
) -> Tuple[Pipeline, pd.DataFrame]:
    """Fit the full feature engineering/preprocessing pipeline on training data only."""
    pipeline = create_heart_disease_feature_pipeline()
    feature_df = train_df.drop(columns=[target_col, SOURCE_COLUMN], errors="ignore")
    transformed = pipeline.fit_transform(feature_df)
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    transformed_df = pd.DataFrame(transformed, columns=feature_names)
    return pipeline, transformed_df


def save_heart_disease_processed_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
) -> Dict[str, Path]:
    """Persist Heart Disease train/validation/test partitions."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": output_dir / "heart_disease_train.csv",
        "val": output_dir / "heart_disease_val.csv",
        "test": output_dir / "heart_disease_test.csv",
    }
    train_df.to_csv(paths["train"], index=False)
    val_df.to_csv(paths["val"], index=False)
    test_df.to_csv(paths["test"], index=False)
    return paths
