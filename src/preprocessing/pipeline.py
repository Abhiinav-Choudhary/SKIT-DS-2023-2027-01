"""Reproducible Scikit-Learn Preprocessing Pipelines and Leakage-Free Splitting.

Project: AI-Powered Health Risk Prediction and Monitoring System
Author: Abhiram (Lead AI/ML Engineer)
Date: 8 October 2026
"""

from pathlib import Path
from typing import Tuple, Dict, Any, Literal, List, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split

from src.features.feature_engineer import DiabetesFeatureEngineer, ENGINEERED_FEATURE_NAMES

# Base feature classifications (21 original features)
BASE_CONTINUOUS_NUMERICAL = ["BMI", "MentHlth", "PhysHlth"]
BASE_ORDINAL_FEATURES = ["GenHlth", "Age", "Education", "Income"]
BASE_BINARY_FEATURES = [
    "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "DiffWalk", "Sex"
]

ALL_BASE_FEATURE_COLUMNS = BASE_BINARY_FEATURES + BASE_ORDINAL_FEATURES + BASE_CONTINUOUS_NUMERICAL

# Backward compatibility aliases
CONTINUOUS_NUMERICAL_FEATURES = BASE_CONTINUOUS_NUMERICAL
ORDINAL_FEATURES = BASE_ORDINAL_FEATURES
BINARY_FEATURES = BASE_BINARY_FEATURES
ALL_FEATURE_COLUMNS = ALL_BASE_FEATURE_COLUMNS

# Engineered feature classification additions
ENGINEERED_NUMERICAL = ["Age_x_GenHlth"]
ENGINEERED_ORDINAL = ["BMI_Category", "CardioMetabolic_Risk_Score", "Healthy_Lifestyle_Score"]
ENGINEERED_BINARY = ["HighBP_x_HighChol", "Sedentary_Obese"]


def create_preprocessor(
    scaling: Literal["standard", "robust", "none"] = "standard",
    include_engineered: bool = False,
) -> ColumnTransformer:
    """Create a ColumnTransformer preprocessor for tabular health indicators.

    Args:
      scaling: 'standard' for StandardScaler, 'robust' for RobustScaler, 'none' for passthrough.
      include_engineered: If True, includes engineered features in column transformer.

    Returns:
      Configured ColumnTransformer instance.
    """
    if scaling == "standard":
        num_transformer = StandardScaler()
    elif scaling == "robust":
        num_transformer = RobustScaler()
    elif scaling == "none":
        num_transformer = "passthrough"
    else:
        raise ValueError(f"Unsupported scaling mode: {scaling}")

    num_cols = BASE_CONTINUOUS_NUMERICAL + (ENGINEERED_NUMERICAL if include_engineered else [])
    ord_cols = BASE_ORDINAL_FEATURES + (ENGINEERED_ORDINAL if include_engineered else [])
    bin_cols = BASE_BINARY_FEATURES + (ENGINEERED_BINARY if include_engineered else [])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_cols),
            ("ord", "passthrough", ord_cols),
            ("bin", "passthrough", bin_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def create_full_pipeline(
    scaling: Literal["standard", "robust", "none"] = "standard",
    use_feature_engineering: bool = True,
) -> Pipeline:
    """Create end-to-end sklearn Pipeline integrating feature engineering and preprocessing."""
    if use_feature_engineering:
        steps = [
            ("feature_engineer", DiabetesFeatureEngineer(include_original=True)),
            ("preprocessor", create_preprocessor(scaling=scaling, include_engineered=True)),
        ]
    else:
        steps = [
            ("preprocessor", create_preprocessor(scaling=scaling, include_engineered=False)),
        ]
    return Pipeline(steps=steps)


def stratified_split(
    df: pd.DataFrame,
    target_col: str = "Diabetes_binary",
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Perform a strictly stratified train/val/test split to prevent data leakage."""
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


def build_and_fit_pipeline(
    train_df: pd.DataFrame,
    target_col: str = "Diabetes_binary",
    scaling: Literal["standard", "robust", "none"] = "standard",
    use_feature_engineering: bool = True,
) -> Tuple[Pipeline, pd.DataFrame]:
    """Fit full preprocessing pipeline strictly on training data and transform it."""
    X_train = train_df.drop(columns=[target_col])
    pipeline = create_full_pipeline(scaling=scaling, use_feature_engineering=use_feature_engineering)
    X_train_transformed = pipeline.fit_transform(X_train)

    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    X_train_df = pd.DataFrame(X_train_transformed, columns=feature_names)

    return pipeline, X_train_df


def save_processed_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
) -> Dict[str, Path]:
    """Save split partitions to CSV for downstream training and auditability."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": output_dir / "diabetes_train.csv",
        "val": output_dir / "diabetes_val.csv",
        "test": output_dir / "diabetes_test.csv",
    }
    train_df.to_csv(paths["train"], index=False)
    val_df.to_csv(paths["val"], index=False)
    test_df.to_csv(paths["test"], index=False)
    return paths
