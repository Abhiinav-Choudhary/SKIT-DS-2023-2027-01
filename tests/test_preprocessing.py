"""Unit tests for Phase 5 Data Cleaning and Preprocessing Pipeline.

Author: Abhiram (Lead AI/ML Engineer)
Date: 8 October 2026
"""

import pytest
import pandas as pd
import numpy as np
import joblib

from src.preprocessing.data_loader import load_cdc_diabetes_data
from src.preprocessing.data_cleaner import audit_data_quality, clean_cdc_diabetes_data
from src.preprocessing.pipeline import (
    create_preprocessor,
    stratified_split,
    build_and_fit_pipeline,
    CONTINUOUS_NUMERICAL_FEATURES,
    ALL_FEATURE_COLUMNS,
)


@pytest.fixture(scope="module")
def raw_data():
    """Load raw dataset once for all tests in module."""
    return load_cdc_diabetes_data(variant="binary")


def test_audit_data_quality_structure(raw_data):
    """Verify data quality audit returns all mandatory sections."""
    audit = audit_data_quality(raw_data)
    assert "missing_values" in audit
    assert "invalid_values" in audit
    assert "suspicious_values" in audit
    assert "identifier_columns" in audit
    assert "target_leakage_risks" in audit
    assert "duplicate_rows" in audit
    assert "class_imbalance" in audit
    assert "feature_types" in audit

    assert audit["missing_values"]["total"] == 0
    assert audit["duplicate_rows"]["exact_duplicate_rows"] == 24206


def test_cleaning_deduplication(raw_data):
    """Verify clean_cdc_diabetes_data removes exact duplicates correctly."""
    df_cleaned, meta = clean_cdc_diabetes_data(raw_data, drop_duplicates=True)
    assert len(df_cleaned) == 229474
    assert meta["dropped_rows"] == 24206
    assert df_cleaned.duplicated().sum() == 0
    # Check that all columns are downcast to integer types
    assert df_cleaned["BMI"].dtype == np.int16
    assert df_cleaned["HighBP"].dtype == np.int8


def test_stratified_split(raw_data):
    """Verify train/val/test split produces correct partitions with preserved stratification."""
    df_cleaned, _ = clean_cdc_diabetes_data(raw_data, drop_duplicates=True)
    train_df, val_df, test_df = stratified_split(
        df_cleaned, target_col="Diabetes_binary", test_size=0.15, val_size=0.15, random_state=42
    )

    total_len = len(df_cleaned)
    assert len(train_df) + len(val_df) + len(test_df) == total_len

    # Expected proportions ~ 70%, 15%, 15%
    assert abs(len(train_df) / total_len - 0.70) < 0.01
    assert abs(len(val_df) / total_len - 0.15) < 0.01
    assert abs(len(test_df) / total_len - 0.15) < 0.01

    # Check stratification preservation
    p_full = df_cleaned["Diabetes_binary"].mean()
    p_train = train_df["Diabetes_binary"].mean()
    p_val = val_df["Diabetes_binary"].mean()
    p_test = test_df["Diabetes_binary"].mean()

    assert abs(p_train - p_full) < 0.005
    assert abs(p_val - p_full) < 0.005
    assert abs(p_test - p_full) < 0.005


def test_preprocessor_transformation(raw_data):
    """Verify ColumnTransformer scales continuous features without data leakage."""
    df_cleaned, _ = clean_cdc_diabetes_data(raw_data, drop_duplicates=True)
    train_df, val_df, _ = stratified_split(df_cleaned, test_size=0.15, val_size=0.15, random_state=42)

    # Verify pipeline without feature engineering (21 base features)
    preprocessor_base, X_train_base = build_and_fit_pipeline(
        train_df, scaling="standard", use_feature_engineering=False
    )
    assert X_train_base.shape[1] == 21

    # Verify pipeline with feature engineering (27 features)
    preprocessor_eng, X_train_eng = build_and_fit_pipeline(
        train_df, scaling="standard", use_feature_engineering=True
    )
    assert X_train_eng.shape[1] == 27

    # Verify numerical features are standardized (mean ~ 0, std ~ 1) on train set
    for num_col in CONTINUOUS_NUMERICAL_FEATURES:
        assert abs(X_train_base[num_col].mean()) < 0.05
        assert abs(X_train_base[num_col].std() - 1.0) < 0.05

    # Verify validation set transformation works seamlessly
    X_val = val_df.drop(columns=["Diabetes_binary"])
    X_val_trans = preprocessor_eng.transform(X_val)
    assert X_val_trans.shape == (len(val_df), 27)
