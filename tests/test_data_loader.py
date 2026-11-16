"""Unit tests for dataset ingestion and schema validation.

Author: Abhiram (AI/ML Engineer)
Date: 8 October 2026
"""

import pytest
from src.preprocessing.data_loader import load_cdc_diabetes_data, CDC_DIABETES_FILES, EXPECTED_CDC_COLUMNS


def test_load_cdc_diabetes_binary_shape_and_columns():
    """Verify primary binary dataset loads with expected dimensions and columns."""
    df = load_cdc_diabetes_data(variant="binary")
    assert df.shape == (253680, 22), f"Expected shape (253680, 22), got {df.shape}"
    assert "Diabetes_binary" in df.columns
    for col in EXPECTED_CDC_COLUMNS:
        assert col in df.columns, f"Missing expected feature column: {col}"


def test_load_cdc_diabetes_012():
    """Verify multiclass dataset loads correctly."""
    df = load_cdc_diabetes_data(variant="012")
    assert df.shape == (253680, 22)
    assert "Diabetes_012" in df.columns
    unique_targets = set(df["Diabetes_012"].unique())
    assert unique_targets == {0.0, 1.0, 2.0}


def test_load_cdc_diabetes_invalid_variant():
    """Verify invalid variant name raises ValueError."""
    with pytest.raises(ValueError):
        load_cdc_diabetes_data(variant="non_existent_variant")


def test_no_null_values_in_raw_data():
    """Verify raw CDC Diabetes dataset has no missing values."""
    df = load_cdc_diabetes_data(variant="binary")
    assert df.isnull().sum().sum() == 0, "Expected zero null values in CDC Diabetes dataset"
