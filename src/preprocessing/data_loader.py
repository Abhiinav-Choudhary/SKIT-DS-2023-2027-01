"""Data Loading and Ingestion Module.

Project: AI-Powered Health Risk Prediction and Monitoring System
Dataset: CDC Diabetes Health Indicators (BRFSS 2015) & UCI Heart Disease
Author: Abhiram (AI/ML Engineer)
"""

import os
from pathlib import Path
from typing import Literal, Tuple, Dict, Any
import pandas as pd

# Define base project directory relative to this file
SRC_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = SRC_DIR.parent
DATA_DIR = REPO_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

CDC_DIABETES_FILES: Dict[str, str] = {
    "binary": "diabetes_binary_health_indicators_BRFSS2015.csv",
    "012": "diabetes_012_health_indicators_BRFSS2015.csv",
    "5050split": "diabetes_binary_5050split_health_indicators_BRFSS2015.csv",
}

# Expected schema for CDC Diabetes dataset
EXPECTED_CDC_COLUMNS = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income"
]


def get_data_dir() -> Path:
    """Return the base data directory path."""
    return DATA_DIR


def load_cdc_diabetes_data(
    variant: Literal["binary", "012", "5050split"] = "binary"
) -> pd.DataFrame:
    """Load the specified variant of the CDC Diabetes Health Indicators dataset.

    Args:
        variant: Which dataset version to load:
            - 'binary': Full 253,680 records, binary target (0 = no diabetes, 1 = diabetes).
            - '012': Full 253,680 records, 3-class target (0 = no, 1 = prediabetes, 2 = diabetes).
            - '5050split': 70,692 records, balanced 50/50 split of binary target.

    Returns:
        pd.DataFrame containing the dataset.

    Raises:
        FileNotFoundError: If the CSV file does not exist in data/raw/diabetes/.
        ValueError: If an unrecognized variant is requested or columns mismatch.
    """
    if variant not in CDC_DIABETES_FILES:
        raise ValueError(f"Unknown variant '{variant}'. Choose from {list(CDC_DIABETES_FILES.keys())}")

    filename = CDC_DIABETES_FILES[variant]
    file_path = RAW_DATA_DIR / "diabetes" / filename

    if not file_path.exists():
        raise FileNotFoundError(
            f"CDC Diabetes dataset file not found at: {file_path}. "
            f"Please verify data/raw/diabetes/ contains the extracted CSVs."
        )

    df = pd.read_csv(file_path)

    # Basic schema validation
    target_col = "Diabetes_012" if variant == "012" else "Diabetes_binary"
    expected_all = [target_col] + EXPECTED_CDC_COLUMNS

    missing_cols = set(expected_all) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Dataset at {file_path} is missing expected columns: {missing_cols}")

    return df
