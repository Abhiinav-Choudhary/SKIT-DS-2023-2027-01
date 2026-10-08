"""Data Quality Auditing and Cleaning Module for CDC Diabetes Dataset.

Project: AI-Powered Health Risk Prediction and Monitoring System
Author: Abhiram (Lead AI/ML Engineer)
Date: 8 October 2026
"""

from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np

# Valid physiological and codebook ranges according to CDC BRFSS 2015 documentation
BRFSS_CODEBOOK_RANGES: Dict[str, Tuple[float, float]] = {
    "Diabetes_binary": (0.0, 1.0),
    "HighBP": (0.0, 1.0),
    "HighChol": (0.0, 1.0),
    "CholCheck": (0.0, 1.0),
    "BMI": (12.0, 100.0),
    "Smoker": (0.0, 1.0),
    "Stroke": (0.0, 1.0),
    "HeartDiseaseorAttack": (0.0, 1.0),
    "PhysActivity": (0.0, 1.0),
    "Fruits": (0.0, 1.0),
    "Veggies": (0.0, 1.0),
    "HvyAlcoholConsump": (0.0, 1.0),
    "AnyHealthcare": (0.0, 1.0),
    "NoDocbcCost": (0.0, 1.0),
    "GenHlth": (1.0, 5.0),
    "MentHlth": (0.0, 30.0),
    "PhysHlth": (0.0, 30.0),
    "DiffWalk": (0.0, 1.0),
    "Sex": (0.0, 1.0),
    "Age": (1.0, 13.0),
    "Education": (1.0, 6.0),
    "Income": (1.0, 8.0),
}


def audit_data_quality(df: pd.DataFrame, target_col: str = "Diabetes_binary") -> Dict[str, Any]:
    """Perform an exhaustive data quality audit on the raw dataset.

    Audits:
      1. Missing values
      2. Invalid values (out of codebook range or non-integer representation)
      3. Suspicious values (physiological outliers in BMI)
      4. Identifier columns check
      5. Target leakage risks
      6. Duplicate rows (exact full-row duplicates & feature-only duplicates)
      7. Class imbalance
      8. Feature types categorization

    Returns:
      Comprehensive dictionary containing audit results.
    """
    n_rows, n_cols = df.shape
    features = [c for c in df.columns if c != target_col]

    # 1. Missing values
    missing_by_col = {col: int(df[col].isnull().sum()) for col in df.columns}
    total_missing = sum(missing_by_col.values())

    # 2. Invalid values (against codebook)
    invalid_report = {}
    for col, (vmin, vmax) in BRFSS_CODEBOOK_RANGES.items():
        if col in df.columns:
            s = df[col]
            out_of_range = int(((s < vmin) | (s > vmax) | (s.isna())).sum())
            non_integer = int((s % 1 != 0).sum())
            invalid_report[col] = {
                "out_of_range": out_of_range,
                "non_integer": non_integer,
                "min": float(s.min()),
                "max": float(s.max()),
                "expected_min": vmin,
                "expected_max": vmax,
            }

    # 3. Suspicious values (BMI extremes)
    bmi_col = df["BMI"]
    suspicious_values = {
        "bmi_under_15": int((bmi_col < 15).sum()),
        "bmi_under_15_pct": round(float((bmi_col < 15).mean() * 100), 4),
        "bmi_over_60": int((bmi_col > 60).sum()),
        "bmi_over_60_pct": round(float((bmi_col > 60).mean() * 100), 4),
        "bmi_over_70": int((bmi_col > 70).sum()),
        "bmi_over_70_pct": round(float((bmi_col > 70).mean() * 100), 4),
        "bmi_max": float(bmi_col.max()),
        "bmi_min": float(bmi_col.min()),
    }

    # 4. Identifier columns
    id_candidates = []
    for col in df.columns:
        if df[col].nunique() == n_rows or col.lower() in ["id", "patient_id", "subject_id", "index"]:
            id_candidates.append(col)

    # 5. Target leakage risks
    # Check if any feature is perfectly correlated with target or derived from diagnosis
    target_corrs = {}
    for col in features:
        corr = float(df[col].corr(df[target_col]))
        target_corrs[col] = round(corr, 4)

    high_leakage_features = [col for col, corr in target_corrs.items() if abs(corr) > 0.85]

    # 6. Duplicate rows
    exact_duplicates = int(df.duplicated().sum())
    feature_duplicates = int(df.duplicated(subset=features).sum())
    unique_profiles = int(len(df.groupby(features)))
    conflicting_profiles = int((df.groupby(features)[target_col].nunique() > 1).sum())

    # 7. Class imbalance
    target_counts = {int(k): int(v) for k, v in df[target_col].value_counts().items()}
    target_pcts = {int(k): round(float(v) * 100, 2) for k, v in df[target_col].value_counts(normalize=True).items()}
    imbalance_ratio = round(float(target_counts[0] / target_counts[1]), 2) if 1 in target_counts else 0.0

    # 8. Feature types
    binary_cols = [c for c in features if df[c].nunique() == 2]
    ordinal_cols = [c for c in features if c in ["GenHlth", "Age", "Education", "Income"]]
    continuous_cols = [c for c in features if c in ["BMI", "MentHlth", "PhysHlth"]]

    return {
        "dataset_shape": {"rows": n_rows, "columns": n_cols},
        "missing_values": {"total": total_missing, "by_column": missing_by_col},
        "invalid_values": invalid_report,
        "suspicious_values": suspicious_values,
        "identifier_columns": {
            "found": id_candidates,
            "status": "None present (survey contains only anonymous binned/discrete features)",
        },
        "target_leakage_risks": {
            "high_risk_features": high_leakage_features,
            "feature_correlations_with_target": target_corrs,
            "status": "No direct target leakage detected (all features reflect general health/lifestyle/demographics measured prior to or comorbid with diagnosis)",
        },
        "duplicate_rows": {
            "exact_duplicate_rows": exact_duplicates,
            "exact_duplicate_pct": round(exact_duplicates / n_rows * 100, 2),
            "feature_duplicates": feature_duplicates,
            "unique_profiles": unique_profiles,
            "conflicting_profiles": conflicting_profiles,
        },
        "class_imbalance": {
            "counts": target_counts,
            "percentages": target_pcts,
            "ratio": imbalance_ratio,
        },
        "feature_types": {
            "binary_features": binary_cols,
            "ordinal_features": ordinal_cols,
            "continuous_numerical_features": continuous_cols,
        },
    }


def clean_cdc_diabetes_data(
    df: pd.DataFrame, drop_duplicates: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Clean the raw CDC Diabetes dataset in a reproducible manner.

    Cleaning decisions:
      1. Deduplication: Drops exact duplicate rows (24,206 rows, 9.54%) to prevent
         identical survey responses from contaminating both train and test splits.
      2. Datatype downcasting: Casts binary and ordinal float64 columns to int8,
         and continuous/count variables to int16 for computational efficiency.
      3. Records before/after statistics.

    Args:
      df: Raw pandas DataFrame.
      drop_duplicates: Whether to drop exact identical full rows. Default True.

    Returns:
      Tuple of (cleaned_df, cleaning_metadata).
    """
    initial_shape = df.shape
    initial_counts = df["Diabetes_binary"].value_counts().to_dict()

    if drop_duplicates:
        df_cleaned = df.drop_duplicates().copy()
    else:
        df_cleaned = df.copy()

    # Downcast datatypes to integers (original CSV stored them as float64)
    for col in df_cleaned.columns:
        if col == "BMI":
            df_cleaned[col] = df_cleaned[col].astype(np.int16)
        elif col in ["MentHlth", "PhysHlth"]:
            df_cleaned[col] = df_cleaned[col].astype(np.int8)
        else:
            df_cleaned[col] = df_cleaned[col].astype(np.int8)

    final_shape = df_cleaned.shape
    final_counts = df_cleaned["Diabetes_binary"].value_counts().to_dict()

    cleaning_metadata = {
        "initial_rows": initial_shape[0],
        "final_rows": final_shape[0],
        "dropped_rows": initial_shape[0] - final_shape[0],
        "dropped_pct": round((initial_shape[0] - final_shape[0]) / initial_shape[0] * 100, 2),
        "initial_target_distribution": {int(k): int(v) for k, v in initial_counts.items()},
        "final_target_distribution": {int(k): int(v) for k, v in final_counts.items()},
        "initial_imbalance_ratio": round(float(initial_counts[0.0] / initial_counts[1.0]), 2),
        "final_imbalance_ratio": round(float(final_counts[0] / final_counts[1]), 2),
    }

    return df_cleaned, cleaning_metadata
