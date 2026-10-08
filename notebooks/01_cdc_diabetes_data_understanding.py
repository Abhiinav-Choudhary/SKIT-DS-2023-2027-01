"""Data Understanding Script for CDC Diabetes Health Indicators.

This script performs rigorous ingestion, schema validation, data quality checks,
missing value analysis, duplicate row inspection, target distribution analysis,
and descriptive statistics on the CDC Diabetes Health Indicators dataset.

Project: AI-Powered Health Risk Prediction and Monitoring System
Author: Abhiram (AI/ML Engineer)
Date: 8 October 2026
"""

import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from src.preprocessing.data_loader import load_cdc_diabetes_data
from src.features.cdc_diabetes_metadata import (
    CDC_FEATURE_METADATA,
    BINARY_FEATURES,
    ORDINAL_FEATURES,
    CONTINUOUS_NUMERICAL_FEATURES,
)


def run_data_understanding():
    print("=" * 80)
    print("PHASE 4: CDC DIABETES HEALTH INDICATORS - DATA UNDERSTANDING")
    print("=" * 80)

    # 1. Ingestion
    print("\n[1] Ingesting Primary Dataset: diabetes_binary_health_indicators_BRFSS2015.csv...")
    df = load_cdc_diabetes_data(variant="binary")
    df_012 = load_cdc_diabetes_data(variant="012")
    df_5050 = load_cdc_diabetes_data(variant="5050split")

    n_rows, n_cols = df.shape
    print(f"Loaded successfully.")
    print(f"Shape: {n_rows:,} rows x {n_cols} columns")

    # 2. Schema and Datatypes
    print("\n[2] Column Names & Datatypes:")
    dtypes_summary = {}
    for col in df.columns:
        dtype_str = str(df[col].dtype)
        meta = CDC_FEATURE_METADATA.get(col, {})
        role = meta.get("role", "Unknown")
        feat_type = meta.get("type", "Unknown")
        dtypes_summary[col] = {
            "dtype": dtype_str,
            "role": role,
            "type": feat_type,
            "n_unique": int(df[col].nunique()),
        }
        print(f"  - {col:22s} | Type: {dtype_str:8s} | Unique: {df[col].nunique():2d} | Role: {role:7s} | Feature Type: {feat_type}")

    # 3. Missing Value Analysis
    print("\n[3] Missing Values Report:")
    missing_series = df.isnull().sum()
    total_missing = int(missing_series.sum())
    print(f"  Total missing values across all cells: {total_missing}")
    if total_missing == 0:
        print("  -> Zero missing cells found. Dataset has complete entries for all 253,680 records.")
    else:
        for col, count in missing_series[missing_series > 0].items():
            print(f"  - {col}: {count} ({count / n_rows * 100:.2f}%)")

    # 4. Duplicate Rows Analysis
    print("\n[4] Duplicate Rows Analysis:")
    feature_cols = [c for c in df.columns if c != "Diabetes_binary"]
    exact_duplicates = int(df.duplicated().sum())
    feature_duplicates = int(df.duplicated(subset=feature_cols).sum())
    unique_profiles = int(len(df.groupby(feature_cols)))

    grouped_targets = df.groupby(feature_cols)["Diabetes_binary"].nunique()
    conflicting_profiles = int((grouped_targets > 1).sum())

    print(f"  Exact full-row duplicates (identical features + identical target): {exact_duplicates:,} ({exact_duplicates / n_rows * 100:.2f}%)")
    print(f"  Feature-vector duplicates (identical survey answers): {feature_duplicates:,} ({feature_duplicates / n_rows * 100:.2f}%)")
    print(f"  Total unique respondent survey profiles: {unique_profiles:,}")
    print(f"  Profiles with conflicting targets (some 0, some 1): {conflicting_profiles:,} ({conflicting_profiles / unique_profiles * 100:.2f}% of unique profiles)")
    print("  * Medical Context: High duplication occurs because all 21 features are discrete survey categories")
    print("    without patient identifiers. Dropping duplicates blindly would distort natural population frequencies.")

    # 5. Target Distribution Analysis
    print("\n[5] Target Variable Distribution (Diabetes_binary):")
    target_counts = df["Diabetes_binary"].value_counts().sort_index()
    target_percentages = df["Diabetes_binary"].value_counts(normalize=True).sort_index() * 100

    for cls_val in target_counts.index:
        cnt = int(target_counts[cls_val])
        pct = float(target_percentages[cls_val])
        label_str = CDC_FEATURE_METADATA["Diabetes_binary"]["values"].get(int(cls_val), "Unknown")
        print(f"  Class {int(cls_val)} ({label_str}): {cnt:,} cases ({pct:.2f}%)")

    imbalance_ratio = float(target_counts[0.0] / target_counts[1.0])
    print(f"  Class Imbalance Ratio (Majority : Minority): {imbalance_ratio:.2f} : 1")
    print(f"  Prevalence of diabetes in cohort: {target_percentages[1.0]:.2f}%")

    # Compare with Multiclass (012)
    print("\n  Comparison with Multiclass Variant (Diabetes_012):")
    target_012_counts = df_012["Diabetes_012"].value_counts().sort_index()
    for cls_val in target_012_counts.index:
        cnt = int(target_012_counts[cls_val])
        pct = float(cnt / len(df_012) * 100)
        label_str = CDC_FEATURE_METADATA["Diabetes_012"]["values"].get(int(cls_val), "Unknown")
        print(f"    Class {int(cls_val)} ({label_str:13s}): {cnt:,} cases ({pct:.2f}%)")

    # Cross-tab between 012 and binary
    ct = pd.crosstab(df_012["Diabetes_012"], df["Diabetes_binary"])
    print(f"\n  Cross-tabulation (Diabetes_012 vs Diabetes_binary):")
    print(ct.to_string())
    print("  * Critical Observation: Diabetes_binary merges Class 0 (No diabetes: 213,703) and")
    print("    Class 1 (Prediabetes: 4,631) into Binary 0.0 (218,334), while Binary 1.0 (35,346) is pure Diabetes.")

    # 6. Descriptive Statistics for Continuous/Discrete Numerical Features
    print("\n[6] Descriptive Statistics for Continuous / Integer Features:")
    num_stats = {}
    for col in CONTINUOUS_NUMERICAL_FEATURES:
        series = df[col]
        stats = {
            "mean": float(series.mean()),
            "std": float(series.std()),
            "min": float(series.min()),
            "q25": float(series.quantile(0.25)),
            "median": float(series.median()),
            "q75": float(series.quantile(0.75)),
            "max": float(series.max()),
            "skew": float(series.skew()),
            "kurtosis": float(series.kurtosis()),
        }
        num_stats[col] = stats
        print(f"  {col:10s} | Mean: {stats['mean']:6.2f} | Std: {stats['std']:6.2f} | "
              f"Min: {stats['min']:4.0f} | 25%: {stats['q25']:4.0f} | Median: {stats['median']:4.0f} | "
              f"75%: {stats['q75']:4.0f} | Max: {stats['max']:4.0f} | Skew: {stats['skew']:+.2f}")

    # 7. Summary of Binary Features Prevalence
    print("\n[7] Prevalence Rates for Binary Indicators (% responding Yes/1):")
    binary_prevalence = {}
    for col in BINARY_FEATURES:
        prev = float(df[col].mean() * 100)
        binary_prevalence[col] = prev
        print(f"  - {col:22s}: {prev:5.2f}%")

    # 8. Summary of Ordinal Features
    print("\n[8] Ordinal Feature Distributions:")
    ordinal_summaries = {}
    for col in ORDINAL_FEATURES:
        dist = (df[col].value_counts(normalize=True).sort_index() * 100).to_dict()
        ordinal_summaries[col] = {int(k): round(float(v), 2) for k, v in dist.items()}
        print(f"  - {col}: {ordinal_summaries[col]}")

    # 9. Save structured JSON report
    report_data = {
        "dataset_name": "CDC Diabetes Health Indicators (BRFSS 2015)",
        "source": "UCI Machine Learning Repository #891",
        "shape": {"rows": n_rows, "columns": n_cols},
        "target_variable": {
            "name": "Diabetes_binary",
            "class_0_count": int(target_counts[0.0]),
            "class_0_pct": round(float(target_percentages[0.0]), 2),
            "class_0_definition": "No diabetes (includes prediabetes)",
            "class_1_count": int(target_counts[1.0]),
            "class_1_pct": round(float(target_percentages[1.0]), 2),
            "class_1_definition": "Diagnosed diabetes",
            "imbalance_ratio": round(imbalance_ratio, 2),
        },
        "quality_metrics": {
            "total_missing_values": total_missing,
            "exact_duplicate_rows": exact_duplicates,
            "exact_duplicate_pct": round(exact_duplicates / n_rows * 100, 2),
            "feature_duplicates": feature_duplicates,
            "unique_profiles": unique_profiles,
            "conflicting_profiles": conflicting_profiles,
        },
        "numerical_statistics": num_stats,
        "binary_prevalence": binary_prevalence,
        "ordinal_distributions": ordinal_summaries,
    }

    report_path = PROJECT_ROOT / "reports" / "cdc_diabetes_data_understanding_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=2)
    print(f"\n[OK] Data understanding report persisted to: {report_path.relative_to(PROJECT_ROOT)}")
    print("=" * 80)


if __name__ == "__main__":
    run_data_understanding()
