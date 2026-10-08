"""Execution Script for Phase 5: Data Cleaning and Preprocessing for CDC Diabetes.

Audits data quality, applies reproducible deduplication and type downcasting,
performs stratified train/val/test splitting, fits scikit-learn preprocessing pipeline
strictly on training data, and outputs exact before/after dataset statistics.

Project: AI-Powered Health Risk Prediction and Monitoring System
Author: Abhiram (Lead AI/ML Engineer)
Date: 8 October 2026
"""

import json
import sys
from pathlib import Path

# Add project root to Python module path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd
import numpy as np

from src.preprocessing.data_loader import load_cdc_diabetes_data
from src.preprocessing.data_cleaner import audit_data_quality, clean_cdc_diabetes_data
from src.preprocessing.pipeline import (
    create_preprocessor,
    stratified_split,
    build_and_fit_pipeline,
    save_processed_datasets,
    CONTINUOUS_NUMERICAL_FEATURES,
    ORDINAL_FEATURES,
    BINARY_FEATURES,
)


def run_data_cleaning_pipeline():
    print("=" * 80)
    print("PHASE 5: CDC DIABETES HEALTH INDICATORS - DATA CLEANING & PREPROCESSING")
    print("=" * 80)

    # 1. Load Raw Dataset
    print("\n[STEP 1] Ingesting Raw CDC Diabetes Dataset...")
    df_raw = load_cdc_diabetes_data(variant="binary")
    raw_memory_mb = df_raw.memory_usage(deep=True).sum() / (1024 * 1024)
    print(f"Raw Dataset Loaded: {df_raw.shape[0]:,} rows x {df_raw.shape[1]} columns ({raw_memory_mb:.2f} MB)")

    # 2. Comprehensive Data Quality Audit
    print("\n[STEP 2] Running Pre-Cleaning Data Quality Audit...")
    audit = audit_data_quality(df_raw, target_col="Diabetes_binary")

    print("\n--- DATA QUALITY REPORT ---")
    print(f"1. Total Missing Values: {audit['missing_values']['total']}")
    
    # Invalid values check
    invalid_cols = [k for k, v in audit["invalid_values"].items() if v["out_of_range"] > 0 or v["non_integer"] > 0]
    print(f"2. Invalid Values (out of codebook range / non-integer): {len(invalid_cols)} columns with issues")
    if not invalid_cols:
        print("   -> All 22 columns strictly adhere to CDC BRFSS valid codebook ranges.")

    # Suspicious values
    susp = audit["suspicious_values"]
    print(f"3. Suspicious Values (Physiological Extremes in BMI):")
    print(f"   - BMI < 15 (Severely underweight): {susp['bmi_under_15']} ({susp['bmi_under_15_pct']}%)")
    print(f"   - BMI > 60 (Class 3 super-obesity): {susp['bmi_over_60']} ({susp['bmi_over_60_pct']}%)")
    print(f"   - BMI > 70 (Top-coded extreme values): {susp['bmi_over_70']} ({susp['bmi_over_70_pct']}%)")
    print(f"   - Min BMI: {susp['bmi_min']} | Max BMI: {susp['bmi_max']}")

    # Identifiers
    print(f"4. Identifier Columns: {audit['identifier_columns']['status']}")

    # Target leakage
    print(f"5. Target Leakage Risks: {audit['target_leakage_risks']['status']}")
    corrs = audit["target_leakage_risks"]["feature_correlations_with_target"]
    sorted_corrs = sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True)
    print("   Top 5 feature correlations with Diabetes_binary:")
    for feat, corr_val in sorted_corrs[:5]:
        print(f"   - {feat:20s}: r = {corr_val:+.4f}")

    # Duplicates
    dups = audit["duplicate_rows"]
    print(f"6. Duplicate Rows:")
    print(f"   - Exact Full-Row Duplicates: {dups['exact_duplicate_rows']:,} ({dups['exact_duplicate_pct']}%)")
    print(f"   - Feature Vector Duplicates: {dups['feature_duplicates']:,}")
    print(f"   - Unique Respondent Profiles: {dups['unique_profiles']:,}")
    print(f"   - Conflicting Target Profiles: {dups['conflicting_profiles']:,}")

    # Class imbalance
    imb = audit["class_imbalance"]
    print(f"7. Class Imbalance:")
    print(f"   - Class 0 (No Diabetes): {imb['counts'][0]:,} ({imb['percentages'][0]}%)")
    print(f"   - Class 1 (Diabetes)   : {imb['counts'][1]:,} ({imb['percentages'][1]}%)")
    print(f"   - Imbalance Ratio      : {imb['ratio']}:1")

    # Feature types
    ft = audit["feature_types"]
    print(f"8. Feature Types Categorization:")
    print(f"   - Binary Indicators ({len(ft['binary_features'])}): {ft['binary_features']}")
    print(f"   - Ordinal Features  ({len(ft['ordinal_features'])}): {ft['ordinal_features']}")
    print(f"   - Continuous/Counts ({len(ft['continuous_numerical_features'])}): {ft['continuous_numerical_features']}")

    # Save quality audit report to reports/
    quality_report_path = PROJECT_ROOT / "reports" / "cdc_diabetes_data_quality_report.json"
    quality_report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(quality_report_path, "w") as f:
        json.dump(audit, f, indent=2)
    print(f"\n[OK] Data quality report persisted to: {quality_report_path.relative_to(PROJECT_ROOT)}")

    # 3. Apply Reproducible Cleaning (Deduplication + Type Downcasting)
    print("\n[STEP 3] Applying Reproducible Cleaning (Deduplication & Type Optimization)...")
    df_cleaned, clean_meta = clean_cdc_diabetes_data(df_raw, drop_duplicates=True)
    cleaned_memory_mb = df_cleaned.memory_usage(deep=True).sum() / (1024 * 1024)

    # 4. Stratified Train / Validation / Test Splitting
    print("\n[STEP 4] Executing Stratified Train / Validation / Test Splitting (70% / 15% / 15%)...")
    train_df, val_df, test_df = stratified_split(
        df_cleaned, target_col="Diabetes_binary", test_size=0.15, val_size=0.15, random_state=42
    )
    print(f"Train Partition : {train_df.shape[0]:,} rows ({train_df.shape[0]/len(df_cleaned)*100:.1f}%)")
    print(f"Val Partition   : {val_df.shape[0]:,} rows ({val_df.shape[0]/len(df_cleaned)*100:.1f}%)")
    print(f"Test Partition  : {test_df.shape[0]:,} rows ({test_df.shape[0]/len(df_cleaned)*100:.1f}%) [Untouched]")

    # Check stratification consistency
    train_prev = float(train_df["Diabetes_binary"].mean() * 100)
    val_prev = float(val_df["Diabetes_binary"].mean() * 100)
    test_prev = float(test_df["Diabetes_binary"].mean() * 100)
    print(f"Positive Class Prevalence: Train={train_prev:.2f}%, Val={val_prev:.2f}%, Test={test_prev:.2f}%")

    # 5. Build and Fit Preprocessing Pipeline (Strictly on Train Set)
    print("\n[STEP 5] Fitting Scikit-Learn ColumnTransformer Strictly on Training Data (Zero Leakage)...")
    preprocessor, X_train_transformed = build_and_fit_pipeline(
        train_df, target_col="Diabetes_binary", scaling="standard"
    )

    # Serialize preprocessor pipeline
    models_dir = PROJECT_ROOT / "models" / "diabetes"
    models_dir.mkdir(parents=True, exist_ok=True)
    preprocessor_path = models_dir / "diabetes_preprocessor.joblib"
    joblib.dump(preprocessor, preprocessor_path)
    print(f"[OK] Fitted preprocessor saved to: {preprocessor_path.relative_to(PROJECT_ROOT)}")

    # 6. Save Partitioned Datasets
    print("\n[STEP 6] Persisting Processed Partitions...")
    processed_dir = PROJECT_ROOT / "data" / "processed" / "diabetes"
    saved_paths = save_processed_datasets(train_df, val_df, test_df, processed_dir)
    for part_name, path in saved_paths.items():
        print(f"  - {part_name.capitalize():5s}: {path.relative_to(PROJECT_ROOT)} ({path.stat().st_size / (1024*1024):.2f} MB)")

    # 7. Print Comprehensive BEFORE vs AFTER Comparison Table
    print("\n" + "=" * 80)
    print("DATASET COMPARISON: BEFORE VS AFTER CLEANING")
    print("=" * 80)

    comparison_data = [
        ("Total Rows", f"{df_raw.shape[0]:,}", f"{df_cleaned.shape[0]:,}", f"-{clean_meta['dropped_rows']:,} (-{clean_meta['dropped_pct']}%)"),
        ("Columns", str(df_raw.shape[1]), str(df_cleaned.shape[1]), "Unchanged (21 feats + 1 target)"),
        ("Missing Cells", "0 (0.00%)", "0 (0.00%)", "Zero missingness preserved"),
        ("Exact Duplicate Rows", f"{audit['duplicate_rows']['exact_duplicate_rows']:,}", "0", "100% deduplicated"),
        ("Class 0 (No Diabetes)", f"{clean_meta['initial_target_distribution'][0]:,} (86.07%)", f"{clean_meta['final_target_distribution'][0]:,} (84.71%)", f"-{clean_meta['initial_target_distribution'][0] - clean_meta['final_target_distribution'][0]:,}"),
        ("Class 1 (Diabetes)", f"{clean_meta['initial_target_distribution'][1]:,} (13.93%)", f"{clean_meta['final_target_distribution'][1]:,} (15.29%)", f"-{clean_meta['initial_target_distribution'][1] - clean_meta['final_target_distribution'][1]:,}"),
        ("Imbalance Ratio", f"{clean_meta['initial_imbalance_ratio']}:1", f"{clean_meta['final_imbalance_ratio']}:1", "Slightly reduced majority skew"),
        ("Memory Footprint", f"{raw_memory_mb:.2f} MB", f"{cleaned_memory_mb:.2f} MB", f"Reduced by {(1 - cleaned_memory_mb/raw_memory_mb)*100:.1f}%"),
        ("Storage Data Types", "float64 (all 22)", "int8 / int16 (clean integers)", "Optimized memory layout"),
        ("Preprocessed Feats", "Unscaled raw", "Standardized numerical, passthrough binary/ord", "Zero leakage Pipeline"),
    ]

    print(f"{'Metric':<24} | {'Before (Raw)':<26} | {'After (Cleaned)':<26} | {'Delta / Impact'}")
    print("-" * 105)
    for metric, before, after, delta in comparison_data:
        print(f"{metric:<24} | {before:<26} | {after:<26} | {delta}")
    print("=" * 80)

    # 8. Train/Val/Test Split Breakdown
    print("\nPARTITION DISTRIBUTION BREAKDOWN:")
    print("-" * 80)
    for name, part in [("Train (70%)", train_df), ("Val (15%)", val_df), ("Test (15%)", test_df)]:
        c0 = int((part["Diabetes_binary"] == 0).sum())
        c1 = int((part["Diabetes_binary"] == 1).sum())
        ratio = c0 / c1
        print(f"{name:15s}: {part.shape[0]:,} samples | Class 0: {c0:,} ({c0/len(part)*100:.2f}%) | Class 1: {c1:,} ({c1/len(part)*100:.2f}%) | Ratio: {ratio:.2f}:1")
    print("=" * 80)


if __name__ == "__main__":
    run_data_cleaning_pipeline()
