"""Exploratory data analysis for the UCI Heart Disease dataset.

Uses the actual processed Heart Disease files already loaded by the Heart
preprocessing module. This script does not train predictive models and does not
change the target definition (`num`, values 0-4).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from scipy.stats import chi2_contingency  # noqa: E402

from src.preprocessing.heart_disease import (  # noqa: E402
    CATEGORICAL_FEATURES,
    HEART_DISEASE_COLUMNS,
    HEART_DISEASE_TARGET,
    NUMERIC_FEATURES,
    SOURCE_COLUMN,
    audit_heart_disease_data,
    clean_heart_disease_data,
    load_heart_disease_data,
)


FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["figure.dpi"] = 300


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


def _percentage_table(series: pd.Series) -> dict[str, dict[str, float]]:
    counts = series.value_counts(dropna=False)
    percentages = series.value_counts(normalize=True, dropna=False) * 100
    ordered_index = sorted(counts.index, key=lambda value: str(value))
    return {
        "counts": {str(k): int(counts.loc[k]) for k in ordered_index},
        "percentages": {str(k): round(float(percentages.loc[k]), 2) for k in ordered_index},
    }


def _cramers_v(feature: pd.Series, target: pd.Series) -> float:
    data = pd.DataFrame({"feature": feature.astype("object"), "target": target})
    data["feature"] = data["feature"].where(data["feature"].notna(), "Missing")
    data["feature"] = data["feature"].map(str)
    contingency = pd.crosstab(data["feature"], data["target"])
    if contingency.empty or min(contingency.shape) < 2:
        return 0.0

    chi2, _, _, _ = chi2_contingency(contingency)
    n = contingency.to_numpy().sum()
    phi2 = chi2 / n
    r, k = contingency.shape

    # Bias-corrected Cramer's V.
    phi2corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    denominator = min(kcorr - 1, rcorr - 1)
    if denominator <= 0:
        return 0.0
    return float(np.sqrt(phi2corr / denominator))


def _iqr_outliers(series: pd.Series) -> dict[str, Any]:
    clean = series.dropna()
    q1 = clean.quantile(0.25)
    q3 = clean.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    mask = (clean < lower) | (clean > upper)
    return {
        "q1": round(float(q1), 3),
        "q3": round(float(q3), 3),
        "iqr": round(float(iqr), 3),
        "lower_fence": round(float(lower), 3),
        "upper_fence": round(float(upper), 3),
        "outlier_count": int(mask.sum()),
        "outlier_pct_non_missing": round(float(mask.mean() * 100), 2) if len(clean) else 0.0,
        "min": round(float(clean.min()), 3) if len(clean) else None,
        "max": round(float(clean.max()), 3) if len(clean) else None,
    }


def plot_target_and_source(cleaned_df: pd.DataFrame, raw_df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    target_counts = cleaned_df[HEART_DISEASE_TARGET].value_counts().sort_index()
    target_pcts = cleaned_df[HEART_DISEASE_TARGET].value_counts(normalize=True).sort_index() * 100
    bars = axes[0].bar(
        [str(int(x)) for x in target_counts.index],
        target_counts.values,
        color=["#4c78a8", "#f58518", "#e45756", "#72b7b2", "#54a24b"],
        edgecolor="black",
    )
    for bar, count, pct in zip(bars, target_counts.values, target_pcts.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, count + 5, f"{count}\n{pct:.1f}%", ha="center", fontsize=9)
    axes[0].set_title("Target `num` Distribution After Cleaning")
    axes[0].set_xlabel("num")
    axes[0].set_ylabel("Rows")

    source_counts = raw_df[SOURCE_COLUMN].value_counts().sort_index()
    axes[1].bar(source_counts.index, source_counts.values, color="#7f7f7f", edgecolor="black")
    for idx, count in enumerate(source_counts.values):
        axes[1].text(idx, count + 5, str(count), ha="center", fontsize=9)
    axes[1].set_title("Raw Rows By Source File")
    axes[1].set_xlabel("source")
    axes[1].set_ylabel("Rows")

    plt.tight_layout()
    path = FIGURES_DIR / "10_heart_target_and_source_distribution.png"
    plt.savefig(path, dpi=300)
    plt.close()
    return path


def plot_numeric_distributions(cleaned_df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for ax, column in zip(axes, NUMERIC_FEATURES):
        sns.histplot(cleaned_df[column], kde=True, ax=ax, color="#4c78a8", edgecolor="black")
        ax.set_title(f"{column} distribution")
        ax.set_xlabel(column)
        ax.set_ylabel("Rows")

    fig.delaxes(axes[-1])
    plt.tight_layout()
    path = FIGURES_DIR / "11_heart_numeric_distributions.png"
    plt.savefig(path, dpi=300)
    plt.close()
    return path


def plot_categorical_distributions(cleaned_df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    axes = axes.flatten()

    for ax, column in zip(axes, CATEGORICAL_FEATURES):
        values = cleaned_df[column].astype("object").where(cleaned_df[column].notna(), "Missing")
        values = values.map(str)
        counts = values.value_counts()
        ordered_index = sorted(counts.index, key=lambda value: str(value))
        ax.bar([str(x) for x in ordered_index], [counts.loc[x] for x in ordered_index], color="#72b7b2", edgecolor="black")
        ax.set_title(f"{column} distribution")
        ax.set_xlabel(column)
        ax.set_ylabel("Rows")

    for idx in range(len(CATEGORICAL_FEATURES), len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    path = FIGURES_DIR / "12_heart_categorical_distributions.png"
    plt.savefig(path, dpi=300)
    plt.close()
    return path


def plot_missing_patterns(raw_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    raw_missing_pct = raw_df.isna().mean().sort_values(ascending=False) * 100
    raw_missing_pct = raw_missing_pct[raw_missing_pct > 0]
    axes[0].bar(raw_missing_pct.index, raw_missing_pct.values, color="#e45756", edgecolor="black")
    axes[0].set_title("Raw Missingness By Column")
    axes[0].set_ylabel("Missing rows (%)")
    axes[0].tick_params(axis="x", rotation=45)

    missing_by_source = raw_df.groupby(SOURCE_COLUMN).apply(lambda part: part.isna().sum(), include_groups=False)
    missing_by_source = missing_by_source[[c for c in raw_df.columns if c != SOURCE_COLUMN]]
    sns.heatmap(missing_by_source, cmap="Reds", annot=True, fmt=".0f", linewidths=0.5, ax=axes[1])
    axes[1].set_title("Raw Missing Counts By Source")
    axes[1].set_xlabel("Column")
    axes[1].set_ylabel("source")

    plt.tight_layout()
    path = FIGURES_DIR / "13_heart_missingness_patterns.png"
    plt.savefig(path, dpi=300)
    plt.close()
    return path


def plot_feature_target_relationships(cleaned_df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    numeric_medians = cleaned_df.groupby(HEART_DISEASE_TARGET)[NUMERIC_FEATURES].median()
    for column in ["age", "thalach", "oldpeak"]:
        axes[0, 0].plot(numeric_medians.index, numeric_medians[column], marker="o", label=column)
    axes[0, 0].set_title("Median Numerical Features By `num`")
    axes[0, 0].set_xlabel("num")
    axes[0, 0].set_ylabel("Median value")
    axes[0, 0].legend()

    cp_table = pd.crosstab(cleaned_df["cp"], cleaned_df[HEART_DISEASE_TARGET], normalize="index") * 100
    cp_table.plot(kind="bar", stacked=True, ax=axes[0, 1], colormap="viridis", edgecolor="black")
    axes[0, 1].set_title("Target Mix By Chest Pain Type (`cp`)")
    axes[0, 1].set_xlabel("cp")
    axes[0, 1].set_ylabel("Row percentage")
    axes[0, 1].legend(title="num", fontsize=8)

    exang_values = cleaned_df["exang"].astype("object").where(cleaned_df["exang"].notna(), "Missing").map(str)
    exang_table = pd.crosstab(exang_values, cleaned_df[HEART_DISEASE_TARGET], normalize="index") * 100
    exang_table.plot(kind="bar", stacked=True, ax=axes[1, 0], colormap="viridis", edgecolor="black")
    axes[1, 0].set_title("Target Mix By Exercise-Induced Angina (`exang`)")
    axes[1, 0].set_xlabel("exang")
    axes[1, 0].set_ylabel("Row percentage")
    axes[1, 0].legend(title="num", fontsize=8)

    thal_values = cleaned_df["thal"].astype("object").where(cleaned_df["thal"].notna(), "Missing").map(str)
    thal_table = pd.crosstab(thal_values, cleaned_df[HEART_DISEASE_TARGET], normalize="index") * 100
    thal_table.plot(kind="bar", stacked=True, ax=axes[1, 1], colormap="viridis", edgecolor="black")
    axes[1, 1].set_title("Target Mix By `thal`")
    axes[1, 1].set_xlabel("thal")
    axes[1, 1].set_ylabel("Row percentage")
    axes[1, 1].legend(title="num", fontsize=8)

    plt.tight_layout()
    path = FIGURES_DIR / "14_heart_feature_target_relationships.png"
    plt.savefig(path, dpi=300)
    plt.close()
    return path


def plot_correlation_heatmap(cleaned_df: pd.DataFrame) -> Path:
    corr_cols = [HEART_DISEASE_TARGET] + NUMERIC_FEATURES
    corr = cleaned_df[corr_cols].corr(method="spearman")

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, linewidths=0.5)
    plt.title("Spearman Correlation: `num` And Numerical Features")
    plt.tight_layout()
    path = FIGURES_DIR / "15_heart_numeric_spearman_correlation.png"
    plt.savefig(path, dpi=300)
    plt.close()
    return path


def write_markdown_report(path: Path, report: dict[str, Any]) -> None:
    raw = report["raw_dataset"]
    cleaned = report["cleaned_dataset"]
    findings = report["key_findings"]

    lines = [
        "# Heart Disease EDA Report",
        "",
        "This report uses the existing Heart Disease target column `num` with values 0-4. No binary target was created.",
        "",
        "## Dataset",
        "",
        f"- Raw shape: {raw['shape'][0]} rows x {raw['shape'][1]} columns.",
        f"- Cleaned EDA shape: {cleaned['shape'][0]} rows x {cleaned['shape'][1]} columns.",
        f"- Raw target distribution: {raw['target_distribution']['counts']}.",
        f"- Cleaned target distribution: {cleaned['target_distribution']['counts']}.",
        f"- Raw missing cells: {raw['missing_values']['total']}.",
        f"- Cleaned missing cells: {cleaned['missing_values']['total']}.",
        "",
        "## Key Findings",
        "",
    ]
    lines.extend(f"- {finding}" for finding in findings)
    lines.extend(
        [
            "",
            "## Generated Figures",
            "",
        ]
    )
    lines.extend(f"- `{figure}`" for figure in report["figures_generated"])
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_eda() -> None:
    print("=" * 80)
    print("HEART DISEASE EXPLORATORY DATA ANALYSIS")
    print("=" * 80)

    raw_df = load_heart_disease_data(source="all", include_source=True)
    raw_audit = audit_heart_disease_data(raw_df)
    cleaned_df, cleaning_metadata = clean_heart_disease_data(
        raw_df,
        drop_duplicates=True,
        zero_measurements_as_missing=True,
    )

    target_distribution_raw = _percentage_table(raw_df[HEART_DISEASE_TARGET])
    target_distribution_cleaned = _percentage_table(cleaned_df[HEART_DISEASE_TARGET])
    source_distribution = _percentage_table(raw_df[SOURCE_COLUMN])

    numeric_summary = cleaned_df[NUMERIC_FEATURES].describe().round(3).to_dict()
    numeric_by_target = cleaned_df.groupby(HEART_DISEASE_TARGET)[NUMERIC_FEATURES].agg(["count", "median", "mean"]).round(3)
    numeric_correlations = cleaned_df[[HEART_DISEASE_TARGET] + NUMERIC_FEATURES].corr(method="spearman")[HEART_DISEASE_TARGET]
    numeric_correlations = numeric_correlations.drop(index=HEART_DISEASE_TARGET).sort_values(key=lambda s: s.abs(), ascending=False)

    categorical_distributions = {
        column: _percentage_table(
            cleaned_df[column].astype("object").where(cleaned_df[column].notna(), "Missing").map(str)
        )
        for column in CATEGORICAL_FEATURES
    }
    categorical_associations = {
        column: round(_cramers_v(cleaned_df[column], cleaned_df[HEART_DISEASE_TARGET]), 4)
        for column in CATEGORICAL_FEATURES
    }
    categorical_associations = dict(sorted(categorical_associations.items(), key=lambda item: item[1], reverse=True))

    missing_by_source = raw_df.groupby(SOURCE_COLUMN).apply(lambda part: part.isna().sum(), include_groups=False)
    missing_by_target = raw_df.groupby(HEART_DISEASE_TARGET).apply(lambda part: part.isna().sum(), include_groups=False)

    outliers = {column: _iqr_outliers(cleaned_df[column]) for column in NUMERIC_FEATURES}
    anomalies = {
        "raw_trestbps_zero_count": int((raw_df["trestbps"] == 0).sum()),
        "raw_chol_zero_count": int((raw_df["chol"] == 0).sum()),
        "cleaned_oldpeak_negative_count": int((cleaned_df["oldpeak"] < 0).sum()),
        "cleaned_chol_above_500_count": int((cleaned_df["chol"] > 500).sum()),
        "cleaned_trestbps_above_180_count": int((cleaned_df["trestbps"] > 180).sum()),
    }

    figure_paths = [
        plot_target_and_source(cleaned_df, raw_df),
        plot_numeric_distributions(cleaned_df),
        plot_categorical_distributions(cleaned_df),
        plot_missing_patterns(raw_df, cleaned_df),
        plot_feature_target_relationships(cleaned_df),
        plot_correlation_heatmap(cleaned_df),
    ]

    top_numeric = numeric_correlations.head(3)
    top_categorical = list(categorical_associations.items())[:4]
    missing_sorted = raw_df.isna().sum().sort_values(ascending=False)
    outlier_sorted = sorted(outliers.items(), key=lambda item: item[1]["outlier_count"], reverse=True)

    key_findings = [
        (
            f"The cleaned EDA dataset has {cleaned_df.shape[0]} rows after dropping "
            f"{cleaning_metadata['dropped_duplicate_rows']} exact duplicate rows; target `num` values remain "
            f"{sorted(cleaned_df[HEART_DISEASE_TARGET].unique().tolist())}."
        ),
        (
            f"Class `num=0` is the largest cleaned class with "
            f"{target_distribution_cleaned['counts']['0']} rows "
            f"({target_distribution_cleaned['percentages']['0']}%); class `num=4` is the smallest with "
            f"{target_distribution_cleaned['counts']['4']} rows "
            f"({target_distribution_cleaned['percentages']['4']}%)."
        ),
        (
            f"Raw missingness is concentrated in {missing_sorted.index[0]} "
            f"({int(missing_sorted.iloc[0])} rows), {missing_sorted.index[1]} "
            f"({int(missing_sorted.iloc[1])} rows), and {missing_sorted.index[2]} "
            f"({int(missing_sorted.iloc[2])} rows)."
        ),
        (
            f"The strongest absolute Spearman correlations with `num` among numerical features are "
            + ", ".join(f"{feature}={value:.3f}" for feature, value in top_numeric.items())
            + "."
        ),
        (
            "The largest categorical associations with `num` by bias-corrected Cramer's V are "
            + ", ".join(f"{feature}={value:.3f}" for feature, value in top_categorical)
            + "."
        ),
        (
            f"IQR outlier counts are highest for {outlier_sorted[0][0]} "
            f"({outlier_sorted[0][1]['outlier_count']} rows), followed by {outlier_sorted[1][0]} "
            f"({outlier_sorted[1][1]['outlier_count']} rows)."
        ),
        (
            f"Observed anomaly counts include raw trestbps=0 in {anomalies['raw_trestbps_zero_count']} row, "
            f"raw chol=0 in {anomalies['raw_chol_zero_count']} rows, and cleaned oldpeak<0 in "
            f"{anomalies['cleaned_oldpeak_negative_count']} rows."
        ),
    ]

    report = {
        "target_definition": "Existing multiclass `num` target, values 0-4; no binary target created.",
        "raw_dataset": {
            "shape": [int(raw_df.shape[0]), int(raw_df.shape[1])],
            "columns": list(raw_df.columns),
            "source_distribution": source_distribution,
            "target_distribution": target_distribution_raw,
            "missing_values": raw_audit["missing_values"],
            "duplicate_rows": raw_audit["duplicate_rows"],
        },
        "cleaned_dataset": {
            "shape": [int(cleaned_df.shape[0]), int(cleaned_df.shape[1])],
            "target_distribution": target_distribution_cleaned,
            "missing_values": cleaning_metadata["missing_values_after_cleaning"],
            "cleaning_metadata": cleaning_metadata,
        },
        "numerical_feature_summary": numeric_summary,
        "numerical_by_target": numeric_by_target.to_dict(),
        "numerical_spearman_correlation_with_target": {
            column: round(float(value), 4) for column, value in numeric_correlations.items()
        },
        "categorical_feature_distributions": categorical_distributions,
        "categorical_cramers_v_with_target": categorical_associations,
        "missing_by_source": missing_by_source.to_dict(),
        "missing_by_target": missing_by_target.to_dict(),
        "outliers_iqr": outliers,
        "potential_anomalies": anomalies,
        "key_findings": key_findings,
        "figures_generated": [str(path.relative_to(PROJECT_ROOT)) for path in figure_paths],
    }

    json_path = PROJECT_ROOT / "reports" / "heart_disease_eda_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(report), f, indent=2)

    markdown_path = PROJECT_ROOT / "reports" / "heart_disease_eda_report.md"
    write_markdown_report(markdown_path, _json_safe(report))

    print(f"Raw shape: {raw_df.shape}")
    print(f"Cleaned EDA shape: {cleaned_df.shape}")
    print(f"Target distribution (cleaned): {target_distribution_cleaned['counts']}")
    print(f"Missing values (raw total): {raw_audit['missing_values']['total']}")
    print(f"Missing values (cleaned total): {cleaning_metadata['missing_values_after_cleaning']['total']}")
    print("Top numerical Spearman correlations with `num`:")
    for feature, value in numeric_correlations.items():
        print(f"  {feature}: {value:.4f}")
    print("Top categorical Cramer's V associations with `num`:")
    for feature, value in categorical_associations.items():
        print(f"  {feature}: {value:.4f}")
    print("Potential anomalies:")
    for key, value in anomalies.items():
        print(f"  {key}: {value}")
    print("Generated figures:")
    for path in figure_paths:
        print(f"  {path.relative_to(PROJECT_ROOT)}")
    print(f"[OK] EDA JSON report saved to: {json_path.relative_to(PROJECT_ROOT)}")
    print(f"[OK] EDA Markdown report saved to: {markdown_path.relative_to(PROJECT_ROOT)}")
    print("No predictive model was trained.")
    print("=" * 80)


if __name__ == "__main__":
    run_eda()
