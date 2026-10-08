"""Exploratory Data Analysis (EDA) Script for CDC Diabetes Health Indicators.

Performs rigorous, question-driven clinical exploratory data analysis strictly on the
training partition (to prevent test snooping / leakage). Generates publication-ready
visualizations saved to reports/figures/ and quantitative summaries saved to reports/.

Project: AI-Powered Health Risk Prediction and Monitoring System
Author: Abhiram (Lead AI/ML Engineer)
Date: 8 October 2026
"""

import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from src.preprocessing.data_loader import load_cdc_diabetes_data
from src.features.cdc_diabetes_metadata import (
    CDC_FEATURE_METADATA,
    BINARY_FEATURES,
    ORDINAL_FEATURES,
    CONTINUOUS_NUMERICAL_FEATURES,
)

# Styling configuration
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["figure.dpi"] = 300

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def plot_target_distribution_and_imbalance(train_df: pd.DataFrame, df_012: pd.DataFrame):
    """Figure 1: Target distribution, class imbalance, and multiclass mapping."""
    print("  Generating Figure 1: Target distribution and class imbalance...")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Subplot A: Binary Target in Train Set
    counts = train_df["Diabetes_binary"].value_counts().sort_index()
    pcts = (train_df["Diabetes_binary"].value_counts(normalize=True).sort_index() * 100)
    labels = ["0: No Diabetes\n(incl. Prediabetes)", "1: Diagnosed\nDiabetes"]
    colors = ["#2b5c8f", "#d95f02"]

    bars = axes[0].bar(labels, counts.values, color=colors, width=0.55, edgecolor="black", linewidth=1.2)
    for bar, count, pct in zip(bars, counts.values, pcts.values):
        yval = bar.get_height()
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            yval + 3000,
            f"{count:,}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    imbalance_ratio = counts[0] / counts[1]
    axes[0].set_title(f"A. Binary Target Distribution (Train Cohort: N={len(train_df):,})\nImbalance Ratio: {imbalance_ratio:.2f} : 1 (Prevalence: {pcts[1]:.2f}%)", pad=12)
    axes[0].set_ylabel("Number of Respondents")
    axes[0].set_ylim(0, max(counts.values) * 1.18)

    # Subplot B: Multiclass 3-Class Variant (012)
    counts_012 = df_012["Diabetes_012"].value_counts().sort_index()
    pcts_012 = (df_012["Diabetes_012"].value_counts(normalize=True).sort_index() * 100)
    labels_012 = ["0: Non-Diabetic", "1: Prediabetic", "2: Diabetic"]
    colors_012 = ["#2b5c8f", "#7570b3", "#d95f02"]

    bars_012 = axes[1].bar(labels_012, counts_012.values, color=colors_012, width=0.55, edgecolor="black", linewidth=1.2)
    for bar, count, pct in zip(bars_012, counts_012.values, pcts_012.values):
        yval = bar.get_height()
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            yval + 3000,
            f"{count:,}\n({pct:.2f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    axes[1].set_title("B. Original 3-Class BRFSS Target Encoding (N=253,680)\nPrediabetes (1.83%) Grouped with Class 0 in Binary Target", pad=12)
    axes[1].set_ylabel("Number of Respondents")
    axes[1].set_ylim(0, max(counts_012.values) * 1.18)

    plt.tight_layout()
    fig_path = FIGURES_DIR / "01_target_distribution_and_imbalance.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"  -> Saved: {fig_path.name}")


def plot_numerical_distributions_and_outliers(train_df: pd.DataFrame):
    """Figure 2: Distribution of continuous/count features, WHO BMI cutoffs, and outliers."""
    print("  Generating Figure 2: Numerical distributions and outliers...")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Subplot A: BMI Distribution with WHO clinical cutoffs
    bmi = train_df["BMI"]
    axes[0].hist(bmi, bins=45, range=(10, 80), color="#1f77b4", edgecolor="black", alpha=0.75, density=True)
    sns.kdeplot(bmi, ax=axes[0], color="#d62728", linewidth=2, clip=(10, 80))

    axes[0].axvline(18.5, color="gray", linestyle="--", linewidth=1.2, label="Underweight (<18.5)")
    axes[0].axvline(25.0, color="orange", linestyle="--", linewidth=1.2, label="Overweight (>=25)")
    axes[0].axvline(30.0, color="red", linestyle="--", linewidth=1.5, label="Obese (>=30)")

    axes[0].set_title(f"A. BMI Distribution (Median={bmi.median():.0f}, Mean={bmi.mean():.1f})\nRight-skewed with Class 3 Super-Obesity Outliers", pad=12)
    axes[0].set_xlabel("Body Mass Index (kg/m²)")
    axes[0].set_ylabel("Density")
    axes[0].legend(loc="upper right", fontsize=8.5)

    # Subplot B: BMI Boxplot stratified by Diabetes status
    data_bmi_0 = train_df[train_df["Diabetes_binary"] == 0]["BMI"]
    data_bmi_1 = train_df[train_df["Diabetes_binary"] == 1]["BMI"]

    bplot1 = axes[1].boxplot(
        [data_bmi_0, data_bmi_1],
        tick_labels=["No Diabetes", "Diabetes"],
        patch_artist=True,
        showmeans=True,
        meanline=True,
        medianprops={"color": "black", "linewidth": 2},
        meanprops={"color": "gold", "linewidth": 2, "linestyle": "--"},
        flierprops={"marker": "o", "markersize": 2, "alpha": 0.25, "color": "gray"},
    )
    bplot1["boxes"][0].set_facecolor("#a6bddb")
    bplot1["boxes"][1].set_facecolor("#fc9272")

    axes[1].set_title(f"B. BMI Stratified by Diabetes Status\nMedian: {data_bmi_0.median():.0f} (Non-Diabetic) vs {data_bmi_1.median():.0f} (Diabetic)", pad=12)
    axes[1].set_ylabel("Body Mass Index (kg/m²)")
    axes[1].set_ylim(10, 75)

    # Subplot C: Days of Poor Physical Health (PhysHlth) Stratified
    data_phys_0 = train_df[train_df["Diabetes_binary"] == 0]["PhysHlth"]
    data_phys_1 = train_df[train_df["Diabetes_binary"] == 1]["PhysHlth"]

    bplot2 = axes[2].boxplot(
        [data_phys_0, data_phys_1],
        tick_labels=["No Diabetes", "Diabetes"],
        patch_artist=True,
        showmeans=True,
        meanline=True,
        medianprops={"color": "black", "linewidth": 2},
        meanprops={"color": "gold", "linewidth": 2, "linestyle": "--"},
        flierprops={"marker": "o", "markersize": 2, "alpha": 0.25, "color": "gray"},
    )
    bplot2["boxes"][0].set_facecolor("#a6bddb")
    bplot2["boxes"][1].set_facecolor("#fc9272")

    axes[2].set_title(f"C. Physical Illness Days/Month (PhysHlth)\nMean: {data_phys_0.mean():.1f} (Non-Diabetic) vs {data_phys_1.mean():.1f} (Diabetic)", pad=12)
    axes[2].set_ylabel("Days Unhealthy in Past 30 Days")
    axes[2].set_ylim(-1, 32)

    plt.tight_layout()
    fig_path = FIGURES_DIR / "02_numerical_distributions_and_outliers.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"  -> Saved: {fig_path.name}")


def plot_binary_risk_prevalence(train_df: pd.DataFrame):
    """Figure 3: Diabetes prevalence across key binary indicators."""
    print("  Generating Figure 3: Binary clinical and lifestyle risk prevalence...")
    fig, ax = plt.subplots(figsize=(14, 6.5))

    indicators = [
        "HighBP", "HighChol", "HeartDiseaseorAttack", "Stroke",
        "DiffWalk", "Smoker", "Sex", "PhysActivity", "HvyAlcoholConsump", "Fruits", "Veggies"
    ]
    labels = [
        "High Blood Pressure", "High Cholesterol", "Heart Disease/Attack", "Stroke History",
        "Difficulty Walking", "Smoker (>=100 cigs)", "Sex (Male vs Female)",
        "Physical Activity", "Heavy Alcohol Consump.", "Fruit >=1/day", "Vegetables >=1/day"
    ]

    prev_no = []
    prev_yes = []
    rr_list = []

    for ind in indicators:
        p0 = float(train_df[train_df[ind] == 0]["Diabetes_binary"].mean() * 100)
        p1 = float(train_df[train_df[ind] == 1]["Diabetes_binary"].mean() * 100)
        prev_no.append(p0)
        prev_yes.append(p1)
        rr_list.append(p1 / p0 if p0 > 0 else 0)

    y_pos = np.arange(len(indicators))
    height = 0.38

    rects1 = ax.barh(y_pos - height/2, prev_no, height, label="Indicator Absent (0)", color="#4575b4", edgecolor="black")
    rects2 = ax.barh(y_pos + height/2, prev_yes, height, label="Indicator Present (1)", color="#d73027", edgecolor="black")

    # Add text annotations with prevalence and Relative Prevalence Ratio
    for i, (p0, p1, rr) in enumerate(zip(prev_no, prev_yes, rr_list)):
        ax.text(p1 + 0.8, i + height/2, f"{p1:.1f}% (RR: {rr:.1f}x)", va="center", fontsize=9, fontweight="bold", color="#a50026")
        ax.text(p0 + 0.8, i - height/2, f"{p0:.1f}%", va="center", fontsize=9, color="#313695")

    # Overall baseline prevalence reference line
    baseline_prev = float(train_df["Diabetes_binary"].mean() * 100)
    ax.axvline(baseline_prev, color="black", linestyle=":", linewidth=1.5, label=f"Cohort Baseline Prevalence ({baseline_prev:.1f}%)")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10.5)
    ax.set_xlabel("Diabetes Prevalence Rate (%) in Respondent Subgroup", fontsize=11)
    ax.set_title("Diabetes Prevalence Stratified by Clinical & Lifestyle Binary Risk Factors\nComparing Subgroups with Indicator Present (1) vs Absent (0)", pad=14, fontsize=13)
    ax.set_xlim(0, 45)
    ax.legend(loc="lower right", frameon=True, fontsize=10)

    plt.tight_layout()
    fig_path = FIGURES_DIR / "03_binary_clinical_lifestyle_risk_prevalence.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"  -> Saved: {fig_path.name}")


def plot_ordinal_risk_gradients(train_df: pd.DataFrame):
    """Figure 4: Monotonic risk gradients across ordinal scales (GenHlth, Age, Income, Education)."""
    print("  Generating Figure 4: Ordinal risk gradients...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Subplot A: GenHlth (General Health 1-5)
    gen_hlth_labels = ["1: Excellent", "2: Very Good", "3: Good", "4: Fair", "5: Poor"]
    gen_prev = train_df.groupby("GenHlth")["Diabetes_binary"].mean() * 100
    axes[0, 0].plot(range(1, 6), gen_prev.values, marker="o", color="#d73027", linewidth=2.5, markersize=8)
    for x, y in zip(range(1, 6), gen_prev.values):
        axes[0, 0].text(x, y + 1.5, f"{y:.1f}%", ha="center", fontweight="bold", fontsize=9.5)
    axes[0, 0].set_xticks(range(1, 6))
    axes[0, 0].set_xticklabels(gen_hlth_labels, rotation=15)
    axes[0, 0].set_title("A. Self-Rated General Health Gradient (GenHlth)\nSharp 10x Elevation in Risk from Excellent to Poor Health", pad=12)
    axes[0, 0].set_ylabel("Diabetes Prevalence (%)")
    axes[0, 0].set_ylim(0, 45)

    # Subplot B: Age Category (1-13)
    age_labels = ["18-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80+"]
    age_prev = train_df.groupby("Age")["Diabetes_binary"].mean() * 100
    axes[0, 1].plot(range(1, 14), age_prev.values, marker="s", color="#fc8d59", linewidth=2.5, markersize=7)
    for x, y in zip(range(1, 14), age_prev.values):
        axes[0, 1].text(x, y + 1.2, f"{y:.1f}%", ha="center", fontsize=8, fontweight="bold")
    axes[0, 1].set_xticks(range(1, 14))
    axes[0, 1].set_xticklabels(age_labels, rotation=35, fontsize=8.5)
    axes[0, 1].set_title("B. Age Category Gradient (Age: 1 to 13)\nSteep Monotonic Age-Associated Increase Plateauing at 70-74", pad=12)
    axes[0, 1].set_ylabel("Diabetes Prevalence (%)")
    axes[0, 1].set_ylim(0, 30)

    # Subplot C: Income Bracket (1-8)
    inc_labels = ["<$10k", "$10-15k", "$15-20k", "$20-25k", "$25-35k", "$35-50k", "$50-75k", ">=$75k"]
    inc_prev = train_df.groupby("Income")["Diabetes_binary"].mean() * 100
    axes[1, 0].plot(range(1, 9), inc_prev.values, marker="^", color="#4575b4", linewidth=2.5, markersize=8)
    for x, y in zip(range(1, 9), inc_prev.values):
        axes[1, 0].text(x, y + 1.2, f"{y:.1f}%", ha="center", fontweight="bold", fontsize=9)
    axes[1, 0].set_xticks(range(1, 9))
    axes[1, 0].set_xticklabels(inc_labels, rotation=20)
    axes[1, 0].set_title("C. Annual Household Income Gradient (Income: 1 to 8)\nInverse Socioeconomic Relationship: Prevalence Drops from 26% to 9%", pad=12)
    axes[1, 0].set_ylabel("Diabetes Prevalence (%)")
    axes[1, 0].set_ylim(0, 32)

    # Subplot D: Education Level (1-6)
    edu_labels = ["None/KG", "Elementary", "Some HS", "HS Grad", "Some College", "College Grad"]
    edu_prev = train_df.groupby("Education")["Diabetes_binary"].mean() * 100
    axes[1, 1].plot(range(1, 7), edu_prev.values, marker="d", color="#91bfdb", linewidth=2.5, markersize=8)
    for x, y in zip(range(1, 7), edu_prev.values):
        axes[1, 1].text(x, y + 1.2, f"{y:.1f}%", ha="center", fontweight="bold", fontsize=9)
    axes[1, 1].set_xticks(range(1, 7))
    axes[1, 1].set_xticklabels(edu_labels, rotation=20)
    axes[1, 1].set_title("D. Education Level Gradient (Education: 1 to 6)\nInverse Gradient: Higher Educational Attainment Associates with Lower Prevalence", pad=12)
    axes[1, 1].set_ylabel("Diabetes Prevalence (%)")
    axes[1, 1].set_ylim(0, 35)

    plt.tight_layout()
    fig_path = FIGURES_DIR / "04_ordinal_risk_gradients.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"  -> Saved: {fig_path.name}")


def plot_correlation_heatmap(train_df: pd.DataFrame):
    """Figure 5: Correlation matrix highlighting target relationships and multicollinearity."""
    print("  Generating Figure 5: Correlation heatmap...")
    corr_matrix = train_df.corr()

    # Reorder columns so Diabetes_binary is first
    cols = ["Diabetes_binary"] + [c for c in corr_matrix.columns if c != "Diabetes_binary"]
    corr_reordered = corr_matrix.loc[cols, cols]

    plt.figure(figsize=(16, 13))
    mask = np.triu(np.ones_like(corr_reordered, dtype=bool))

    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(
        corr_reordered,
        mask=mask,
        cmap=cmap,
        vmax=0.55,
        vmin=-0.25,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.75, "label": "Pearson Correlation Coefficient (r)"},
        annot=True,
        fmt=".2f",
        annot_kws={"size": 7.5},
    )

    plt.title("Correlation Matrix of CDC Diabetes Indicators (Train Cohort: N=160,631)\nOrdered by Target (`Diabetes_binary`) with Pairwise Inter-Feature Collinearity", pad=16, fontsize=13)
    plt.xticks(rotation=45, ha="right", fontsize=9.5)
    plt.yticks(fontsize=9.5)

    plt.tight_layout()
    fig_path = FIGURES_DIR / "05_correlation_heatmap_and_feature_associations.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"  -> Saved: {fig_path.name}")


def run_eda():
    print("=" * 80)
    print("PHASE 6: CDC DIABETES HEALTH INDICATORS - EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 80)

    # Load training partition strictly (avoid test snooping)
    train_path = PROJECT_ROOT / "data" / "processed" / "diabetes" / "diabetes_train.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Training dataset not found at {train_path}. Run Phase 5 data cleaning first.")

    train_df = pd.read_csv(train_path)
    print(f"Loaded Clean Training Partition: {train_df.shape[0]:,} rows x {train_df.shape[1]} columns")

    # Load 3-class raw data for comparison figure
    df_012 = load_cdc_diabetes_data("012")

    # Generate the 5 focused figures
    plot_target_distribution_and_imbalance(train_df, df_012)
    plot_numerical_distributions_and_outliers(train_df)
    plot_binary_risk_prevalence(train_df)
    plot_ordinal_risk_gradients(train_df)
    plot_correlation_heatmap(train_df)

    # Save quantitative EDA summary JSON
    eda_summary = {
        "dataset_analyzed": "CDC Diabetes Clean Training Partition (70% split)",
        "sample_size": len(train_df),
        "target_prevalence_pct": round(float(train_df["Diabetes_binary"].mean() * 100), 2),
        "top_positive_correlations": {
            k: round(float(v), 4)
            for k, v in train_df.corr()["Diabetes_binary"].sort_values(ascending=False).items()
            if k != "Diabetes_binary"
        },
        "bmi_stats_by_class": {
            "non_diabetic": {
                "median": float(train_df[train_df["Diabetes_binary"] == 0]["BMI"].median()),
                "mean": round(float(train_df[train_df["Diabetes_binary"] == 0]["BMI"].mean()), 2),
                "q25": float(train_df[train_df["Diabetes_binary"] == 0]["BMI"].quantile(0.25)),
                "q75": float(train_df[train_df["Diabetes_binary"] == 0]["BMI"].quantile(0.75)),
            },
            "diabetic": {
                "median": float(train_df[train_df["Diabetes_binary"] == 1]["BMI"].median()),
                "mean": round(float(train_df[train_df["Diabetes_binary"] == 1]["BMI"].mean()), 2),
                "q25": float(train_df[train_df["Diabetes_binary"] == 1]["BMI"].quantile(0.25)),
                "q75": float(train_df[train_df["Diabetes_binary"] == 1]["BMI"].quantile(0.75)),
            },
        },
        "figures_generated": [
            "01_target_distribution_and_imbalance.png",
            "02_numerical_distributions_and_outliers.png",
            "03_binary_clinical_lifestyle_risk_prevalence.png",
            "04_ordinal_risk_gradients.png",
            "05_correlation_heatmap_and_feature_associations.png",
        ],
    }

    summary_path = PROJECT_ROOT / "reports" / "eda_summary.json"
    with open(summary_path, "w") as f:
        json.dump(eda_summary, f, indent=2)
    print(f"\n[OK] Quantitative EDA summary persisted to: {summary_path.relative_to(PROJECT_ROOT)}")
    print("=" * 80)


if __name__ == "__main__":
    run_eda()
