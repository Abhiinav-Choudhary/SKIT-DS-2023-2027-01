# Heart Disease EDA Report

This report uses the existing Heart Disease target column `num` with values 0-4. No binary target was created.

## Dataset

- Raw shape: 920 rows x 15 columns.
- Cleaned EDA shape: 918 rows x 15 columns.
- Raw target distribution: {'0': 411, '1': 265, '2': 109, '3': 107, '4': 28}.
- Cleaned target distribution: {'0': 410, '1': 265, '2': 108, '3': 107, '4': 28}.
- Raw missing cells: 1759.
- Cleaned missing cells: 1925.

## Key Findings

- The cleaned EDA dataset has 918 rows after dropping 2 exact duplicate rows; target `num` values remain [0, 1, 2, 3, 4].
- Class `num=0` is the largest cleaned class with 410 rows (44.66%); class `num=4` is the smallest with 28 rows (3.05%).
- Raw missingness is concentrated in ca (611 rows), thal (486 rows), and slope (309 rows).
- The strongest absolute Spearman correlations with `num` among numerical features are oldpeak=0.436, thalach=-0.397, age=0.346.
- The largest categorical associations with `num` by bias-corrected Cramer's V are exang=0.321, cp=0.308, sex=0.300, slope=0.236.
- IQR outlier counts are highest for trestbps (27 rows), followed by chol (23 rows).
- Observed anomaly counts include raw trestbps=0 in 1 row, raw chol=0 in 172 rows, and cleaned oldpeak<0 in 12 rows.

## Generated Figures

- `reports\figures\10_heart_target_and_source_distribution.png`
- `reports\figures\11_heart_numeric_distributions.png`
- `reports\figures\12_heart_categorical_distributions.png`
- `reports\figures\13_heart_missingness_patterns.png`
- `reports\figures\14_heart_feature_target_relationships.png`
- `reports\figures\15_heart_numeric_spearman_correlation.png`
