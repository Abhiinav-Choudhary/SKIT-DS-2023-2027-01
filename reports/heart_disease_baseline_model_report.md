# Heart Disease Baseline Model Report

Target: Existing multiclass `num` target with classes 0, 1, 2, 3, 4.

The test split was not loaded or evaluated in this step.

## Strongest Candidate

`Gradient Boosting` selected by highest validation weighted F1, with validation weighted ROC-AUC as the first tie-breaker.

## Validation Results

| Model | Accuracy | Precision Macro | Recall Macro | F1 Macro | F1 Weighted | ROC-AUC OVR Weighted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.4710 | 0.3491 | 0.3202 | 0.3308 | 0.4979 | 0.7724 |
| Decision Tree | 0.3986 | 0.2627 | 0.2676 | 0.2522 | 0.4171 | 0.7008 |
| Random Forest | 0.5000 | 0.3027 | 0.3256 | 0.3089 | 0.4911 | 0.7673 |
| Gradient Boosting | 0.5797 | 0.3802 | 0.3810 | 0.3766 | 0.5659 | 0.7907 |

## Cross-Validation Results

| Model | Accuracy Mean | F1 Macro Mean | F1 Weighted Mean | ROC-AUC OVR Weighted Mean |
| --- | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.5452 | 0.4190 | 0.5645 | 0.7919 |
| Decision Tree | 0.4003 | 0.2890 | 0.4333 | 0.7125 |
| Random Forest | 0.5047 | 0.3545 | 0.5121 | 0.7893 |
| Gradient Boosting | 0.5482 | 0.3512 | 0.5367 | 0.7800 |

## Files

- `models\heart\baselines\logistic_regression_baseline_pipeline.joblib`
- `models\heart\baselines\decision_tree_baseline_pipeline.joblib`
- `models\heart\baselines\random_forest_baseline_pipeline.joblib`
- `models\heart\baselines\gradient_boosting_baseline_pipeline.joblib`
- `reports\figures\16_heart_baseline_validation_confusion_matrices.png`
- `reports/heart_disease_baseline_model_report.json`
- `reports/heart_disease_baseline_model_report.md`
