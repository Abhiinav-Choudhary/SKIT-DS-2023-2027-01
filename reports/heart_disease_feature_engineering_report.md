# Heart Disease Feature Engineering Report

## Target Definition

Using existing multiclass target `num` with values [0, 1, 2, 3, 4]. Existing Heart code and reports define `num` as the target and do not define a project-level binary remapping.

## Feature Changes

- Original model-input predictors: 13.
- Engineered predictors added: 11.
- Raw predictors after engineering: 24.
- Final transformed feature count: 46.
- Removed from model input: `source` and `num`.

## Engineered Features

- `age_x_oldpeak`: age multiplied by exercise-induced ST depression (`oldpeak`).
- `resting_bp_cholesterol_index`: resting blood pressure multiplied by serum cholesterol.
- `heart_rate_reserve_proxy`: (220 - age) - maximum heart rate achieved (`thalach`).
- `oldpeak_per_age`: exercise-induced ST depression divided by age.
- `asymptomatic_chest_pain`: 1 when `cp` equals 4, otherwise 0; missing preserved.
- `exercise_ischemia_signal`: 1 when `exang` is 1 or `oldpeak` is above 0, otherwise 0 when observed.
- `has_major_vessels_colored`: 1 when `ca` is above 0, otherwise 0; missing preserved.
- `thal_reversible_defect`: 1 when `thal` equals 7, otherwise 0; missing preserved.
- `high_resting_bp`: 1 when `trestbps` is at least 140, otherwise 0; missing preserved.
- `high_cholesterol`: 1 when `chol` is at least 240, otherwise 0; missing preserved.
- `abnormal_restecg`: 1 when `restecg` is above 0, otherwise 0; missing preserved.

## Leakage Checks

- target_column_excluded_from_model_features: PASSED
- source_column_excluded_from_model_features: PASSED
- engineered_features_are_row_wise: PASSED
- no_target_encoding_or_target_derived_features: PASSED
- pipeline_fit_only_on_training_split: PASSED
- validation_and_test_transformed_without_refit: PASSED
- target_distribution_preserved_in_saved_feature_files: PASSED

## Outputs

- `data\processed\heart_disease\features\heart_disease_train_features.csv`
- `data\processed\heart_disease\features\heart_disease_val_features.csv`
- `data\processed\heart_disease\features\heart_disease_test_features.csv`
- `models\heart\heart_disease_feature_pipeline.joblib`
- `reports/heart_disease_feature_engineering_report.json`
- `reports/heart_disease_feature_engineering_report.md`
