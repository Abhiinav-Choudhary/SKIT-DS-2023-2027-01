# AI/ML Verification & Test Report

**Project**: SKIT-DS-2023-2027-01 (AI-Powered Health Risk Prediction and Monitoring System)  
**Author**: Lead AI/ML Engineer (Abhiram)  
**Date**: 15 November 2026  
**Status**: All Tests Passed (53 / 53 Passed)  
**Execution Environment**: Python 3.13.5, Pytest 8.4.2 / 9.1.1, Scikit-learn 1.9.1 / 1.6.0+, Windows 11  

---

## 1. Executive Summary

This test report details the verification and validation of the entire AI/ML subsystem for project `SKIT-DS-2023-2027-01`. The suite has been expanded from the initial 34 tests to **53 automated unit, integration, and regression tests**.

All 53 tests execute and pass with zero failures and zero regressions.

```
============================= 53 passed in ~35s =============================
```

### Coverage Breakdown
- Data Ingestion & Quality: 4 tests (100% Pass)
- Data Cleaning, Deduplication & Preprocessing: 4 tests (100% Pass)
- Feature Engineering & Leakage Isolation: 5 tests (100% Pass)
- Baseline Model Architectures & Metric Formulation: 4 tests (100% Pass)
- CDC Diabetes Production Inference Pipeline: 14 tests (100% Pass)
- UCI Heart Disease Production Inference Pipeline: 16 tests (100% Pass)
- CLI Runner & Backend Integration Contracts: 6 tests (100% Pass)

---

## 2. Test Execution Results Matrix

### 2.1 CDC Diabetes Ingestion (`tests/test_data_loader.py`)

| # | Test Identifier | Expected Outcome | Actual Outcome | Status |
|---|---|---|---|---|
| 1 | `test_load_cdc_diabetes_binary_shape_and_columns` | Raw binary dataset loads with 253,680 rows and 22 expected columns. | Exactly 253,680 rows and 22 columns loaded matching schema. | PASS |
| 2 | `test_load_cdc_diabetes_012` | 3-class raw dataset loads with 253,680 rows and `Diabetes_012` target. | Correctly loaded with target categories 0, 1, and 2. | PASS |
| 3 | `test_load_cdc_diabetes_invalid_variant` | Invalid variant name raises `ValueError`. | `ValueError` raised as expected. | PASS |
| 4 | `test_no_null_values_in_raw_data` | Zero null or missing cells exist in raw CDC dataset. | 0 null cells across all 22 columns verified. | PASS |

### 2.2 Preprocessing, Cleaning & Stratification (`tests/test_preprocessing.py`)

| # | Test Identifier | Expected Outcome | Actual Outcome | Status |
|---|---|---|---|---|
| 5 | `test_audit_data_quality_structure` | Audit returns dictionary with record counts, null counts, duplicate counts, and imbalance. | Complete audit dictionary returned and verified. | PASS |
| 6 | `test_cleaning_deduplication` | Deduplication drops identical feature profiles; verifies unique counts. | Correctly detects 24,206 duplicate rows in BRFSS survey responses. | PASS |
| 7 | `test_stratified_split` | Stratified split retains target prevalence ratio (~13.93%) across train, val, and test partitions. | Class 1 proportion in train (13.93%), val (13.93%), test (13.93%) verified. | PASS |
| 8 | `test_preprocessor_transformation` | Scikit-learn preprocessor scales continuous features and passes binary features cleanly. | Output shape matches feature space without numeric distortion. | PASS |

### 2.3 Feature Engineering & Data Leakage Protection (`tests/test_feature_engineering.py`)

| # | Test Identifier | Expected Outcome | Actual Outcome | Status |
|---|---|---|---|---|
| 9 | `test_feature_engineer_output_shape` | Engineered feature transformer outputs 27 columns from 21 raw columns. | Exactly 27 columns produced with expected feature names. | PASS |
| 10 | `test_bmi_category_calculation` | `BMI_Category` binned into underweight (1), normal (2), overweight (3), obese (4). | Precise clinical category mappings verified. | PASS |
| 11 | `test_cardiometabolic_risk_score` | Sum of cardiometabolic risk indicators computed accurately. | Valid linear accumulation verified across risk indicators. | PASS |
| 12 | `test_zero_leakage_row_independence` | Feature transformation of any row is mathematically independent of neighboring rows. | Zero leakage confirmed; single row transformed in isolation matches batch transformation. | PASS |
| 13 | `test_full_pipeline_persistence` | Pipeline containing preprocessor and estimator can be serialized and deserialized cleanly via joblib. | Roundtrip serialization succeeds and retains pipeline parameters. | PASS |

### 2.4 Baseline Models & Evaluation Functions (`tests/test_models.py`)

| # | Test Identifier | Expected Outcome | Actual Outcome | Status |
|---|---|---|---|---|
| 14 | `test_get_baseline_models` | Factory returns 4 canonical baseline estimators (LR, DT, RF, GB). | Dict with all 4 estimators instantiated with specified random seed. | PASS |
| 15 | `test_evaluate_predictions_metrics` | Metric evaluator computes accuracy, precision, recall, f1, roc_auc, pr_auc, and confusion matrix. | Exact mathematical calculation verified against known truth vector. | PASS |
| 16 | `test_baseline_model_prediction_shape` | Model fits synthetic matrix and produces expected prediction and probability shapes. | Output shapes (20,) and (20, 2) verified with valid probabilities in [0, 1]. | PASS |
| 17 | `test_saved_model_artifact_loading` | Baseline model artifacts load from disk and provide prediction interfaces. | Verified baseline estimators load and expose `predict` / `predict_proba`. | PASS |

### 2.5 CDC Diabetes Production Inference (`tests/test_inference_diabetes.py`)

| # | Test Identifier | Expected Outcome | Actual Outcome | Status |
|---|---|---|---|---|
| 18 | `test_saved_model_loads` | Champion pipeline loads from `models/diabetes/diabetes_risk_pipeline.joblib`. | Loads successfully; steps: `preprocess` and `clf` (`HistGradientBoostingClassifier`). | PASS |
| 19 | `test_valid_prediction` | Valid 21-feature survey record produces binary prediction and continuous probability. | Predicted class in `[0, 1]`, probability in `[0.0, 1.0]`. | PASS |
| 20 | `test_output_structure` | Single dict and batch DataFrame produce matching output structure. | Correct key set: `{"predicted_class", "diabetes_probability"}`. | PASS |
| 21 | `test_missing_input_rejected` | Omission of any of 21 raw columns raises `ValueError`. | `ValueError` raised identifying missing column. | PASS |
| 22 | `test_out_of_range_input_rejected` | Out-of-range BMI (e.g. 200) or Age (e.g. 99) raises `ValueError`. | `ValueError` raised with specific out-of-range feature details. | PASS |
| 23 | `test_non_integer_input_rejected` | Fractional values in integer-coded survey fields raise `ValueError`. | `ValueError` raised specifying field must be integer-coded. | PASS |
| 24 | `test_non_numeric_input_rejected` | String or non-numeric values in numeric fields raise `ValueError`. | `ValueError` raised cleanly during coercion validation. | PASS |
| 25 | `test_inference_does_not_retrain` | Inference execution never triggers `fit` or `fit_transform`. | Monkeypatched `fit` never invoked; inference is strictly read-only. | PASS |
| 26 | `test_inference_does_not_rewrite_model_file` | SHA256 checksum of saved `.joblib` model artifact unchanged before and after inference. | Checksums before and after are byte-identical. | PASS |
| 27 | `test_model_loading_missing_file_raises_error` | `load_diabetes_risk_pipeline` raises `FileNotFoundError` when artifact is absent. | `FileNotFoundError` raised with descriptive message. | PASS |
| 28 | `test_model_loading_invalid_type_raises_error` | `load_diabetes_risk_pipeline` raises `TypeError` when artifact is not a Pipeline. | `TypeError` raised when non-pipeline object is supplied. | PASS |
| 29 | `test_output_json_serializability` | Output dictionary serializes to JSON string and deserializes cleanly. | Natively JSON-serializable; `json.dumps()` roundtrip matches perfectly. | PASS |
| 30 | `test_deterministic_regression` | Fixed canonical test sample produces deterministic predicted class 0 and probability 0.1785. | Output matches expected benchmark within 1e-4 tolerance. | PASS |
| 31 | `test_extra_fields_ignored_and_order_invariance` | Extra payload fields (e.g. `patient_id`) ignored; shuffled column order produces identical output. | Prediction identical to baseline within 1e-6 tolerance. | PASS |

### 2.6 UCI Heart Disease Production Inference (`tests/test_inference_heart_disease.py`)

| # | Test Identifier | Expected Outcome | Actual Outcome | Status |
|---|---|---|---|---|
| 32 | `test_production_model_loads_from_default_path` | Heart disease pipeline loads from `models/heart/heart_disease_risk_pipeline.joblib`. | Loads successfully; steps: `features` and `clf` (`LogisticRegression` classes 0–4). | PASS |
| 33 | `test_valid_prediction` | Valid 13-feature clinical record produces predicted multiclass label in [0, 1, 2, 3, 4]. | Label in [0, 1, 2, 3, 4] returned. | PASS |
| 34 | `test_probability_output_valid` | Returns 5-class probability vector where each probability $\in [0, 1]$ and sum equals 1.0. | Verified: 5 class probabilities summing to 1.000000. | PASS |
| 35 | `test_output_structure` | Keys match `{"predicted_class", "class_labels", "class_probabilities"}` for dict and DataFrame. | Exact key contract verified across single and batch runs. | PASS |
| 36 | `test_missing_input_rejected` | Omission of any required clinical feature raises `ValueError`. | `ValueError` raised identifying missing column. | PASS |
| 37 | `test_invalid_input_rejected` | Bad categorical values (e.g. `cp=5`), zero measurements (`trestbps=0`), or non-numeric values raise `ValueError`. | `ValueError` raised explaining validation violation. | PASS |
| 38 | `test_inference_does_not_retrain` | Inference execution never triggers `fit` or `fit_transform`. | Monkeypatched `fit` never called during inference. | PASS |
| 39 | `test_inference_does_not_rewrite_model_file` | SHA256 checksum of saved heart disease `.joblib` model artifact unchanged after inference. | Checksums before and after are byte-identical. | PASS |
| 40 | `test_model_loading_missing_file_raises_error` | Missing model file raises `FileNotFoundError`. | `FileNotFoundError` raised with descriptive message. | PASS |
| 41 | `test_model_loading_invalid_type_raises_error` | Artifact of improper type or missing pipeline steps raises `TypeError`. | `TypeError` raised when non-pipeline object is supplied. | PASS |
| 42 | `test_out_of_range_numerical_input_rejected` | Physiologically invalid values (negative age, trestbps > 350, negative thalach) raise `ValueError`. | `ValueError` raised with specific out-of-range bounds. | PASS |
| 43 | `test_non_finite_numerical_input_rejected` | Infinite numerical values (`inf`, `-inf`) raise `ValueError`. | `ValueError` raised on non-finite input. | PASS |
| 44 | `test_output_json_serializability` | Heart prediction output serializes to JSON string and deserializes cleanly. | Natively JSON-serializable; `json.dumps()` roundtrip matches perfectly. | PASS |
| 45 | `test_deterministic_regression` | Fixed canonical test sample produces predicted class 2 and probability vector matching baseline. | Deterministic class 2 and probability [0.0283, 0.0298, 0.5260, 0.2081, 0.2077] confirmed. | PASS |
| 46 | `test_multiclass_probabilities_batch_consistency` | Batch of $N$ records produces $(N, 5)$ probability matrix, each row summing to 1.0. | Verified: batch dimensions and probability normalization confirmed. | PASS |
| 47 | `test_extra_fields_ignored_and_order_invariance` | Extra patient metadata ignored; shuffled column order produces identical probability vector. | Prediction identical to baseline within 1e-6 tolerance. | PASS |

### 2.7 CLI Runner & Integration Contract (`tests/test_cli_and_contracts.py`)

| # | Test Identifier | Expected Outcome | Actual Outcome | Status |
|---|---|---|---|---|
| 48 | `test_cli_diabetes_input_flag` | Invoking CLI with `--model diabetes --input '{...}'` outputs JSON on stdout with code 0. | Success status, model name, and prediction data returned. | PASS |
| 49 | `test_cli_heart_input_flag` | Invoking CLI with `--model heart --input '{...}'` outputs JSON on stdout with code 0. | Success status, model name, and multiclass prediction data returned. | PASS |
| 50 | `test_cli_stdin_pipe` | Piping JSON via standard input to CLI runner generates valid prediction. | Piped JSON parsed and executed cleanly. | PASS |
| 51 | `test_cli_invalid_json_handling` | Malformed JSON on stdin/input causes non-zero exit code (1) and JSON error on stderr. | Exits with code 1; structured JSON error emitted on stderr. | PASS |
| 52 | `test_cli_validation_error_handling` | Input violating validation bounds emits descriptive JSON error on stderr with code 1. | Exits with code 1; validation message detailing invalid field returned. | PASS |
| 53 | `test_dict_and_dataframe_equivalence` | Calling inference functions with a Python dictionary or single-row DataFrame produces identical outputs. | Output dictionaries match exactly across both models. | PASS |

---

## 3. Bug Fixes and Enhancements Applied

1. **Heart Disease Numerical Validation**:
   - Added `NUMERIC_FEATURE_VALUE_RANGES` to `src/inference/heart_disease_model.py` covering physiological bounds:
     - `age`: `[1, 120]`
     - `trestbps`: `[40, 350]`
     - `chol`: `[40, 1000]`
     - `thalach`: `[30, 260]`
     - `oldpeak`: `[-10.0, 15.0]`
   - Prevents absurd inputs (e.g. negative age, extreme non-physiological blood pressure) from reaching the model.

2. **Headless Subprocess CLI Runner**:
   - Created `src/inference/cli.py` to allow Node.js / Express backend to execute inference via stdin/stdout JSON streaming without requiring external Python HTTP frameworks or microservices.

3. **Robust Model Error Handling**:
   - Added explicit unit tests verifying `FileNotFoundError` and `TypeError` when model paths are missing or artifacts are corrupted.

4. **JSON Serialization Assurance**:
   - Verified that numpy arrays and integer types in prediction outputs convert cleanly into native Python `int` and `float` primitives, guaranteeing zero `TypeError: Object of type int64 is not JSON serializable` errors during backend serialization.

---

## 4. Conclusion

The AI/ML module has achieved complete test coverage for all core operations, inference pathways, input validation, edge cases, error conditions, and integration contracts. All 53 tests pass deterministically.
