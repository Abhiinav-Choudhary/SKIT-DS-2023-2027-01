# AI-Powered Health Risk Prediction and Monitoring System

**College Final Year Project (2026–2027)**
**Role**: Abhiram — Lead AI/ML Engineer
**Current Milestone Date**: 15 November 2026
**Primary Dataset**: CDC Diabetes Health Indicators (UCI Dataset #891 / BRFSS 2015)
**Secondary Dataset**: UCI Heart Disease (UCI Dataset #45)

> **Medical & Academic Disclaimer**
> This project contains **prediction / risk-classification components only**. It is an academic research prototype. It is **not** a medical diagnosis system, is **not** medically certified or clinically validated, and must **not** be used for diagnostic decisions or as a substitute for professional clinical judgment.

---

## 1. Project Objective

Develop a modular, reproducible, production-grade AI-powered health risk prediction system that uses demographic, behavioral, and clinical indicator data to produce calibrated probability outputs for two chronic conditions (diabetes and heart disease). The system is designed so the trained pipelines can be served by backend APIs and surfaced on an interactive dashboard.

The scope of this repository is **model training, evaluation, persistence, and inference**. It does not implement a medical diagnosis system.

---

## 2. Repository Structure

```text
SKIT-DS-2023-2027-01/
├── data/
│   ├── raw/                      # Raw CDC diabetes + UCI heart disease files
│   └── processed/                # Cleaned, stratified train/val/test splits
├── notebooks/                    # Step-by-step pipeline scripts (01–13)
├── src/
│   ├── preprocessing/            # data_loader, data_cleaner, pipeline, heart_disease
│   ├── features/                 # cdc_diabetes_metadata, feature_engineer, heart_disease_features
│   ├── models/                   # baseline_models, hyperparameter_tuning
│   ├── evaluation/               # metrics
│   └── inference/                # diabetes_risk_model, heart_disease_model, cli
├── models/
│   ├── diabetes/                 # diabetes_risk_pipeline.joblib + metadata.json
│   └── heart/                    # heart_disease_risk_pipeline.joblib + metadata.json
├── reports/                      # JSON/MD reports + figures/
├── tests/                        # pytest suite (53 tests)
├── docs/                         # ML_BACKEND_INTEGRATION.md, AI_ML_FINAL_SUMMARY.md, ...
├── requirements.txt
└── README.md
```

---

## 3. Datasets

### 3.1 CDC Diabetes Health Indicators (BRFSS 2015)

- **Source**: UCI Machine Learning Repository [Dataset #891](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators), derived from CDC BRFSS 2015.
- **Size**: 253,680 respondents × 22 columns (21 predictor features + 1 target).
- **Target `Diabetes_binary`**:
  - `0` (no diabetes, includes prediabetes): 218,334 (**86.07%**)
  - `1` (diagnosed diabetes): 35,346 (**13.93%**)
  - Class imbalance ratio **6.18 : 1**.
- **Data quality**: 0 missing cells; 24,206 exact duplicate rows (9.54%) — expected because all 21 features are discrete/binned survey responses without patient identifiers; 227,908 unique feature profiles, of which 1,566 (0.69%) have conflicting labels.
- **Stratified splits**: train 160,631 / validation 34,421 / test 34,422.

### 3.2 UCI Heart Disease

- **Source**: UCI Heart Disease (Dataset #45), four processed files: `cleveland` (303), `hungarian` (294), `va` (200), `switzerland` (123) = **920 rows × 15 columns** (13 predictors + `source` + target).
- **Target `num`**: existing multiclass target, values **0–4** (no binary remapping). Cleaned distribution: 0=410, 1=265, 2=108, 3=107, 4=28.
- **Cleaning**: 2 exact duplicate rows dropped → **918 rows**; zero measurements converted to missing for `trestbps` (1 row) and `chol` (172 rows); total missing cells 1,759 (raw) → 1,925 (cleaned). Missingness is concentrated in `ca`, `thal`, and `slope`.
- **Stratified splits**: train 642 / validation 138 / test 138.

---

## 4. Setup & Reproducibility

Python 3.13+ is supported. Install dependencies via:

```bash
pip install -r requirements.txt
```

Run the test suite:

```bash
pytest tests/ -v
```

---

## 5. Preprocessing

**Diabetes** (`src/preprocessing/`): raw CSV ingestion with schema verification, cleaning, and a stratified train/validation/test split. The production pipeline applies feature engineering and a `ColumnTransformer` (standard scaling on continuous columns; ordinal/binary passthrough) inside the serialized pipeline.

**Heart Disease** (`src/preprocessing/heart_disease.py`): loads the four processed UCI files with `?` as the missing marker, coerces all columns to numeric, converts zero `trestbps`/`chol` to missing, drops exact duplicates, then performs a target-stratified train/validation/test split. Preprocessing uses median imputation + `StandardScaler` for numeric features and most-frequent imputation + `OneHotEncoder(handle_unknown="ignore")` for categorical features.

---

## 6. Feature Engineering

**Diabetes** (`src/features/feature_engineer.py`): 21 → **27** features. Six engineered features, all strictly row-wise (leakage audit PASSED):

| Feature | Type | Target correlation |
| --- | --- | --- |
| `BMI_Category` | Ordinal (1–6) | +0.2156 |
| `CardioMetabolic_Risk_Score` | Ordinal (0–4) | +0.2999 |
| `HighBP_x_HighChol` | Binary | +0.2566 |
| `Healthy_Lifestyle_Score` | Ordinal (0–5) | −0.0723 |
| `Age_x_GenHlth` | Continuous (scaled) | +0.3032 |
| `Sedentary_Obese` | Binary | +0.1508 |

**Heart Disease** (`src/features/heart_disease_features.py`): 13 predictors → 24 (11 engineered) → **46** transformed features. Engineered numeric: `age_x_oldpeak`, `resting_bp_cholesterol_index`, `heart_rate_reserve_proxy`, `oldpeak_per_age`. Engineered categorical/binary: `asymptomatic_chest_pain`, `exercise_ischemia_signal`, `has_major_vessels_colored`, `thal_reversible_defect`, `high_resting_bp`, `high_cholesterol`, `abnormal_restecg`.

---

## 7. Models Used

Four baseline estimators are used for both conditions (`src/models/baseline_models.py`): **Logistic Regression**, **Decision Tree**, **Random Forest**, and **Gradient Boosting** (`HistGradientBoostingClassifier`).

- **Diabetes champion**: tuned `HistGradientBoostingClassifier` (class_weight=balanced).
- **Heart Disease selected model**: tuned `LogisticRegression` (C=0.3, class_weight=balanced) on the existing multiclass `num` target.

---

## 8. Baseline Results

### 8.1 Diabetes (5-fold CV on train + one evaluation on validation)

| Model | CV ROC-AUC | Validation Accuracy | Validation ROC-AUC |
| --- | --- | --- | --- |
| Logistic Regression | 0.8070 | 0.7134 | 0.8110 |
| Decision Tree | 0.7901 | 0.7089 | 0.7993 |
| Random Forest | 0.8108 | 0.7439 | 0.8139 |
| Gradient Boosting | 0.8134 | 0.7126 | 0.8169 |

### 8.2 Heart Disease (5-fold CV on train + one evaluation on validation)

Validation weighted-F1: Gradient Boosting **0.5659** > Logistic Regression 0.4979 > Random Forest 0.4911 > Decision Tree 0.4171. Baseline training CV favored Logistic Regression, so both were carried into tuning. The test split was untouched during baselining.

---

## 9. Hyperparameter Tuning Results

**Diabetes** (`reports/hyperparameter_tuning_and_final_evaluation.json`): `RandomizedSearchCV`, `StratifiedKFold(5)`, scoring `roc_auc`, **training split only**, preprocessing inside the pipeline.

| Candidate | Best CV ROC-AUC | Best params |
| --- | --- | --- |
| Gradient Boosting (champion) | **0.8139** | min_samples_leaf=75, max_iter=125, max_depth=None, learning_rate=0.08, l2_regularization=10.0 |
| Random Forest | 0.8114 | n_estimators=100, min_samples_split=20, min_samples_leaf=20, max_features=sqrt, max_depth=12 |

Champion = highest training CV ROC-AUC (Gradient Boosting).

**Heart Disease** (`reports/heart_disease_hyperparameter_tuning_report.json`): training split only; `GridSearchCV` for Logistic Regression and `RandomizedSearchCV(n_iter=16)` for Gradient Boosting; refit metric `f1_weighted`; feature pipeline refit inside each CV fold.

| Candidate | Search | Best CV weighted-F1 | Best params |
| --- | --- | --- | --- |
| Logistic Regression (selected) | GridSearchCV | **0.5671** | C=0.3, class_weight=balanced |
| Gradient Boosting | RandomizedSearchCV | 0.5573 | min_samples_leaf=30, max_leaf_nodes=31, max_iter=50, max_depth=3, learning_rate=0.15, l2_regularization=1.0, class_weight=balanced |

Selected = highest mean CV weighted-F1 (Logistic Regression). Validation and test splits were not used.

---

## 10. Final Test Results

**Diabetes** (champion, single evaluation on the untouched test set, N=34,422):

| Metric | Value |
| --- | --- |
| Accuracy | 0.7116 |
| Precision | 0.3210 |
| Recall | 0.7941 |
| Specificity | 0.6967 |
| F1 | 0.4572 |
| Balanced accuracy | 0.7454 |
| ROC-AUC | 0.8217 |
| PR-AUC | 0.4522 |

Confusion matrix: TN=20,314, FP=8,843, FN=1,084, TP=4,181. Recall is high but precision is low, reflecting the 13.93% class imbalance.

**Heart Disease** (selected Logistic Regression, single evaluation on the untouched test set, N=138):

| Metric | Value |
| --- | --- |
| Accuracy | 0.4783 |
| Precision (macro / weighted) | 0.3615 / 0.5608 |
| Recall (macro / weighted) | 0.3980 / 0.4783 |
| F1 (macro / weighted) | 0.3528 / 0.5104 |
| ROC-AUC OVR (macro / weighted) | 0.7550 / 0.7816 |

These are modest multiclass results, driven mainly by strong class-0 performance and weak class-3/4 performance (class 3 precision 0.0476, class 4 precision 0.1176), consistent with the small sample size and few class-3/4 cases.

---

## 11. Saved Model Paths

| Condition | Production pipeline | Metadata |
| --- | --- | --- |
| Diabetes | `models/diabetes/diabetes_risk_pipeline.joblib` | `models/diabetes/metadata.json` |
| Heart Disease | `models/heart/heart_disease_risk_pipeline.joblib` | `models/heart/metadata.json` |

Supporting artifacts:
- Diabetes baselines: `models/diabetes/baselines/*.joblib`
- Heart baselines: `models/heart/baselines/*_baseline_pipeline.joblib`
- Heart tuned pipelines: `models/heart/tuned/` (the production heart pipeline is byte-identical to `models/heart/tuned/heart_disease_selected_tuned_pipeline.joblib`)

---

## 12. Inference Functions

**Diabetes** (`src/inference/diabetes_risk_model.py`):

```python
predict_diabetes_risk(features, pipeline=None) -> dict
```
- Accepts a `dict` (one record) or `pandas.DataFrame` (one or more records).
- Returns `{"predicted_class": [...], "diabetes_probability": [...]}`.
- Contract documented in `docs/diabetes_backend_integration_contract.md` and unified guide in `docs/ML_BACKEND_INTEGRATION.md`.

**Heart Disease** (`src/inference/heart_disease_model.py`):

```python
predict_heart_disease(features, pipeline=None) -> dict
```
- Accepts a `dict` (one record) or `pandas.DataFrame` (one or more records).
- Returns `{"predicted_class": [...], "class_labels": [0,1,2,3,4], "class_probabilities": [[...]]}`.
- Contract documented in `docs/ML_BACKEND_INTEGRATION.md`.

Both load their production pipeline from the default path when `pipeline` is not supplied, and perform input validation before prediction.

A unified headless CLI runner (`src/inference/cli.py`) is also provided for Node.js / Express backend subprocess integration:
```bash
python -m src.inference.cli --model <diabetes|heart>
```

---

## 13. Input / Output Behavior

**Diabetes input**: 21 raw CDC feature columns. All values must be numeric and non-missing; all fields except `BMI` must be integer-coded. Allowed ranges: binary fields 0–1; `BMI` 12–100; `GenHlth` 1–5; `MentHlth`/`PhysHlth` 0–30; `Age` 1–13; `Education` 1–6; `Income` 1–8. Missing columns, non-numeric/missing values, out-of-range values, and non-integer integer-coded fields each raise `ValueError`.

**Heart Disease input**: 13 raw UCI feature columns. All values must be numeric, finite, and non-missing; `trestbps` and `chol` must be non-zero (existing preprocessing treats zero as missing); categorical fields must be within the values seen during training; numerical fields must be within valid physiological bounds (`age` 1–120, `trestbps` 40–350, `chol` 40–1000, `thalach` 30–260, `oldpeak` -10.0–15.0). Violations raise `ValueError`.

**Output**: class predictions and class probabilities only. No clinical threshold, risk category, or diagnostic text is produced by the inference functions.

---

## 14. Automated Testing

```bash
pytest tests/ -v
```

The suite contains **53 tests** (all passing), including dedicated inference, regression, and contract tests for both systems:

- **Diabetes** (`tests/test_inference_diabetes.py`): 14 tests covering model loading, predictions, output structure, missing input, out-of-range input, non-integer input, non-numeric input, read-only inference, checksum immutability, `FileNotFoundError`, `TypeError`, JSON serializability, deterministic regression, and extra field invariance.
- **Heart Disease** (`tests/test_inference_heart_disease.py`): 16 tests covering default loading, predictions, 5-class probability normalization, output structure, missing input, invalid categories, zero measurements, read-only inference, checksum immutability, `FileNotFoundError`, `TypeError`, physiological out-of-range bounds, non-finite values, JSON serializability, deterministic regression, and batch consistency.
- **CLI & Backend Contracts** (`tests/test_cli_and_contracts.py`): 6 tests covering `--model` flags, stdin streaming, invalid JSON handling, validation error formatting, and dict/DataFrame parity.
- **Data, Preprocessing & Feature Engineering** (`tests/test_*.py`): 17 tests covering ingestion, quality audits, deduplication, stratified splitting, and feature transformers.


---

## 15. Limitations

- **Not a diagnosis system**: both models are prediction/risk-classification components; outputs are probabilities, not clinical diagnoses.
- **Diabetes**: severe class imbalance (13.93% prevalence) yields high recall (0.7941) but low precision (0.3210); data are self-reported survey responses without clinical biomarkers; duplicate rows are inherent to the discrete feature space.
- **Heart Disease**: small dataset (918 rows after cleaning); the multiclass `num` target has very few class-3/4 cases, so per-class metrics for those classes are unreliable; final test accuracy is modest (0.4783); missingness is concentrated in `ca`, `thal`, and `slope`.
- **General**: single random seed (42); no external validation cohort; performance has not been calibrated across demographic subgroups.

---

## 16. Future Work

- Probability calibration and decision-threshold selection tuned to screening objectives.
- External validation on independent cohorts and subgroup fairness analysis.
- Additional engineered features and model families (e.g., calibrated ensembles).
- Backend API serving layer and integration tests against the documented contract.
- Monitoring, drift detection, and a retraining governance process.

---

*All metrics, model paths, and test counts in this document are taken from the generated reports and verified artifacts in this repository. No results have been invented or rounded misleadingly.*