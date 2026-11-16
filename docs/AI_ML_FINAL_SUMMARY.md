# AI/ML Subsystem Final Technical Summary & Academic Defense Report

**Project**: SKIT-DS-2023-2027-01 (AI-Powered Health Risk Prediction and Monitoring System)  
**Role**: Lead AI/ML Engineer (Abhiram)  
**Academic Year**: Final Year B.Tech (2026–2027)  
**Milestone Completion Date**: 15 November 2026  
**Status**: Complete, Validated, Tested, and Integration-Ready  

---

> ### Medical & Academic Disclaimer
> This project contains **machine learning risk-estimation and probability-scoring components only**. It is an academic engineering research prototype developed for final-year engineering evaluation. It is **not** a certified medical diagnostic device, has not been clinically evaluated by healthcare regulatory agencies (such as FDA or CDSCO), and must **never** be used as a standalone diagnostic tool or as a substitute for professional clinical judgment, diagnosis, or patient management.

---

## 1. Project Overview & Scope

The AI/ML module of project `SKIT-DS-2023-2027-01` provides two trained, validated, serialized, and production-ready machine learning pipelines to quantify disease risk:
1. **CDC Diabetes Risk Pipeline**: Classifies chronic diabetes risk and estimates probability using demographic, lifestyle, and behavioral survey indicators.
2. **UCI Heart Disease Risk Pipeline**: Multi-center cardiovascular risk classifier predicting coronary artery disease presence and severity grades (multiclass `0–4`).

In this collaborative four-member engineering project, the AI/ML engineer is exclusively responsible for data ingestion, cleaning, exploratory data analysis, feature engineering, model architecture development, hyperparameter optimization, model serialization, production inference pipelines, input validation, backend integration interfaces, automated testing, and academic documentation.

---

## 2. Dataset Descriptions & Quality Audits

### 2.1 CDC Diabetes Health Indicators (BRFSS 2015)
- **Source**: UCI Machine Learning Repository (Dataset #891), derived from the CDC Behavioral Risk Factor Surveillance System (BRFSS) 2015 annual survey.
- **Dimensionality**: 253,680 records $\times$ 22 columns (21 predictor variables + 1 binary target).
- **Target Variable**: `Diabetes_binary`
  - Class `0` (No diabetes / prediabetes): 218,334 records (**86.07%**)
  - Class `1` (Diagnosed diabetes): 35,346 records (**13.93%**)
  - Severe class imbalance ratio: **6.18 : 1**.
- **Data Quality & Integrity**:
  - Null / Missing cells: Exactly **0** missing cells.
  - Survey duplicate profiles: 24,206 exact duplicate rows (9.54%). In discrete, binned demographic surveys without individual identifiers, identical survey profiles are statistically expected.
  - Conflicting labels: Out of 227,908 unique feature profiles, 1,566 (0.69%) have discordant diabetes labels, representing intrinsic stochasticity in self-reported survey data.
- **Stratified Partitioning**: 70% Train (160,631), 15% Validation (34,421), 15% Test (34,422). Class distribution was strictly preserved across all splits.

### 2.2 UCI Heart Disease Aggregated Cohort
- **Source**: UCI Machine Learning Repository (Dataset #45), combining four clinical recording facilities:
  - Cleveland Clinic Foundation (303 records)
  - Hungarian Institute of Cardiology, Budapest (294 records)
  - V.A. Medical Center, Long Beach, CA (200 records)
  - University Hospital, Zurich, Switzerland (123 records)
- **Dimensionality**: 920 raw records $\times$ 15 columns (13 clinical predictors + `source` center + `num` target).
- **Target Variable**: `num` (Multiclass coronary artery disease severity grades `0` to `4`):
  - Class `0` (<50% diameter stenosis / No significant CAD): 410 records (**44.66%**)
  - Class `1` (Mild stenosis): 265 records (**28.87%**)
  - Class `2` (Moderate stenosis): 108 records (**11.76%**)
  - Class `3` (Severe stenosis): 107 records (**11.66%**)
  - Class `4` (Critical / total occlusion): 28 records (**3.05%**)
- **Cleaning & Missingness Handling**:
  - Dropped 2 exact duplicate records $\rightarrow$ **918 clean records**.
  - Identified physiological zero-measurements: `trestbps == 0` (1 record) and `chol == 0` (172 records, concentrated in the Hungarian/Swiss cohorts). In clinical medicine, resting blood pressure and serum cholesterol cannot be 0; hence, zeroes were treated as true missing values.
  - Total missing cells: 1,925 cells, concentrated in invasive fluoroscopy/angiography features (`ca`, `thal`, `slope`).
- **Stratified Partitioning**: Train (642), Validation (138), Test (138) stratified by multiclass target `num`.

---

## 3. Data Preprocessing & Leakage-Free Feature Engineering

### 3.1 Preprocessing Architectures
All transformations are implemented via Scikit-learn `Pipeline` and `ColumnTransformer` constructs. Imputers and scalers fit strictly on training splits during cross-validation, preventing data leakage.

### 3.2 Diabetes Feature Engineering (`DiabetesFeatureEngineer`)
The transformer computes 6 clinically grounded composite indicators, expanding the 21 raw survey features into 27 transformed features:
1. `BMI_Category`: Ordinal BMI binning (1: Underweight <18.5, 2: Normal 18.5–24.9, 3: Overweight 25.0–29.9, 4: Obese $\ge 30.0$).
2. `CardioMetabolic_Risk_Score`: Additive index of vascular comorbidities ($\text{HighBP} + \text{HighChol} + \text{Stroke} + \text{HeartDiseaseorAttack}$).
3. `HighBP_x_HighChol`: Interaction term capturing simultaneous hypertension and hyperlipidemia.
4. `Healthy_Lifestyle_Score`: Cumulative positive lifestyle behaviors ($\text{PhysActivity} + \text{Fruits} + \text{Veggies} + (1 - \text{Smoker}) + (1 - \text{HvyAlcoholConsump})$).
5. `Age_x_GenHlth`: Interaction term capturing the compounded vulnerability of advanced age and subjective poor health.
6. `Sedentary_Obese`: Binary flag identifying individuals with $\text{BMI} \ge 30$ and zero physical activity ($\text{PhysActivity} == 0$).

Continuous features (`BMI`, `MentHlth`, `PhysHlth`, `Age_x_GenHlth`) are scaled with `StandardScaler`. Categorical/binary features pass through untouched.

### 3.3 Heart Disease Feature Engineering (`HeartDiseaseFeatureEngineer`)
The transformer generates 11 domain features, combined with median imputation for numeric features and most-frequent imputation + one-hot encoding for categorical features, yielding 46 transformed features:
1. `age_x_oldpeak`: Cross-product of patient age and exercise ST depression.
2. `resting_bp_cholesterol_index`: Composite vascular hemodynamic burden index ($\text{trestbps} \times \text{chol} / 1000$).
3. `heart_rate_reserve_proxy`: Age-predicted maximum heart rate reserve ($220 - \text{age} - \text{thalach}$).
4. `oldpeak_per_age`: Relative ST depression per year of life ($\text{oldpeak} / \text{age}$).
5. Clinical indicator flags: `asymptomatic_chest_pain` ($\text{cp} == 4$), `exercise_ischemia_signal` ($\text{exang} == 1 \land \text{oldpeak} > 1.0$), `has_major_vessels_colored` ($\text{ca} > 0$), `thal_reversible_defect` ($\text{thal} == 7$), `high_resting_bp` ($\text{trestbps} \ge 140$), `high_cholesterol` ($\text{chol} \ge 240$), `abnormal_restecg` ($\text{restecg} > 0$).

---

## 4. Model Selection, Tuning & Evaluation

### 4.1 Diabetes Model Development & Champion Selection
Four baseline algorithms were benchmarked using 5-fold Stratified Cross-Validation on the training set:
- **Logistic Regression (balanced)**: CV ROC-AUC $0.8070 \pm 0.0028$, Recall $0.7615$
- **Decision Tree (balanced)**: CV ROC-AUC $0.7901 \pm 0.0035$, Recall $0.7561$
- **Random Forest (balanced)**: CV ROC-AUC $0.8108 \pm 0.0030$, Recall $0.7145$
- **HistGradientBoostingClassifier (balanced)**: CV ROC-AUC $0.8134 \pm 0.0024$, Recall $0.7743$

#### Hyperparameter Optimization
`HistGradientBoostingClassifier` and `RandomForestClassifier` underwent extensive hyperparameter tuning using `RandomizedSearchCV` (5-fold Stratified K-Fold, scoring on ROC-AUC).

Selected Champion: `HistGradientBoostingClassifier`
- `min_samples_leaf`: 75
- `max_iter`: 125
- `learning_rate`: 0.08
- `l2_regularization`: 10.0
- `class_weight`: balanced
- Best CV ROC-AUC: **0.8139**

#### Final Test Set Evaluation (N=34,422 holdout records)
| Metric | Value | Technical Interpretation |
|---|---|---|
| **ROC-AUC** | **0.8217** | Strong discrimination ability across risk thresholds. |
| **PR-AUC** | **0.4522** | Over $3.2\times$ above the baseline prevalence ($0.1393$). |
| **Recall (Sensitivity)** | **0.7941** | Detects ~80% of true diabetic cases, prioritizing screening safety. |
| **Specificity** | **0.6967** | Successfully eliminates ~70% of false non-cases. |
| **Accuracy** | **0.7116** | Balanced across classes. |
| **Precision** | **0.3210** | Expected trade-off from balanced class weighting under 6.18:1 imbalance. |
| **F1-Score** | **0.4572** | Harmonic mean reflecting high recall and moderate precision. |

Confusion Matrix:
- True Negatives (TN): 20,314
- False Positives (FP): 8,843
- False Negatives (FN): 1,084
- True Positives (TP): 4,181

---

### 4.2 Heart Disease Model Development & Champion Selection
Four baseline algorithms were evaluated on multiclass `num` targets:
- **Logistic Regression (balanced)**: CV Weighted F1 $0.5489$, Macro ROC-AUC $0.7592$
- **Decision Tree (balanced)**: CV Weighted F1 $0.4357$, Macro ROC-AUC $0.6272$
- **Random Forest (balanced)**: CV Weighted F1 $0.5369$, Macro ROC-AUC $0.7486$
- **HistGradientBoostingClassifier**: CV Weighted F1 $0.5284$, Macro ROC-AUC $0.7397$

#### Hyperparameter Optimization
`LogisticRegression` and `HistGradientBoostingClassifier` were tuned via `GridSearchCV` (5-fold Stratified K-Fold, refit metric `f1_weighted`).

Selected Champion: Tuned `LogisticRegression`
- `C`: 0.3 (L2 penalty)
- `class_weight`: balanced
- `solver`: lbfgs
- `max_iter`: 3000
- Best CV Weighted F1: **0.5671**

#### Final Test Set Evaluation (N=138 holdout records)
| Metric | Value | Technical Interpretation |
|---|---|---|
| **Accuracy** | **0.4783** | Multiclass 5-way classification on small clinical cohort. |
| **Macro Precision** | **0.3615** | Unweighted average precision across all 5 severity classes. |
| **Weighted Precision** | **0.5608** | Prevalence-weighted precision. |
| **Macro Recall** | **0.3980** | Unweighted average recall across all 5 severity classes. |
| **Weighted Recall** | **0.4783** | Prevalence-weighted recall. |
| **Macro F1-Score** | **0.3528** | Average F1 across 5 classes. |
| **Weighted F1-Score** | **0.5104** | Weighted harmonic balance across severity tiers. |
| **Macro OvR ROC-AUC** | **0.7550** | Average One-vs-Rest AUC across classes. |
| **Weighted OvR ROC-AUC**| **0.7816** | Weighted One-vs-Rest discrimination ability. |

Per-Class Performance on Test Set:
- **Class 0 (No Disease)**: Precision $0.8400$, Recall $0.6774$, F1 $0.7500$ (Support: 62)
- **Class 1 (Mild)**: Precision $0.4688$, Recall $0.3750$, F1 $0.4167$ (Support: 40)
- **Class 2 (Moderate)**: Precision $0.3333$, Recall $0.3750$, F1 $0.3529$ (Support: 16)
- **Class 3 (Severe)**: Precision $0.0476$, Recall $0.0625$, F1 $0.0541$ (Support: 16)
- **Class 4 (Critical)**: Precision $0.1176$, Recall $0.5000$, F1 $0.1905$ (Support: 4)

Multiclass Confusion Matrix:
$$\begin{bmatrix}
42 & 10 & 1 & 5 & 4 \\
8 & 15 & 3 & 9 & 5 \\
0 & 2 & 6 & 5 & 3 \\
0 & 5 & 7 & 1 & 3 \\
0 & 0 & 1 & 1 & 2
\end{bmatrix}$$

---

## 5. Production Inference & Integration Architecture

The inference system was architected to be modular, efficient, deterministic, and strictly decoupled from training code.

### 5.1 Artifact Storage & Loading
- Serialized pipelines reside in `models/diabetes/diabetes_risk_pipeline.joblib` and `models/heart/heart_disease_risk_pipeline.joblib`.
- Metadata files (`metadata.json`) provide machine-readable provenance, hyperparameter records, and evaluation metrics.
- Model loading is performed via Scikit-learn `Pipeline` loaders with full type-checking and path verification.
- Inference functions are verified to be strictly read-only: `fit` or `fit_transform` calls are never invoked, and model binaries remain byte-identical before and after prediction.

### 5.2 Input Validation Layer
Each inference module executes comprehensive input validation prior to model ingestion:
- **Diabetes**: Enforces exact 21 raw survey features, numeric non-missing values, discrete integer coding for all variables except `BMI`, and physiological/survey domain bounds.
- **Heart Disease**: Enforces exact 13 raw clinical features, numeric non-missing and finite values, rejection of zero measurements for blood pressure and cholesterol, categorical adherence to training categories, and physiological bounds (`age` 1–120, `trestbps` 40–350, `chol` 40–1000, `thalach` 30–260, `oldpeak` -10.0–15.0).

### 5.3 Headless CLI & Subprocess Bridge
To support the project's Express.js (Node.js) backend, a dedicated headless CLI runner `src/inference/cli.py` is provided:
- Reads payloads from JSON strings, files, or standard input streams (`stdin`).
- Emits structured JSON results on standard output (`stdout`).
- Captures validation errors and emits structured JSON error payloads on standard error (`stderr`) with exit code 1.

---

## 6. Testing Methodology & Verification Results

A 53-test automated test suite was constructed and executed using Pytest:
- **Data Loaders**: 4 tests verifying shapes, columns, 3-class/binary variants, and zero nulls.
- **Preprocessing**: 4 tests verifying quality audits, deduplication detection, stratified splitting, and transformer behavior.
- **Feature Engineering**: 5 tests verifying output dimensionality, categorical binning, risk scoring, row independence, and pipeline persistence.
- **Baseline Models**: 4 tests verifying baseline factories, metric evaluation formulas, output shapes, and baseline persistence.
- **Diabetes Inference**: 14 tests verifying artifact loading, valid predictions, output structure, missing columns rejection, out-of-range rejection, non-integer rejection, non-numeric rejection, read-only inference, checksum immutability, `FileNotFoundError` on missing artifact, `TypeError` on invalid artifact, JSON serializability, deterministic regression, and extra field invariance.
- **Heart Disease Inference**: 16 tests verifying artifact loading, valid predictions, 5-class probability normalization, output structure, missing inputs rejection, invalid categories rejection, read-only inference, checksum immutability, `FileNotFoundError`, `TypeError`, physiological out-of-range rejection, non-finite rejection, JSON serializability, deterministic regression, batch probability consistency, and extra field invariance.
- **CLI & Contract Verification**: 6 tests verifying CLI flags, stdin piping, invalid JSON handling, validation error formatting, and dict vs. DataFrame equivalence.

All 53 tests passed deterministically.

---

## 7. Model Limitations & Ethical Considerations

1. **Academic Prototype Only**: The system is designed for engineering research and demonstrative evaluation; it has not undergone clinical trials.
2. **Self-Reported Survey Data (Diabetes)**: The CDC BRFSS dataset relies on phone interview responses without laboratory glucose or HbA1c verification, introducing recall bias and subjective health variance.
3. **Severe Class Imbalance**: With 13.93% diabetic prevalence, optimizing for high screening sensitivity (Recall ~79.4%) inherently produces lower precision (~32.1%), leading to false positives that require confirmation by clinical testing.
4. **Small Sample Size for Advanced Heart Disease (Classes 3 & 4)**: The multiclass heart disease dataset has only 28 total class-4 cases (4 in the test set). Performance metrics for severe grades carry wide confidence intervals.
5. **No External Demographic Generalization**: Models were trained on specific US cohorts (BRFSS 2015 and 4-center clinical sites); performance may shift when deployed on disparate international patient populations.

---

## 8. Future Improvements

1. **Dynamic Decision Threshold Calibration**: Allow clinical users to tune probability decision thresholds depending on screening cost vs. false negative risk tolerance.
2. **Clinical Biomarker Integration**: Incorporate continuous laboratory biomarkers (HbA1c, fasting glucose, troponin, LDL/HDL subfractions).
3. **Model Explainability (XAI)**: Implement TreeSHAP and KernelSHAP explanation modules to visualize individualized patient risk factor contributions.
4. **Automated Drift Monitoring**: Deploy continuous monitoring for covariate shift and concept drift in production health indicator inputs.
