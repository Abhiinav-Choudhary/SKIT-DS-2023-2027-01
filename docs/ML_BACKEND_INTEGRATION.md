# ML Backend Integration Guide & API Contracts

**Project**: SKIT-DS-2023-2027-01 (AI-Powered Health Risk Prediction and Monitoring System)  
**Target Audience**: Backend Engineer (Node.js/Express & Python services)  
**Author**: Lead AI/ML Engineer (Abhiram)  
**Status**: Production Ready  
**Date**: 15 November 2026  

---

## 1. Overview

The AI/ML module provides two trained, production-grade risk prediction pipelines:
1. **CDC Diabetes Risk Pipeline**: Binary risk classification and probability scoring.
2. **UCI Heart Disease Risk Pipeline**: Multiclass (grades 0–4) cardiovascular risk classification and per-grade probability distribution.

Both models are exposed through clean Python functions and a headless CLI runner (`src.inference.cli`) designed specifically for Node.js / Express `child_process.spawn()` or `execFile()`.

```
+-----------------------------------------------------------+
|              Backend (Node.js / Express.js)               |
+-----------------------------------------------------------+
              |                                  ^
       stdin JSON payload                 stdout JSON result
              v                                  |
+-----------------------------------------------------------+
|               Python Inference CLI Runner                 |
|                (src.inference.cli)                        |
+-----------------------------------------------------------+
         |                                           |
         v                                           v
+-----------------------------+     +-----------------------------+
|    predict_diabetes_risk    |     |    predict_heart_disease    |
+-----------------------------+     +-----------------------------+
         |                                           |
         v                                           v
+-----------------------------+     +-----------------------------+
| models/diabetes/            |     | models/heart/               |
| diabetes_risk_pipeline.joblib|     | heart_disease_risk_pipeline |
+-----------------------------+     +-----------------------------+
```

---

## 2. Python Environment & Dependencies

Ensure the machine or container running the ML inference has Python 3.10+ (tested on Python 3.13) with dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Core runtime dependencies:
- `scikit-learn>=1.6.0`
- `pandas>=2.2.0`
- `numpy>=2.0.0`
- `joblib>=1.4.0`

---

## 3. Diabetes Risk Prediction Contract

### 3.1 Model Artifact
- **Path**: `models/diabetes/diabetes_risk_pipeline.joblib`
- **Metadata**: `models/diabetes/metadata.json`
- **Type**: `sklearn.pipeline.Pipeline` (`preprocess` -> `clf`)
- **Classifier**: `HistGradientBoostingClassifier`

### 3.2 Loading Function
```python
from src.inference.diabetes_risk_model import load_diabetes_risk_pipeline

pipeline = load_diabetes_risk_pipeline()
```

### 3.3 Inference Signature
```python
from src.inference.diabetes_risk_model import predict_diabetes_risk

# Single record (dict) or batch (pandas.DataFrame)
result = predict_diabetes_risk(features: dict | pd.DataFrame, pipeline=None)
```

### 3.4 Required Input Schema (21 Features)

All 21 features must be present. Extra fields (e.g. `patient_id`) are safely ignored.

| Field Name | Type | Valid Range / Categories | Description |
|---|---|---|---|
| `HighBP` | Integer | `0` or `1` | High blood pressure (0 = no, 1 = yes) |
| `HighChol` | Integer | `0` or `1` | High cholesterol (0 = no, 1 = yes) |
| `CholCheck` | Integer | `0` or `1` | Cholesterol check in past 5 years (0 = no, 1 = yes) |
| `BMI` | Float/Int | `12.0` to `100.0` | Body Mass Index ($kg/m^2$) |
| `Smoker` | Integer | `0` or `1` | Smoked at least 100 cigarettes (0 = no, 1 = yes) |
| `Stroke` | Integer | `0` or `1` | History of stroke (0 = no, 1 = yes) |
| `HeartDiseaseorAttack` | Integer | `0` or `1` | History of CAD or MI (0 = no, 1 = yes) |
| `PhysActivity` | Integer | `0` or `1` | Physical activity in past 30 days (0 = no, 1 = yes) |
| `Fruits` | Integer | `0` or `1` | Consume fruit 1+ times per day (0 = no, 1 = yes) |
| `Veggies` | Integer | `0` or `1` | Consume vegetables 1+ times per day (0 = no, 1 = yes) |
| `HvyAlcoholConsump` | Integer | `0` or `1` | Heavy drinkers (males >14 drinks/week, females >7) |
| `AnyHealthcare` | Integer | `0` or `1` | Has any health coverage (0 = no, 1 = yes) |
| `NoDocbcCost` | Integer | `0` or `1` | Past 12 mo needed doctor but couldn't due to cost |
| `GenHlth` | Integer | `1` to `5` | General health rating (1=Excellent, 2=Very Good, 3=Good, 4=Fair, 5=Poor) |
| `MentHlth` | Integer | `0` to `30` | Days of poor mental health in past 30 days |
| `PhysHlth` | Integer | `0` to `30` | Days of poor physical health in past 30 days |
| `DiffWalk` | Integer | `0` or `1` | Serious difficulty walking or climbing stairs (0 = no, 1 = yes) |
| `Sex` | Integer | `0` or `1` | Gender (0 = female, 1 = male) |
| `Age` | Integer | `1` to `13` | 5-year age category (1: 18–24, 2: 25–29, ..., 9: 60–64, 13: 80+) |
| `Education` | Integer | `1` to `6` | Highest education level (1=Never attended, 6=College 4+ years) |
| `Income` | Integer | `1` to `8` | Income category (1: <\$10,000, 8: \$75,000+) |

### 3.5 Output Structure & Probability Interpretation
```json
{
  "predicted_class": [0],
  "diabetes_probability": [0.1785]
}
```

- `predicted_class`: `[0]` indicates low/non-diabetic risk; `[1]` indicates elevated diabetic risk.
- `diabetes_probability`: Calibrated continuous probability in `[0.0, 1.0]` of diabetes presence.
- **Frontend / Reporting Guidance**: Display the continuous probability as a percentage or risk band:
  - `< 0.30`: Low Risk
  - `0.30 - 0.60`: Moderate Risk (Screening Recommended)
  - `> 0.60`: High Risk (Clinical Follow-up Advised)

---

## 4. Heart Disease Risk Prediction Contract

### 4.1 Model Artifact
- **Path**: `models/heart/heart_disease_risk_pipeline.joblib`
- **Metadata**: `models/heart/metadata.json`
- **Type**: `sklearn.pipeline.Pipeline` (`features` -> `clf`)
- **Classifier**: Tuned `LogisticRegression(C=0.3, class_weight='balanced')`

### 4.2 Loading Function
```python
from src.inference.heart_disease_model import load_heart_disease_pipeline

pipeline = load_heart_disease_pipeline()
```

### 4.3 Inference Signature
```python
from src.inference.heart_disease_model import predict_heart_disease

result = predict_heart_disease(features: dict | pd.DataFrame, pipeline=None)
```

### 4.4 Required Input Schema (13 Features)

| Field Name | Type | Valid Range / Categories | Description |
|---|---|---|---|
| `age` | Float/Int | `1` to `120` | Patient age in years |
| `sex` | Float/Int | `0.0` or `1.0` | Sex (0 = female, 1 = male) |
| `cp` | Float/Int | `1.0`, `2.0`, `3.0`, `4.0` | Chest pain type (1 = typical angina, 2 = atypical angina, 3 = non-anginal pain, 4 = asymptomatic) |
| `trestbps` | Float/Int | `40.0` to `350.0` (non-zero) | Resting blood pressure ($mm\ Hg$) |
| `chol` | Float/Int | `40.0` to `1000.0` (non-zero)| Serum cholesterol ($mg/dl$) |
| `fbs` | Float/Int | `0.0` or `1.0` | Fasting blood sugar > 120 mg/dl (1 = true, 0 = false) |
| `restecg` | Float/Int | `0.0`, `1.0`, `2.0` | Resting ECG (0 = normal, 1 = ST-T wave abnormality, 2 = left ventricular hypertrophy) |
| `thalach` | Float/Int | `30.0` to `260.0` | Maximum heart rate achieved |
| `exang` | Float/Int | `0.0` or `1.0` | Exercise-induced angina (1 = yes, 0 = no) |
| `oldpeak` | Float/Int | `-10.0` to `15.0` | ST depression induced by exercise relative to rest |
| `slope` | Float/Int | `1.0`, `2.0`, `3.0` | Slope of peak exercise ST segment (1 = upsloping, 2 = flat, 3 = downsloping) |
| `ca` | Float/Int | `0.0`, `1.0`, `2.0`, `3.0` | Number of major vessels (0–3) colored by flourosopy |
| `thal` | Float/Int | `3.0`, `6.0`, `7.0` | Thallium heart scan (3 = normal, 6 = fixed defect, 7 = reversible defect) |

*Important*: `trestbps` and `chol` must be non-zero (zero is treated as missing data in clinical preprocessing and will be rejected).

### 4.5 Output Structure & Probability Interpretation
```json
{
  "predicted_class": [2],
  "class_labels": [0, 1, 2, 3, 4],
  "class_probabilities": [
    [0.0283, 0.0298, 0.5260, 0.2081, 0.2077]
  ]
}
```

#### Meaning of Predicted Classes (`num` 0–4):
- **Class 0**: No significant coronary artery disease (< 50% angiographic diameter stenosis).
- **Class 1**: Mild heart disease (Grade 1 stenosis).
- **Class 2**: Moderate heart disease (Grade 2 stenosis).
- **Class 3**: Severe heart disease (Grade 3 stenosis).
- **Class 4**: Critical / near-total occlusion (Grade 4 stenosis).

`class_probabilities`: An array of 5 floats corresponding to classes `[0, 1, 2, 3, 4]`. The values sum to `1.0`.

---

## 5. Node.js / Express Integration Guide

The Express backend can execute inference asynchronously using Node's `child_process.spawn`.

### 5.1 CLI Runner Usage
The CLI runner `src/inference/cli.py` reads JSON from stdin and emits formatted JSON on stdout.

Command pattern:
```bash
python -m src.inference.cli --model <diabetes|heart>
```

### 5.2 Node.js Helper Service (`mlService.js`)

Copy this utility directly into your Express backend:

```javascript
// mlService.js
const { spawn } = require('child_process');
const path = require('path');

const PROJECT_ROOT = path.resolve(__dirname, '../../'); // SKIT-DS-2023-2027-01 root
const PYTHON_EXEC = process.env.PYTHON_PATH || 'python';

/**
 * Predict risk using Python ML inference CLI.
 * @param {'diabetes' | 'heart'} modelType 
 * @param {Object} inputPayload 
 * @returns {Promise<Object>}
 */
function predictRisk(modelType, inputPayload) {
  return new Promise((resolve, reject) => {
    const py = spawn(PYTHON_EXEC, ['-m', 'src.inference.cli', '--model', modelType], {
      cwd: PROJECT_ROOT,
    });

    let stdoutData = '';
    let stderrData = '';

    py.stdout.on('data', (chunk) => {
      stdoutData += chunk.toString();
    });

    py.stderr.on('data', (chunk) => {
      stderrData += chunk.toString();
    });

    py.on('close', (code) => {
      if (code !== 0) {
        try {
          const parsedErr = JSON.parse(stderrData);
          return reject(new Error(parsedErr.message || 'ML Inference Error'));
        } catch {
          return reject(new Error(`ML Process exited with code ${code}: ${stderrData}`));
        }
      }

      try {
        const response = JSON.parse(stdoutData);
        resolve(response.data);
      } catch (err) {
        reject(new Error(`Failed to parse ML response: ${err.message}`));
      }
    });

    // Write input payload to stdin and close pipe
    py.stdin.write(JSON.stringify(inputPayload));
    py.stdin.end();
  });
}

module.exports = { predictRisk };
```

### 5.3 Express.js Route Handler Example

```javascript
// routes/prediction.js
const express = require('express');
const router = express.Router();
const { predictRisk } = require('../services/mlService');

// POST /api/predict/diabetes
router.post('/diabetes', async (req, res, next) => {
  try {
    const result = await predictRisk('diabetes', req.body);
    return res.status(200).json({
      success: true,
      condition: 'diabetes',
      prediction: result.predicted_class[0],
      risk_score: result.diabetes_probability[0],
      disclaimer: 'Risk estimate only; not a diagnostic evaluation.',
    });
  } catch (error) {
    return res.status(400).json({
      success: false,
      error: error.message,
    });
  }
});

// POST /api/predict/heart-disease
router.post('/heart-disease', async (req, res, next) => {
  try {
    const result = await predictRisk('heart', req.body);
    const predictedClass = result.predicted_class[0];
    const probabilities = result.class_probabilities[0];

    return res.status(200).json({
      success: true,
      condition: 'heart_disease',
      severity_class: predictedClass,
      class_labels: result.class_labels,
      probabilities: probabilities,
      no_disease_probability: probabilities[0],
      disease_presence_probability: 1.0 - probabilities[0],
      disclaimer: 'Risk estimate only; not a clinical diagnosis.',
    });
  } catch (error) {
    return res.status(400).json({
      success: false,
      error: error.message,
    });
  }
});

module.exports = router;
```

---

## 6. Error Handling and Status Codes

| Error Type | Exception Raised | HTTP Code | Cause / Remedy |
|---|---|---|---|
| Missing Columns | `ValueError("Missing required input columns: [...]")` | `400 Bad Request` | Ensure client payload includes all 21 (diabetes) or 13 (heart) required fields. |
| Non-numeric / NaN | `ValueError("Input columns must contain numeric, non-missing values: [...]")` | `400 Bad Request` | Client passed `null`, missing value, or string. Convert client types to numbers. |
| Out of Range | `ValueError("Invalid input values: [...] outside [min, max]")` | `400 Bad Request` | Value exceeds physiological or survey bounds (e.g. `BMI > 100`, `Age > 13`, `cp not in [1,2,3,4]`). |
| Zero Measurement | `ValueError("[trestbps/chol] is invalid because existing preprocessing treats zero as missing")` | `400 Bad Request` | Resting blood pressure or cholesterol was passed as 0. Must be clinical measurement. |
| Missing Model File | `FileNotFoundError` | `500 Internal Server Error` | Verify `.joblib` model file exists in `models/<condition>/`. |
| Corrupt Artifact | `TypeError` | `500 Internal Server Error` | Serialized model file is corrupt or wrong format. |

---

## 7. Direct Python Integration (Microservice / FastAPI alternative)

If backend opts to run a Python microservice, models can be called directly in-process:

```python
from src.inference.diabetes_risk_model import predict_diabetes_risk
from src.inference.heart_disease_model import predict_heart_disease

# Diabetes single record
diabetes_output = predict_diabetes_risk({
    "HighBP": 1, "HighChol": 1, "CholCheck": 1, "BMI": 29.0, "Smoker": 0,
    "Stroke": 0, "HeartDiseaseorAttack": 0, "PhysActivity": 1, "Fruits": 1,
    "Veggies": 1, "HvyAlcoholConsump": 0, "AnyHealthcare": 1, "NoDocbcCost": 0,
    "GenHlth": 3, "MentHlth": 0, "PhysHlth": 2, "DiffWalk": 0, "Sex": 1,
    "Age": 8, "Education": 5, "Income": 6
})

# Heart disease single record
heart_output = predict_heart_disease({
    "age": 58, "sex": 1, "cp": 4, "trestbps": 140, "chol": 260,
    "fbs": 0, "restecg": 1, "thalach": 125, "exang": 1, "oldpeak": 2.0,
    "slope": 2, "ca": 2, "thal": 7
})
```

---

## 8. Clinical Disclaimer Requirement

Any API response or frontend UI that presents results from these models **MUST** include the following disclaimer:

> **Medical Disclaimer**: This software produces risk-classification and probability estimates for research and preliminary screening purposes only. It is not clinically certified and must not be used as a medical diagnosis or as a substitute for professional clinical judgment.
