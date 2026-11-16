"""AI ML Inference Bridge for Backend API.

Integrates Abhiram's trained production models (Diabetes & Heart Disease)
with the Node.js Express backend.
Does NOT train or modify ML models.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Add repository root to Python path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Scikit-learn compatibility shims for environment consistency
try:
    import sklearn.compose._column_transformer as ct
    if not hasattr(ct, '_RemainderColsList'):
        ct._RemainderColsList = type('_RemainderColsList', (list,), {})
    import sklearn._loss
    sys.modules['_loss'] = sklearn._loss
    if not hasattr(sklearn._loss, 'CyHalfBinomialLoss'):
        sklearn._loss.CyHalfBinomialLoss = getattr(sklearn._loss, 'HalfBinomialLoss', None)
except Exception:
    pass

from src.inference.diabetes_risk_model import (
    predict_diabetes_risk,
    load_diabetes_risk_pipeline,
    MODEL_NAME as DIABETES_MODEL_NAME,
    MODEL_VERSION as DIABETES_MODEL_VERSION,
)
from src.inference.heart_disease_model import (
    predict_heart_disease,
    load_heart_disease_pipeline,
    MODEL_NAME as HEART_MODEL_NAME,
    MODEL_VERSION as HEART_MODEL_VERSION,
)


def _patch_heart_imputer(pipeline):
    """Ensure SimpleImputer compatibility across scikit-learn versions."""
    for step in pipeline.named_steps.values():
        if hasattr(step, "named_steps"):
            for s in step.named_steps.values():
                if hasattr(s, "named_transformers_"):
                    for t in s.named_transformers_.values():
                        if hasattr(t, "named_steps"):
                            for sub in t.named_steps.values():
                                if hasattr(sub, "_fit_dtype") and not hasattr(sub, "_fill_dtype"):
                                    sub._fill_dtype = sub._fit_dtype


def run_diabetes_prediction(input_data: dict) -> dict:
    pipeline = load_diabetes_risk_pipeline()
    raw_result = predict_diabetes_risk(input_data, pipeline=pipeline)

    predicted_class = int(raw_result["predicted_class"][0])
    probability = float(raw_result["diabetes_probability"][0])
    risk_score = round(probability * 100, 1)

    if risk_score >= 70:
        risk_level = "Critical"
        recommendation = "High probability of Type 2 Diabetes or glucose intolerance. Immediate clinical evaluation and OGTT/HbA1c testing recommended."
    elif risk_score >= 45:
        risk_level = "High"
        recommendation = "Elevated diabetes risk detected. Clinical fasting blood glucose test and lifestyle modifications strongly advised."
    elif risk_score >= 20:
        risk_level = "Moderate"
        recommendation = "Moderate risk indicators observed. Adopt balanced nutrition, regular physical exercise, and annual screening."
    else:
        risk_level = "Low"
        recommendation = "Metabolic indicators within normal healthy range. Continue balanced diet and active lifestyle."

    return {
        "modelType": "diabetes",
        "predictedClass": predicted_class,
        "riskScore": risk_score,
        "riskLevel": risk_level,
        "probability": probability,
        "recommendation": recommendation,
        "modelName": DIABETES_MODEL_NAME,
        "modelVersion": DIABETES_MODEL_VERSION,
    }


def run_heart_prediction(input_data: dict) -> dict:
    pipeline = load_heart_disease_pipeline()
    _patch_heart_imputer(pipeline)
    raw_result = predict_heart_disease(input_data, pipeline=pipeline)

    predicted_class = int(raw_result["predicted_class"][0])
    probabilities = raw_result.get("class_probabilities", [[]])[0]
    probabilities = [float(p) for p in probabilities]

    # Non-zero target classes (1, 2, 3, 4) indicate varying stages of cardiovascular disease
    disease_prob = sum(probabilities[1:]) if len(probabilities) > 1 else (1.0 if predicted_class > 0 else 0.0)
    risk_score = round(disease_prob * 100, 1)

    severity_levels = {
        0: ("Low", "No significant coronary artery disease indicators detected. Maintain cardiovascular exercise and heart-healthy diet."),
        1: ("Mild", "Mild coronary artery disease risk detected. Recommend monitoring blood pressure, lipid profile, and regular check-ups."),
        2: ("Moderate", "Moderate cardiovascular disease indicators present. Consult a cardiologist for stress ECG or echocardiogram."),
        3: ("High", "Significant coronary artery narrowing indicated. Cardiology consultation and diagnostic angiography advised."),
        4: ("Critical", "Severe cardiovascular risk markers present. Urgent medical consultation and comprehensive cardiovascular assessment required."),
    }

    risk_level, recommendation = severity_levels.get(
        predicted_class,
        ("Elevated", "Cardiovascular risk markers detected. Consult healthcare professional.")
    )

    return {
        "modelType": "heart",
        "predictedClass": predicted_class,
        "riskScore": risk_score,
        "riskLevel": risk_level,
        "probabilities": probabilities,
        "recommendation": recommendation,
        "modelName": HEART_MODEL_NAME,
        "modelVersion": HEART_MODEL_VERSION,
    }


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"success": False, "error": "Model type argument required ('diabetes' or 'heart')"}))
        sys.exit(1)

    model_type = sys.argv[1].strip().lower()

    try:
        raw_input = sys.stdin.read()
        if not raw_input.strip():
            print(json.dumps({"success": False, "error": "No input JSON received on stdin"}))
            sys.exit(1)

        input_data = json.loads(raw_input)

        if model_type == "diabetes":
            result = run_diabetes_prediction(input_data)
        elif model_type in ("heart", "heart_disease"):
            result = run_heart_prediction(input_data)
        else:
            print(json.dumps({"success": False, "error": f"Unsupported model type: {model_type}"}))
            sys.exit(1)

        print(json.dumps({"success": True, "data": result}))
        sys.exit(0)

    except ValueError as ve:
        print(json.dumps({"success": False, "error": str(ve)}))
        sys.exit(1)
    except Exception as ex:
        print(json.dumps({"success": False, "error": f"Inference execution failed: {str(ex)}"}))
        sys.exit(1)


if __name__ == "__main__":
    main()
