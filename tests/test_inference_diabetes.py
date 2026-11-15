"""Unit tests for the CDC Diabetes inference system.

Author: Abhiram (Lead AI/ML Engineer)
Date: 8 October 2026
"""

import hashlib

import pytest
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

from src.inference.diabetes_risk_model import (
    load_diabetes_risk_pipeline,
    predict_diabetes_risk,
    get_pipeline_path,
)


DIABETES_VALID_SAMPLE = {
    "HighBP": 1,
    "HighChol": 0,
    "CholCheck": 1,
    "BMI": 28.5,
    "Smoker": 0,
    "Stroke": 0,
    "HeartDiseaseorAttack": 0,
    "PhysActivity": 1,
    "Fruits": 1,
    "Veggies": 1,
    "HvyAlcoholConsump": 0,
    "AnyHealthcare": 1,
    "NoDocbcCost": 0,
    "GenHlth": 2,
    "MentHlth": 0,
    "PhysHlth": 0,
    "DiffWalk": 0,
    "Sex": 1,
    "Age": 5,
    "Education": 4,
    "Income": 5,
}


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_saved_model_loads():
    """Verify the saved diabetes pipeline loads and has the expected structure."""
    pipeline = load_diabetes_risk_pipeline()
    assert isinstance(pipeline, Pipeline)
    assert list(pipeline.named_steps.keys()) == ["preprocess", "clf"]
    assert isinstance(pipeline.named_steps["clf"], HistGradientBoostingClassifier)


def test_valid_prediction():
    """Verify a valid single-record input produces a prediction."""
    result = predict_diabetes_risk(DIABETES_VALID_SAMPLE)
    assert result["predicted_class"] in ([0], [1])
    probability = result["diabetes_probability"][0]
    assert 0.0 <= probability <= 1.0


def test_output_structure():
    """Verify the prediction output structure for dict and DataFrame inputs."""
    result = predict_diabetes_risk(DIABETES_VALID_SAMPLE)
    assert set(result.keys()) == {"predicted_class", "diabetes_probability"}
    assert len(result["predicted_class"]) == 1
    assert len(result["diabetes_probability"]) == 1

    frame = pd.DataFrame([DIABETES_VALID_SAMPLE, DIABETES_VALID_SAMPLE])
    multi = predict_diabetes_risk(frame)
    assert len(multi["predicted_class"]) == 2
    assert len(multi["diabetes_probability"]) == 2


def test_missing_input_rejected():
    """Verify a missing required column raises ValueError."""
    incomplete = {key: value for key, value in DIABETES_VALID_SAMPLE.items() if key != "BMI"}
    with pytest.raises(ValueError):
        predict_diabetes_risk(incomplete)


def test_out_of_range_input_rejected():
    """Verify out-of-range feature values raise ValueError."""
    too_high_bmi = dict(DIABETES_VALID_SAMPLE, BMI=200)
    with pytest.raises(ValueError):
        predict_diabetes_risk(too_high_bmi)

    out_of_range_age = dict(DIABETES_VALID_SAMPLE, Age=99)
    with pytest.raises(ValueError):
        predict_diabetes_risk(out_of_range_age)


def test_non_integer_input_rejected():
    """Verify non-integer integer-coded fields raise ValueError."""
    fractional_age = dict(DIABETES_VALID_SAMPLE, Age=5.5)
    with pytest.raises(ValueError):
        predict_diabetes_risk(fractional_age)


def test_non_numeric_input_rejected():
    """Verify non-numeric values raise ValueError."""
    non_numeric = dict(DIABETES_VALID_SAMPLE, Age="not_a_number")
    with pytest.raises(ValueError):
        predict_diabetes_risk(non_numeric)


def test_inference_does_not_retrain(monkeypatch):
    """Verify inference never calls fit/fit_transform (no retraining)."""
    pipeline = load_diabetes_risk_pipeline()

    def _fail(*args, **kwargs):
        raise AssertionError("Inference must not retrain: fit was called")

    monkeypatch.setattr(Pipeline, "fit", _fail)
    monkeypatch.setattr(Pipeline, "fit_transform", _fail)
    monkeypatch.setattr(HistGradientBoostingClassifier, "fit", _fail)

    result = predict_diabetes_risk(DIABETES_VALID_SAMPLE, pipeline=pipeline)
    assert result["predicted_class"] in ([0], [1])


def test_inference_does_not_rewrite_model_file():
    """Verify running inference leaves the saved model artifact byte-identical."""
    path = get_pipeline_path()
    before = _sha256(path)
    predict_diabetes_risk(DIABETES_VALID_SAMPLE)
    after = _sha256(path)
    assert before == after


def test_model_loading_missing_file_raises_error(monkeypatch, tmp_path):
    """Verify load_diabetes_risk_pipeline raises FileNotFoundError when path is missing."""
    fake_path = tmp_path / "nonexistent_model.joblib"
    monkeypatch.setattr("src.inference.diabetes_risk_model.get_pipeline_path", lambda: fake_path)
    with pytest.raises(FileNotFoundError, match="Diabetes risk pipeline not found"):
        load_diabetes_risk_pipeline()


def test_model_loading_invalid_type_raises_error(monkeypatch, tmp_path):
    """Verify load_diabetes_risk_pipeline raises TypeError when artifact is not a Pipeline."""
    import joblib
    fake_path = tmp_path / "not_a_pipeline.joblib"
    joblib.dump({"not": "a pipeline"}, fake_path)
    monkeypatch.setattr("src.inference.diabetes_risk_model.get_pipeline_path", lambda: fake_path)
    with pytest.raises(TypeError, match="Expected sklearn Pipeline"):
        load_diabetes_risk_pipeline()


def test_output_json_serializability():
    """Verify prediction output is natively JSON serializable."""
    import json
    result = predict_diabetes_risk(DIABETES_VALID_SAMPLE)
    serialized = json.dumps(result)
    deserialized = json.loads(serialized)
    assert deserialized == result


def test_deterministic_regression():
    """Regression test ensuring fixed canonical input yields deterministic prediction."""
    result = predict_diabetes_risk(DIABETES_VALID_SAMPLE)
    assert result["predicted_class"] == [0]
    assert abs(result["diabetes_probability"][0] - 0.17852) < 1e-4


def test_extra_fields_ignored_and_order_invariance():
    """Verify extra fields are ignored and column order does not alter output."""
    baseline = predict_diabetes_risk(DIABETES_VALID_SAMPLE)

    # Shuffled key order + extra metadata fields
    shuffled = dict(reversed(list(DIABETES_VALID_SAMPLE.items())))
    shuffled["extra_patient_id"] = "PT-9941"
    shuffled["timestamp"] = "2026-11-15T12:00:00Z"

    result = predict_diabetes_risk(shuffled)
    assert result["predicted_class"] == baseline["predicted_class"]
    assert abs(result["diabetes_probability"][0] - baseline["diabetes_probability"][0]) < 1e-6

