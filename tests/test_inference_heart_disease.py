"""Unit tests for the UCI Heart Disease inference system.

Author: Abhiram (Lead AI/ML Engineer)
Date: 8 October 2026
"""

import hashlib

import pytest
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.inference.heart_disease_model import (
    load_heart_disease_pipeline,
    predict_heart_disease,
    get_pipeline_path,
)


HEART_VALID_SAMPLE = {
    "age": 54,
    "sex": 1,
    "cp": 3,
    "trestbps": 130,
    "chol": 240,
    "fbs": 0,
    "restecg": 1,
    "thalach": 140,
    "exang": 0,
    "oldpeak": 1.0,
    "slope": 2,
    "ca": 1,
    "thal": 6,
}


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_production_model_loads_from_default_path():
    """Verify the production pipeline loads from its default path."""
    pipeline = load_heart_disease_pipeline()
    assert isinstance(pipeline, Pipeline)
    assert list(pipeline.named_steps.keys()) == ["features", "clf"]
    assert isinstance(pipeline.named_steps["clf"], LogisticRegression)
    assert [int(value) for value in pipeline.named_steps["clf"].classes_] == [0, 1, 2, 3, 4]


def test_valid_prediction():
    """Verify a valid single-record input produces a prediction."""
    result = predict_heart_disease(HEART_VALID_SAMPLE)
    assert result["predicted_class"][0] in [0, 1, 2, 3, 4]


def test_probability_output_valid():
    """Verify the probability output is well-formed and sums to 1."""
    result = predict_heart_disease(HEART_VALID_SAMPLE)
    assert result["class_labels"] == [0, 1, 2, 3, 4]
    probabilities = result["class_probabilities"]
    assert len(probabilities) == 1
    row = probabilities[0]
    assert len(row) == 5
    assert all(0.0 <= value <= 1.0 for value in row)
    assert abs(sum(row) - 1.0) < 1e-6


def test_output_structure():
    """Verify the prediction output structure for dict and DataFrame inputs."""
    result = predict_heart_disease(HEART_VALID_SAMPLE)
    assert set(result.keys()) == {"predicted_class", "class_labels", "class_probabilities"}
    assert len(result["predicted_class"]) == 1

    frame = pd.DataFrame([HEART_VALID_SAMPLE, HEART_VALID_SAMPLE])
    multi = predict_heart_disease(frame)
    assert len(multi["predicted_class"]) == 2
    assert len(multi["class_probabilities"]) == 2


def test_missing_input_rejected():
    """Verify a missing required column raises ValueError."""
    incomplete = {key: value for key, value in HEART_VALID_SAMPLE.items() if key != "thal"}
    with pytest.raises(ValueError):
        predict_heart_disease(incomplete)


def test_invalid_input_rejected():
    """Verify invalid categorical, zero-measurement, and non-numeric values raise ValueError."""
    invalid_category = dict(HEART_VALID_SAMPLE, cp=5)
    with pytest.raises(ValueError):
        predict_heart_disease(invalid_category)

    zero_measurement = dict(HEART_VALID_SAMPLE, trestbps=0)
    with pytest.raises(ValueError):
        predict_heart_disease(zero_measurement)

    non_numeric = dict(HEART_VALID_SAMPLE, age="not_a_number")
    with pytest.raises(ValueError):
        predict_heart_disease(non_numeric)


def test_inference_does_not_retrain(monkeypatch):
    """Verify inference never calls fit/fit_transform (no retraining)."""
    pipeline = load_heart_disease_pipeline()

    def _fail(*args, **kwargs):
        raise AssertionError("Inference must not retrain: fit was called")

    monkeypatch.setattr(Pipeline, "fit", _fail)
    monkeypatch.setattr(Pipeline, "fit_transform", _fail)
    monkeypatch.setattr(LogisticRegression, "fit", _fail)

    result = predict_heart_disease(HEART_VALID_SAMPLE, pipeline=pipeline)
    assert result["predicted_class"][0] in [0, 1, 2, 3, 4]


def test_inference_does_not_rewrite_model_file():
    """Verify running inference leaves the saved model artifact byte-identical."""
    path = get_pipeline_path()
    before = _sha256(path)
    predict_heart_disease(HEART_VALID_SAMPLE)
    after = _sha256(path)
    assert before == after


def test_model_loading_missing_file_raises_error(monkeypatch, tmp_path):
    """Verify load_heart_disease_pipeline raises FileNotFoundError when path is missing."""
    fake_path = tmp_path / "nonexistent_heart_model.joblib"
    monkeypatch.setattr("src.inference.heart_disease_model.get_pipeline_path", lambda: fake_path)
    with pytest.raises(FileNotFoundError, match="Heart Disease pipeline not found"):
        load_heart_disease_pipeline()


def test_model_loading_invalid_type_raises_error(monkeypatch, tmp_path):
    """Verify load_heart_disease_pipeline raises TypeError when artifact is not a Pipeline."""
    import joblib
    fake_path = tmp_path / "not_a_pipeline.joblib"
    joblib.dump({"not": "a pipeline"}, fake_path)
    monkeypatch.setattr("src.inference.heart_disease_model.get_pipeline_path", lambda: fake_path)
    with pytest.raises(TypeError, match="Expected sklearn Pipeline"):
        load_heart_disease_pipeline()


def test_out_of_range_numerical_input_rejected():
    """Verify physiologically out-of-range numerical values raise ValueError."""
    negative_age = dict(HEART_VALID_SAMPLE, age=-5)
    with pytest.raises(ValueError, match="outside \\[1, 120\\]"):
        predict_heart_disease(negative_age)

    extreme_trestbps = dict(HEART_VALID_SAMPLE, trestbps=450)
    with pytest.raises(ValueError, match="outside \\[40, 350\\]"):
        predict_heart_disease(extreme_trestbps)

    negative_thalach = dict(HEART_VALID_SAMPLE, thalach=-10)
    with pytest.raises(ValueError, match="outside \\[30, 260\\]"):
        predict_heart_disease(negative_thalach)


def test_non_finite_numerical_input_rejected():
    """Verify infinite numeric values raise ValueError."""
    inf_chol = dict(HEART_VALID_SAMPLE, chol=float("inf"))
    with pytest.raises(ValueError):
        predict_heart_disease(inf_chol)


def test_output_json_serializability():
    """Verify prediction output is natively JSON serializable."""
    import json
    result = predict_heart_disease(HEART_VALID_SAMPLE)
    serialized = json.dumps(result)
    deserialized = json.loads(serialized)
    assert deserialized == result


def test_deterministic_regression():
    """Regression test ensuring fixed canonical input yields deterministic multiclass prediction."""
    result = predict_heart_disease(HEART_VALID_SAMPLE)
    assert result["predicted_class"] == [2]
    assert result["class_labels"] == [0, 1, 2, 3, 4]
    probs = result["class_probabilities"][0]
    assert abs(probs[2] - 0.5260) < 1e-3
    assert abs(sum(probs) - 1.0) < 1e-6


def test_multiclass_probabilities_batch_consistency():
    """Verify batch predictions yield (N, 5) probabilities each summing to 1.0."""
    sample2 = dict(HEART_VALID_SAMPLE, age=65, cp=4, thalach=110)
    batch = pd.DataFrame([HEART_VALID_SAMPLE, sample2])
    result = predict_heart_disease(batch)

    assert len(result["predicted_class"]) == 2
    assert len(result["class_probabilities"]) == 2
    for prob_row in result["class_probabilities"]:
        assert len(prob_row) == 5
        assert all(0.0 <= p <= 1.0 for p in prob_row)
        assert abs(sum(prob_row) - 1.0) < 1e-6


def test_extra_fields_ignored_and_order_invariance():
    """Verify extra metadata fields are ignored and column order does not alter output."""
    baseline = predict_heart_disease(HEART_VALID_SAMPLE)

    shuffled = dict(reversed(list(HEART_VALID_SAMPLE.items())))
    shuffled["patient_mrn"] = "MRN-100234"
    shuffled["encounter_date"] = "2026-11-15"

    result = predict_heart_disease(shuffled)
    assert result["predicted_class"] == baseline["predicted_class"]
    for p_res, p_base in zip(result["class_probabilities"][0], baseline["class_probabilities"][0]):
        assert abs(p_res - p_base) < 1e-6

