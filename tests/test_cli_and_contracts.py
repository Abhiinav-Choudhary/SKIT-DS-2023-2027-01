"""Integration and CLI contract tests for backend serving.

Verifies that the CLI runner and inference interfaces fulfill contracts
required by Node.js / Express integration.

Author: Lead AI/ML Engineer
Date: 15 November 2026
"""

import json
import subprocess
import sys
import pandas as pd
import pytest

from tests.test_inference_diabetes import DIABETES_VALID_SAMPLE
from tests.test_inference_heart_disease import HEART_VALID_SAMPLE
from src.inference.diabetes_risk_model import predict_diabetes_risk
from src.inference.heart_disease_model import predict_heart_disease


def test_cli_diabetes_input_flag():
    """Verify CLI executes diabetes prediction via --input flag."""
    cmd = [
        sys.executable,
        "-m",
        "src.inference.cli",
        "--model",
        "diabetes",
        "--input",
        json.dumps(DIABETES_VALID_SAMPLE),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"CLI stderr: {res.stderr}"

    data = json.loads(res.stdout)
    assert data["status"] == "success"
    assert data["model"] == "diabetes"
    assert "predicted_class" in data["data"]
    assert "diabetes_probability" in data["data"]


def test_cli_heart_input_flag():
    """Verify CLI executes heart disease prediction via --input flag."""
    cmd = [
        sys.executable,
        "-m",
        "src.inference.cli",
        "--model",
        "heart",
        "--input",
        json.dumps(HEART_VALID_SAMPLE),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"CLI stderr: {res.stderr}"

    data = json.loads(res.stdout)
    assert data["status"] == "success"
    assert data["model"] == "heart"
    assert "predicted_class" in data["data"]
    assert "class_labels" in data["data"]
    assert "class_probabilities" in data["data"]


def test_cli_stdin_pipe():
    """Verify CLI executes prediction by reading JSON payload from stdin."""
    cmd = [
        sys.executable,
        "-m",
        "src.inference.cli",
        "--model",
        "diabetes",
    ]
    res = subprocess.run(
        cmd,
        input=json.dumps(DIABETES_VALID_SAMPLE),
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    data = json.loads(res.stdout)
    assert data["status"] == "success"
    assert data["data"]["predicted_class"] in ([0], [1])


def test_cli_invalid_json_handling():
    """Verify CLI gracefully rejects unparseable JSON on stdin with non-zero exit code."""
    cmd = [
        sys.executable,
        "-m",
        "src.inference.cli",
        "--model",
        "diabetes",
        "--input",
        "{invalid_json: 123",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 1
    err = json.loads(res.stderr)
    assert err["status"] == "error"
    assert "JSON" in err["message"]


def test_cli_validation_error_handling():
    """Verify CLI catches model validation errors and formats JSON error response."""
    bad_payload = dict(DIABETES_VALID_SAMPLE, BMI=500)
    cmd = [
        sys.executable,
        "-m",
        "src.inference.cli",
        "--model",
        "diabetes",
        "--input",
        json.dumps(bad_payload),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 1
    err = json.loads(res.stderr)
    assert err["status"] == "error"
    assert "BMI" in err["message"]


def test_dict_and_dataframe_equivalence():
    """Verify that calling predict with single dict or single-row DataFrame gives identical results."""
    # Diabetes
    diab_dict_res = predict_diabetes_risk(DIABETES_VALID_SAMPLE)
    diab_df_res = predict_diabetes_risk(pd.DataFrame([DIABETES_VALID_SAMPLE]))
    assert diab_dict_res == diab_df_res

    # Heart
    heart_dict_res = predict_heart_disease(HEART_VALID_SAMPLE)
    heart_df_res = predict_heart_disease(pd.DataFrame([HEART_VALID_SAMPLE]))
    assert heart_dict_res == heart_df_res
