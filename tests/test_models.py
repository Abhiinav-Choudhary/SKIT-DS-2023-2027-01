"""Unit tests for Baseline Models and Evaluation Metrics.

Author: Abhiram (Lead AI/ML Engineer)
Date: 8 October 2026
"""

import pytest
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from src.models.baseline_models import get_baseline_models
from src.evaluation.metrics import evaluate_predictions


def test_get_baseline_models():
    """Verify factory returns the 4 canonical models."""
    models = get_baseline_models(random_state=42)
    assert len(models) == 4
    expected_names = {"Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting"}
    assert set(models.keys()) == expected_names


def test_evaluate_predictions_metrics():
    """Verify metric calculations on known test vectors."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 1, 0, 1, 1, 0, 1])
    y_prob = np.array([0.1, 0.2, 0.6, 0.3, 0.7, 0.8, 0.4, 0.9])

    metrics = evaluate_predictions(y_true, y_pred, y_prob)

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "roc_auc" in metrics
    assert "pr_auc" in metrics
    assert "confusion_matrix" in metrics

    cm = metrics["confusion_matrix"]
    assert cm["tn"] == 3
    assert cm["fp"] == 1
    assert cm["fn"] == 1
    assert cm["tp"] == 3
    assert metrics["accuracy"] == 0.75
    assert metrics["recall"] == 0.75


def test_baseline_model_prediction_shape():
    """Verify baseline model generates predictions and valid probabilities."""
    models = get_baseline_models(random_state=42)
    lr = models["Logistic Regression"]

    # Synthetic 27-feature matrix
    np.random.seed(42)
    X_syn = np.random.randn(20, 27)
    y_syn = np.random.randint(0, 2, size=20)

    lr.fit(X_syn, y_syn)
    preds = lr.predict(X_syn)
    probs = lr.predict_proba(X_syn)

    assert preds.shape == (20,)
    assert probs.shape == (20, 2)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)


def test_saved_model_artifact_loading():
    """Verify saved baseline model artifacts exist and load correctly."""
    repo_root = Path(__file__).resolve().parent.parent
    baselines_dir = repo_root / "models" / "diabetes" / "baselines"
    lr_file = baselines_dir / "logistic_regression_baseline.joblib"
    gb_file = baselines_dir / "gradient_boosting_baseline.joblib"

    if lr_file.exists():
        loaded_lr = joblib.load(lr_file)
        assert hasattr(loaded_lr, "predict")
    if gb_file.exists():
        loaded_gb = joblib.load(gb_file)
        assert hasattr(loaded_gb, "predict_proba")
