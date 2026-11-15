"""Unit tests for Phase 7 Feature Engineering Module.

Author: Abhiram (Lead AI/ML Engineer)
Date: 8 October 2026
"""

import pytest
import pandas as pd
import numpy as np
import joblib

from src.features.feature_engineer import DiabetesFeatureEngineer, ENGINEERED_FEATURE_NAMES
from src.preprocessing.pipeline import create_full_pipeline


@pytest.fixture
def sample_survey_data():
    """Create synthetic test DataFrame matching the 21 CDC features."""
    return pd.DataFrame([
        {
            "HighBP": 1, "HighChol": 1, "CholCheck": 1, "BMI": 32, "Smoker": 0, "Stroke": 0,
            "HeartDiseaseorAttack": 1, "PhysActivity": 0, "Fruits": 1, "Veggies": 1,
            "HvyAlcoholConsump": 0, "AnyHealthcare": 1, "NoDocbcCost": 0, "GenHlth": 4,
            "MentHlth": 5, "PhysHlth": 10, "DiffWalk": 1, "Sex": 1, "Age": 9,
            "Education": 4, "Income": 5
        },
        {
            "HighBP": 0, "HighChol": 0, "CholCheck": 1, "BMI": 22, "Smoker": 0, "Stroke": 0,
            "HeartDiseaseorAttack": 0, "PhysActivity": 1, "Fruits": 1, "Veggies": 1,
            "HvyAlcoholConsump": 0, "AnyHealthcare": 1, "NoDocbcCost": 0, "GenHlth": 1,
            "MentHlth": 0, "PhysHlth": 0, "DiffWalk": 0, "Sex": 0, "Age": 3,
            "Education": 6, "Income": 8
        }
    ])


def test_feature_engineer_output_shape(sample_survey_data):
    """Verify transformer expands 21 features to 27 features."""
    fe = DiabetesFeatureEngineer(include_original=True)
    out_df = fe.fit_transform(sample_survey_data)
    assert out_df.shape == (2, 27)
    for name in ENGINEERED_FEATURE_NAMES:
        assert name in out_df.columns


def test_bmi_category_calculation():
    """Verify WHO BMI category boundaries match clinical definitions."""
    df_bmis = pd.DataFrame({
        "BMI": [16.0, 22.0, 27.0, 32.0, 37.0, 45.0],
        "HighBP": [0]*6, "HighChol": [0]*6, "CholCheck": [1]*6, "Smoker": [0]*6,
        "Stroke": [0]*6, "HeartDiseaseorAttack": [0]*6, "PhysActivity": [1]*6,
        "Fruits": [1]*6, "Veggies": [1]*6, "HvyAlcoholConsump": [0]*6,
        "AnyHealthcare": [1]*6, "NoDocbcCost": [0]*6, "GenHlth": [2]*6,
        "MentHlth": [0]*6, "PhysHlth": [0]*6, "DiffWalk": [0]*6, "Sex": [0]*6,
        "Age": [5]*6, "Education": [5]*6, "Income": [5]*6
    })
    fe = DiabetesFeatureEngineer()
    out = fe.fit_transform(df_bmis)
    expected_cats = [1, 2, 3, 4, 5, 6]
    assert list(out["BMI_Category"]) == expected_cats


def test_cardiometabolic_risk_score(sample_survey_data):
    """Verify sum of the 4 vascular conditions."""
    fe = DiabetesFeatureEngineer()
    out = fe.fit_transform(sample_survey_data)
    # Row 0: HighBP=1 + HighChol=1 + Heart=1 + Stroke=0 -> 3
    # Row 1: HighBP=0 + HighChol=0 + Heart=0 + Stroke=0 -> 0
    assert out.loc[0, "CardioMetabolic_Risk_Score"] == 3
    assert out.loc[1, "CardioMetabolic_Risk_Score"] == 0


def test_zero_leakage_row_independence(sample_survey_data):
    """Verify row-level independence: transforming a row alone gives identical result to batch."""
    fe = DiabetesFeatureEngineer()
    batch_res = fe.fit_transform(sample_survey_data)
    single_res = fe.transform(sample_survey_data.iloc[[0]])

    for col in ENGINEERED_FEATURE_NAMES:
        assert batch_res.loc[0, col] == single_res.loc[0, col]


def test_full_pipeline_persistence(sample_survey_data, tmp_path):
    """Verify full end-to-end Pipeline can be fit, transform, saved, and reloaded."""
    pipeline = create_full_pipeline(scaling="standard", use_feature_engineering=True)
    pipeline.fit(sample_survey_data)
    trans1 = pipeline.transform(sample_survey_data)

    model_file = tmp_path / "test_pipeline.joblib"
    joblib.dump(pipeline, model_file)
    loaded_pipeline = joblib.load(model_file)
    trans2 = loaded_pipeline.transform(sample_survey_data)

    assert np.allclose(trans1, trans2)
    assert trans1.shape == (2, 27)
