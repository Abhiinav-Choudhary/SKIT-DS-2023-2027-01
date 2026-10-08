"""Feature Engineering Module for CDC Diabetes Health Indicators.

This module provides a reproducible, stateless scikit-learn transformer that
derives clinically grounded and statistically validated features from raw BRFSS survey responses.

All transformations are strictly row-wise and independent of target values or cohort
aggregations, guaranteeing zero data leakage at training, cross-validation, and inference time.

Project: AI-Powered Health Risk Prediction and Monitoring System
Author: Abhiram (Lead AI/ML Engineer)
Date: 8 October 2026
"""

from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Candidate Engineered Features Definitions and Rationale:
# 1. BMI_Category: WHO standard obesity classification (1: Underweight to 6: Severe Obese Class III).
# 2. CardioMetabolic_Risk_Score: Cumulative count of 4 key comorbidities (HighBP + HighChol + HeartDisease + Stroke).
# 3. HighBP_x_HighChol: Binary interaction capturing concurrent hypertension and dyslipidemia.
# 4. Healthy_Lifestyle_Score: Sum of 5 protective habits (PhysActivity + Fruits + Veggies + NonSmoker + NonHeavyDrinker).
# 5. Age_x_GenHlth: Multiplicative interaction between two strongest ordinal predictors.
# 6. Sedentary_Obese: Indicator combining physical inactivity with clinical obesity (BMI >= 30).

ENGINEERED_FEATURE_NAMES: List[str] = [
    "BMI_Category",
    "CardioMetabolic_Risk_Score",
    "HighBP_x_HighChol",
    "Healthy_Lifestyle_Score",
    "Age_x_GenHlth",
    "Sedentary_Obese",
]


class DiabetesFeatureEngineer(BaseEstimator, TransformerMixin):
    """Stateless Scikit-Learn Transformer for Clinically Grounded Feature Engineering.

    Calculates derived features without looking at target variables or cohort statistics,
    guaranteeing zero data leakage between train, validation, and test splits.
    """

    def __init__(self, include_original: bool = True):
        self.include_original = include_original
        self.feature_names_out_: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame, y=None):
        """Fit method (stateless; validates input columns)."""
        required_cols = [
            "BMI", "HighBP", "HighChol", "HeartDiseaseorAttack", "Stroke",
            "PhysActivity", "Fruits", "Veggies", "Smoker", "HvyAlcoholConsump",
            "Age", "GenHlth"
        ]
        if isinstance(X, pd.DataFrame):
            missing = set(required_cols) - set(X.columns)
            if missing:
                raise ValueError(f"Input DataFrame is missing required feature columns: {missing}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Derive engineered features row-wise."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("DiabetesFeatureEngineer expects a pandas DataFrame as input.")

        df = X.copy()

        # 1. WHO Standard BMI Category (1 to 6)
        # 1: Underweight (<18.5)
        # 2: Normal weight (18.5 - 24.9)
        # 3: Overweight (25.0 - 29.9)
        # 4: Obese Class I (30.0 - 34.9)
        # 5: Obese Class II (35.0 - 39.9)
        # 6: Obese Class III (>= 40.0)
        bmi = df["BMI"].values
        bmi_cat = np.select(
            [bmi < 18.5, bmi < 25.0, bmi < 30.0, bmi < 35.0, bmi < 40.0],
            [1, 2, 3, 4, 5],
            default=6,
        ).astype(np.int8)
        df["BMI_Category"] = bmi_cat

        # 2. Cardiometabolic Comorbidity Risk Score (0 to 4)
        df["CardioMetabolic_Risk_Score"] = (
            df["HighBP"] + df["HighChol"] + df["HeartDiseaseorAttack"] + df["Stroke"]
        ).astype(np.int8)

        # 3. HighBP x HighChol Interaction
        df["HighBP_x_HighChol"] = (df["HighBP"] * df["HighChol"]).astype(np.int8)

        # 4. Healthy Lifestyle Composite Score (0 to 5)
        non_smoker = 1 - df["Smoker"]
        non_heavy_drinker = 1 - df["HvyAlcoholConsump"]
        df["Healthy_Lifestyle_Score"] = (
            df["PhysActivity"] + df["Fruits"] + df["Veggies"] + non_smoker + non_heavy_drinker
        ).astype(np.int8)

        # 5. Age x GenHlth Interaction
        df["Age_x_GenHlth"] = (df["Age"] * df["GenHlth"]).astype(np.int16)

        # 6. Sedentary Obesity Indicator (0 or 1)
        is_sedentary = 1 - df["PhysActivity"]
        is_obese = (df["BMI"] >= 30).astype(int)
        df["Sedentary_Obese"] = (is_sedentary * is_obese).astype(np.int8)

        if not self.include_original:
            df = df[ENGINEERED_FEATURE_NAMES]

        self.feature_names_out_ = list(df.columns)
        return df

    def get_feature_names_out(self, input_features=None) -> List[str]:
        """Return names of output features."""
        if self.feature_names_out_ is not None:
            return self.feature_names_out_
        if input_features is not None:
            return list(input_features) + ENGINEERED_FEATURE_NAMES
        return ENGINEERED_FEATURE_NAMES
