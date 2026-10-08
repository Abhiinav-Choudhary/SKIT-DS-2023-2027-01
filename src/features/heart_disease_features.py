"""Feature engineering for the UCI Heart Disease dataset.

All features are row-wise transformations of predictor columns available in the
existing 14-column processed Heart Disease files. The target column is never
used by this transformer.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


HEART_ENGINEERED_NUMERIC_FEATURES: List[str] = [
    "age_x_oldpeak",
    "resting_bp_cholesterol_index",
    "heart_rate_reserve_proxy",
    "oldpeak_per_age",
]

HEART_ENGINEERED_CATEGORICAL_FEATURES: List[str] = [
    "asymptomatic_chest_pain",
    "exercise_ischemia_signal",
    "has_major_vessels_colored",
    "thal_reversible_defect",
    "high_resting_bp",
    "high_cholesterol",
    "abnormal_restecg",
]

HEART_ENGINEERED_FEATURE_NAMES: List[str] = (
    HEART_ENGINEERED_NUMERIC_FEATURES + HEART_ENGINEERED_CATEGORICAL_FEATURES
)


def _binary_indicator(series: pd.Series, condition: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=series.index, dtype="float64")
    observed = series.notna()
    result.loc[observed] = condition.loc[observed].astype(float)
    return result


class HeartDiseaseFeatureEngineer(BaseEstimator, TransformerMixin):
    """Stateless sklearn transformer for Heart Disease predictor features."""

    def __init__(self, include_original: bool = True):
        self.include_original = include_original
        self.feature_names_out_: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame, y=None):
        required_cols = [
            "age",
            "cp",
            "trestbps",
            "chol",
            "restecg",
            "thalach",
            "exang",
            "oldpeak",
            "ca",
            "thal",
        ]
        if not isinstance(X, pd.DataFrame):
            raise TypeError("HeartDiseaseFeatureEngineer expects a pandas DataFrame as input.")

        missing = set(required_cols) - set(X.columns)
        if missing:
            raise ValueError(f"Input DataFrame is missing required feature columns: {sorted(missing)}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("HeartDiseaseFeatureEngineer expects a pandas DataFrame as input.")

        df = X.copy()
        for column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

        df["age_x_oldpeak"] = df["age"] * df["oldpeak"]
        df["resting_bp_cholesterol_index"] = df["trestbps"] * df["chol"]
        df["heart_rate_reserve_proxy"] = (220 - df["age"]) - df["thalach"]
        df["oldpeak_per_age"] = df["oldpeak"] / df["age"].replace(0, np.nan)

        df["asymptomatic_chest_pain"] = _binary_indicator(df["cp"], df["cp"] == 4)

        exang = df["exang"]
        oldpeak = df["oldpeak"]
        ischemia_observed = exang.notna() | oldpeak.notna()
        ischemia_positive = (exang == 1) | (oldpeak > 0)
        df["exercise_ischemia_signal"] = pd.Series(np.nan, index=df.index, dtype="float64")
        df.loc[ischemia_observed, "exercise_ischemia_signal"] = (
            ischemia_positive.loc[ischemia_observed].astype(float)
        )

        df["has_major_vessels_colored"] = _binary_indicator(df["ca"], df["ca"] > 0)
        df["thal_reversible_defect"] = _binary_indicator(df["thal"], df["thal"] == 7)
        df["high_resting_bp"] = _binary_indicator(df["trestbps"], df["trestbps"] >= 140)
        df["high_cholesterol"] = _binary_indicator(df["chol"], df["chol"] >= 240)
        df["abnormal_restecg"] = _binary_indicator(df["restecg"], df["restecg"] > 0)

        if not self.include_original:
            df = df[HEART_ENGINEERED_FEATURE_NAMES]

        self.feature_names_out_ = list(df.columns)
        return df

    def get_feature_names_out(self, input_features=None) -> List[str]:
        if self.feature_names_out_ is not None:
            return self.feature_names_out_
        if input_features is not None and self.include_original:
            return list(input_features) + HEART_ENGINEERED_FEATURE_NAMES
        return HEART_ENGINEERED_FEATURE_NAMES
