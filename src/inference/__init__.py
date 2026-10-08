"""Inference entry points for deployed health risk models."""

from src.inference.diabetes_risk_model import (
    load_diabetes_risk_pipeline,
    predict_diabetes_risk,
    save_production_metadata,
)
from src.inference.heart_disease_model import (
    load_heart_disease_pipeline,
    predict_heart_disease,
    save_heart_production_metadata,
)

__all__ = [
    "load_diabetes_risk_pipeline",
    "predict_diabetes_risk",
    "save_production_metadata",
    "load_heart_disease_pipeline",
    "predict_heart_disease",
    "save_heart_production_metadata",
]
