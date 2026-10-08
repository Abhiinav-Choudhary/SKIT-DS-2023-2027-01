"""Baseline Model Definitions for Health Risk Classification.

Defines the four required baseline models:
  1. Logistic Regression
  2. Decision Tree
  3. Random Forest
  4. Gradient Boosting (HistGradientBoostingClassifier)

Supports both balanced class weighting (for clinical screening sensitivity)
and default unweighted mode (for baseline comparison).

Project: AI-Powered Health Risk Prediction and Monitoring System
Author: Abhiram (Lead AI/ML Engineer)
Date: 8 October 2026
"""

from typing import Dict, Any, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier


def get_baseline_models(
    random_state: int = 42,
    class_weight: Optional[str] = "balanced",
    n_jobs: int = -1,
) -> Dict[str, Any]:
    """Instantiate the 4 canonical baseline classification models.

    Args:
      random_state: Random seed for reproducibility.
      class_weight: 'balanced' or None.
      n_jobs: Parallel worker threads for compatible estimators.

    Returns:
      Dictionary mapping model names to configured estimator instances.
    """
    models = {
        "Logistic Regression": LogisticRegression(
            C=1.0,
            penalty="l2",
            solver="lbfgs",
            max_iter=1000,
            class_weight=class_weight,
            random_state=random_state,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight=class_weight,
            random_state=random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=14,
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight=class_weight,
            n_jobs=n_jobs,
            random_state=random_state,
        ),
        "Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=100,
            max_depth=8,
            learning_rate=0.1,
            class_weight=class_weight,
            random_state=random_state,
        ),
    }
    return models
