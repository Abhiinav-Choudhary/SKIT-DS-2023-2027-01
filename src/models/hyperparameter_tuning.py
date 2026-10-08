"""Targeted hyperparameter tuning with preprocessing inside the sklearn Pipeline.

Tunes the two strongest baseline architectures:
  1. Gradient Boosting (HistGradientBoostingClassifier)
  2. Random Forest (RandomForestClassifier)

Search uses only the training partition with StratifiedKFold cross-validation.
Preprocessing (feature engineering + scaling) is refit inside each CV fold.

Project: AI-Powered Health Risk Prediction and Monitoring System
Author: Abhiram (Lead AI/ML Engineer)
Date: 8 October 2026
"""

from typing import Any, Dict, Tuple, Union

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.preprocessing.pipeline import create_full_pipeline


def _stratified_cv(n_splits: int, random_state: int) -> StratifiedKFold:
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def _build_preprocessing_classifier_pipeline(
    classifier: Union[HistGradientBoostingClassifier, RandomForestClassifier],
) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "preprocess",
                create_full_pipeline(scaling="standard", use_feature_engineering=True),
            ),
            ("clf", classifier),
        ]
    )


def tune_gradient_boosting(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 15,
    cv_splits: int = 5,
    random_state: int = 42,
    n_jobs: int = -1,
) -> Tuple[Pipeline, Dict[str, Any]]:
    """Tune Gradient Boosting with RandomizedSearchCV on raw training features only."""
    pipeline = _build_preprocessing_classifier_pipeline(
        HistGradientBoostingClassifier(class_weight="balanced", random_state=random_state)
    )
    param_dist = {
        "clf__learning_rate": [0.05, 0.08, 0.10, 0.12],
        "clf__max_depth": [6, 8, 10, None],
        "clf__min_samples_leaf": [20, 35, 50, 75],
        "clf__l2_regularization": [0.0, 1.0, 5.0, 10.0],
        "clf__max_iter": [100, 125, 150],
    }

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=_stratified_cv(cv_splits, random_state),
        scoring="roc_auc",
        refit=True,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=1,
    )
    search.fit(X_train, y_train)

    summary = {
        "model": "Gradient Boosting (HistGradientBoostingClassifier)",
        "best_score_cv_roc_auc": float(search.best_score_),
        "best_params": search.best_params_,
        "best_params_classifier_only": {
            key.removeprefix("clf__"): value for key, value in search.best_params_.items()
        },
    }
    return search.best_estimator_, summary


def tune_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 8,
    cv_splits: int = 5,
    random_state: int = 42,
    n_jobs: int = -1,
) -> Tuple[Pipeline, Dict[str, Any]]:
    """Tune Random Forest with RandomizedSearchCV on raw training features only."""
    pipeline = _build_preprocessing_classifier_pipeline(
        RandomForestClassifier(
            class_weight="balanced",
            n_jobs=1,
            random_state=random_state,
        )
    )
    param_dist = {
        "clf__n_estimators": [100, 125],
        "clf__max_depth": [12, 14, 16],
        "clf__min_samples_leaf": [10, 20, 40],
        "clf__min_samples_split": [20, 40],
        "clf__max_features": ["sqrt", 0.5],
    }

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=_stratified_cv(cv_splits, random_state),
        scoring="roc_auc",
        refit=True,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=1,
    )
    search.fit(X_train, y_train)

    summary = {
        "model": "Random Forest (RandomForestClassifier)",
        "best_score_cv_roc_auc": float(search.best_score_),
        "best_params": search.best_params_,
        "best_params_classifier_only": {
            key.removeprefix("clf__"): value for key, value in search.best_params_.items()
        },
    }
    return search.best_estimator_, summary


def select_champion_by_cv_score(
    candidates: Dict[str, Tuple[Pipeline, Dict[str, Any]]],
) -> Tuple[str, Pipeline, Dict[str, Any]]:
    """Pick the tuned pipeline with the highest cross-validated ROC-AUC on train."""
    best_name = max(
        candidates,
        key=lambda name: candidates[name][1]["best_score_cv_roc_auc"],
    )
    pipeline, summary = candidates[best_name]
    return best_name, pipeline, summary
