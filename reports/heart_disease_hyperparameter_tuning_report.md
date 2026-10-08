# Heart Disease Hyperparameter Tuning Report

Target: Existing multiclass `num` target with classes 0, 1, 2, 3, 4.

Only the training split was loaded for tuning. Validation and test were not used.

## Selection

Selected `Logistic Regression` by highest mean CV weighted F1. Weighted one-vs-rest ROC-AUC was the first tie-breaker.

## Tuned Candidate Results

| Model | Search | Candidates | CV F1 Weighted | CV F1 Macro | CV Accuracy | CV ROC-AUC OVR Weighted | Best Params |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Logistic Regression | GridSearchCV | 14 | 0.5671 | 0.4139 | 0.5483 | 0.7954 | `{'clf__C': 0.3, 'clf__class_weight': 'balanced'}` |
| Gradient Boosting | RandomizedSearchCV | 16 | 0.5573 | 0.3829 | 0.5483 | 0.7836 | `{'clf__min_samples_leaf': 30, 'clf__max_leaf_nodes': 31, 'clf__max_iter': 50, 'clf__max_depth': 3, 'clf__learning_rate': 0.15, 'clf__l2_regularization': 1.0, 'clf__class_weight': 'balanced'}` |

## Files

- `reports\heart_disease_logistic_regression_tuning_cv_results.csv`
- `reports\heart_disease_gradient_boosting_tuning_cv_results.csv`
- `models\heart\tuned\logistic_regression_tuned_pipeline.joblib`
- `models\heart\tuned\gradient_boosting_tuned_pipeline.joblib`
- `models\heart\tuned\heart_disease_selected_tuned_pipeline.joblib`
- `reports/heart_disease_hyperparameter_tuning_report.json`
- `reports/heart_disease_hyperparameter_tuning_report.md`
