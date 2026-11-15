# Diabetes Backend Integration Contract

This contract describes the currently implemented Python inference interface in
`src/inference/diabetes_risk_model.py`. It does not define an HTTP API.

## Entry Point

Use `predict_diabetes_risk(features, pipeline=None)`.

- `features` may be a `dict` for one record or a `pandas.DataFrame` for one or more records.
- If `pipeline` is not provided, the function calls `load_diabetes_risk_pipeline()`.
- `load_diabetes_risk_pipeline()` loads `models/diabetes/diabetes_risk_pipeline.joblib`.
- The loaded artifact must be a `sklearn.pipeline.Pipeline`.

## Required Input Fields

The inference input must include these 21 raw CDC diabetes feature columns:

| Field | Type accepted by current validation |
| --- | --- |
| `HighBP` | numeric integer-coded |
| `HighChol` | numeric integer-coded |
| `CholCheck` | numeric integer-coded |
| `BMI` | numeric |
| `Smoker` | numeric integer-coded |
| `Stroke` | numeric integer-coded |
| `HeartDiseaseorAttack` | numeric integer-coded |
| `PhysActivity` | numeric integer-coded |
| `Fruits` | numeric integer-coded |
| `Veggies` | numeric integer-coded |
| `HvyAlcoholConsump` | numeric integer-coded |
| `AnyHealthcare` | numeric integer-coded |
| `NoDocbcCost` | numeric integer-coded |
| `GenHlth` | numeric integer-coded |
| `MentHlth` | numeric integer-coded |
| `PhysHlth` | numeric integer-coded |
| `DiffWalk` | numeric integer-coded |
| `Sex` | numeric integer-coded |
| `Age` | numeric integer-coded |
| `Education` | numeric integer-coded |
| `Income` | numeric integer-coded |

Extra fields are ignored because the function reorders/selects only the required
columns before prediction.

## Validation And Ranges

All required fields are converted with `pandas.to_numeric(errors="coerce")`.
Non-numeric or missing values fail validation.

| Field(s) | Allowed range |
| --- | --- |
| `HighBP`, `HighChol`, `CholCheck`, `Smoker`, `Stroke`, `HeartDiseaseorAttack`, `PhysActivity`, `Fruits`, `Veggies`, `HvyAlcoholConsump`, `AnyHealthcare`, `NoDocbcCost`, `DiffWalk`, `Sex` | `0` to `1` |
| `BMI` | `12` to `100` |
| `GenHlth` | `1` to `5` |
| `MentHlth` | `0` to `30` |
| `PhysHlth` | `0` to `30` |
| `Age` | `1` to `13` |
| `Education` | `1` to `6` |
| `Income` | `1` to `8` |

All fields except `BMI` must be integer-coded.

## Output

`predict_diabetes_risk(...)` returns a dictionary with:

- `predicted_class`: list of labels returned by `pipeline.predict(X)`.
- `diabetes_probability`: list of class-1 probabilities returned by
  `pipeline.predict_proba(X)[:, 1]`.

No clinical threshold, risk category, or decision text is implemented in this
inference function.

## Errors And Validation Behavior

- Missing required input columns raise `ValueError` with the missing column names.
- Non-numeric or missing required values raise `ValueError`.
- Out-of-range values raise `ValueError` with the invalid values and expected ranges.
- Non-integer values for integer-coded fields raise `ValueError`.
- If the model file is missing, `load_diabetes_risk_pipeline()` raises `FileNotFoundError`.
- If the loaded artifact is not a `sklearn.pipeline.Pipeline`,
  `load_diabetes_risk_pipeline()` raises `TypeError`.

## Actual Model File

The model file used by the inference loader is:

`models/diabetes/diabetes_risk_pipeline.joblib`

The metadata file alongside it identifies the model as
`Tuned HistGradientBoostingClassifier (Gradient Boosting champion)`.
