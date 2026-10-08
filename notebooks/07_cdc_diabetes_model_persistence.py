"""Step 11: Production-ready persistence for the final diabetes risk model.

Uses the verified Step 10 champion pipeline on disk. Does not retrain.

Project: AI-Powered Health Risk Prediction and Monitoring System
Author: Abhiram (Lead AI/ML Engineer)
Date: 8 October 2026
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.inference.diabetes_risk_model import (
    get_metadata_path,
    get_pipeline_path,
    load_diabetes_risk_pipeline,
    save_production_metadata,
    predict_diabetes_risk,
    INPUT_FEATURE_COLUMNS,
)


def run_model_persistence_and_verification():
    print("=" * 80)
    print("STEP 11: PRODUCTION MODEL PERSISTENCE & INFERENCE VERIFICATION")
    print("=" * 80)

    pipeline_path = get_pipeline_path()
    if not pipeline_path.is_file():
        raise FileNotFoundError(
            f"Expected pipeline at {pipeline_path}. Complete Step 10 before Step 11."
        )

    print(f"\n[1] Pipeline artifact found: {pipeline_path.relative_to(PROJECT_ROOT)}")
    print(f"    Size: {pipeline_path.stat().st_size / (1024 * 1024):.2f} MB")

    pipeline = load_diabetes_risk_pipeline()
    print(f"    Steps: {[name for name, _ in pipeline.steps]}")

    metadata_path = save_production_metadata(pipeline)
    print(f"\n[2] Metadata written: {metadata_path.relative_to(PROJECT_ROOT)}")

    with open(metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)
    print(f"    model_name   : {metadata['model_name']}")
    print(f"    model_version: {metadata['model_version']}")
    print(f"    target       : {metadata['target_variable']}")

    print("\n[3] Loading pipeline from disk (fresh load)...")
    reloaded = load_diabetes_risk_pipeline()

    print("\n[4] Sample prediction (first row of held-out test partition)...")
    test_path = PROJECT_ROOT / "data" / "processed" / "diabetes" / "diabetes_test.csv"
    sample = pd.read_csv(test_path, nrows=1)
    actual = int(sample["Diabetes_binary"].iloc[0])
    sample_features = sample[INPUT_FEATURE_COLUMNS].iloc[0].to_dict()

    result = predict_diabetes_risk(sample_features, pipeline=reloaded)
    pred = int(result["predicted_class"][0])
    prob = float(result["diabetes_probability"][0])

    print(f"    Actual label     : {actual}")
    print(f"    Predicted class  : {pred}")
    print(f"    P(diabetes=1)    : {prob:.6f}")
    print("\n[OK] Prediction completed successfully.")
    print("=" * 80)


if __name__ == "__main__":
    run_model_persistence_and_verification()
