"""CLI interface for production health risk model inference.

Enables headless subprocess execution (e.g. from Node.js / Express backend)
via JSON arguments, file paths, or stdin pipes.

Author: Lead AI/ML Engineer
Date: 15 November 2026
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict

from src.inference.diabetes_risk_model import predict_diabetes_risk
from src.inference.heart_disease_model import predict_heart_disease


def run_cli() -> int:
    parser = argparse.ArgumentParser(
        description="Health Risk Prediction CLI for Backend Integration"
    )
    parser.add_argument(
        "--model",
        choices=["diabetes", "heart"],
        required=True,
        help="Model to invoke for inference ('diabetes' or 'heart').",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="JSON string containing input feature dictionary or list of feature dictionaries.",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to JSON file containing input features.",
    )

    args = parser.parse_args()

    # Read payload from --input, --file, or stdin
    try:
        if args.input is not None:
            raw_data = args.input
        elif args.file is not None:
            with open(args.file, "r", encoding="utf-8") as f:
                raw_data = f.read()
        else:
            if sys.stdin.isatty():
                parser.error("No input provided via --input, --file, or stdin pipe.")
            raw_data = sys.stdin.read()

        payload = json.loads(raw_data)
    except Exception as exc:
        error_output = {
            "status": "error",
            "error_type": type(exc).__name__,
            "message": f"Failed to parse input JSON: {exc}",
        }
        sys.stderr.write(json.dumps(error_output) + "\n")
        return 1

    # Execute model prediction
    try:
        if args.model == "diabetes":
            prediction = predict_diabetes_risk(payload)
        elif args.model == "heart":
            prediction = predict_heart_disease(payload)
        else:
            raise ValueError(f"Unknown model identifier: {args.model}")

        output: Dict[str, Any] = {
            "status": "success",
            "model": args.model,
            "data": prediction,
        }
        sys.stdout.write(json.dumps(output, indent=2) + "\n")
        return 0

    except Exception as exc:
        error_output = {
            "status": "error",
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
        sys.stderr.write(json.dumps(error_output) + "\n")
        return 1


if __name__ == "__main__":
    sys.exit(run_cli())
