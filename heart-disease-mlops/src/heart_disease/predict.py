"""Predict one example with the packaged fitted pipeline; never fit at runtime."""

from __future__ import annotations

import argparse
import json
import math
import sys
from numbers import Real
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from .feature_engineering import (
    CATEGORICAL_FEATURES,
    EXPECTED_COLUMNS,
    NUMERICAL_FEATURES,
    project_root,
)

MODEL_FILENAME = "final_heart_disease_pipeline.joblib"
METADATA_FILENAME = "final_heart_disease_metadata.json"
CATEGORICAL_CODES = {
    "sex": {0, 1},
    "cp": {1, 2, 3, 4},
    "fbs": {0, 1},
    "restecg": {0, 1, 2},
    "exang": {0, 1},
    "slope": {1, 2, 3},
    "ca": {0, 1, 2, 3},
    "thal": {3, 6, 7},
}
_TARGET_LABELS = {0: "absence", 1: "presence"}


class ModelPredictionError(Exception):
    """Explain why one prediction could not be produced."""


def _read_user_input(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    if args.input_json is not None:
        try:
            value = json.loads(args.input_json)
        except json.JSONDecodeError as error:
            raise ModelPredictionError(
                f"--input-json must contain valid JSON: {error}"
            ) from error
        return _require_object(value), "user-provided JSON"

    if args.input_file is not None:
        try:
            value = json.loads(args.input_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ModelPredictionError(
                f"Could not read JSON input file '{args.input_file}': {error}"
            ) from error
        return _require_object(value), f"JSON file: {args.input_file}"

    dataset_path = project_root() / "data" / "processed" / "heart_disease_cleaned.csv"
    if not dataset_path.is_file():
        raise ModelPredictionError(
            "No input was supplied and the cleaned dataset is unavailable for "
            "the demonstration sample. Supply --input-json or --input-file."
        )
    try:
        row = pd.read_csv(dataset_path, usecols=EXPECTED_COLUMNS[:-1]).iloc[0]
    except (OSError, pd.errors.ParserError, ValueError, IndexError) as error:
        raise ModelPredictionError(
            f"Could not load the demonstration feature row from "
            f"'{dataset_path}': {error}"
        ) from error

    sample = {
        name: (
            None
            if pd.isna(value)
            else value.item()
            if hasattr(value, "item")
            else value
        )
        for name, value in row.items()
    }
    return sample, "demonstration only: first cleaned-data row's features; not evaluated"


def _require_object(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ModelPredictionError("The prediction input must be a JSON object.")
    return value


def _validate_input(example: dict[str, Any]) -> pd.DataFrame:
    expected_features = EXPECTED_COLUMNS[:-1]
    missing = [name for name in expected_features if name not in example]
    unexpected = [name for name in example if name not in expected_features]
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing required features: {missing}")
        if unexpected:
            details.append(f"unexpected fields: {unexpected}")
        raise ModelPredictionError("; ".join(details))

    normalized: dict[str, Any] = {}
    for name in expected_features:
        value = example[name]
        if value is None:
            normalized[name] = None
            continue
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ModelPredictionError(
                f"Feature '{name}' must be a finite number"
                + (" or null." if name in NUMERICAL_FEATURES else " integer code or null.")
            )
        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            raise ModelPredictionError(
                f"Feature '{name}' must be finite; use JSON null for an unknown value."
            )
        if name in CATEGORICAL_FEATURES:
            if not numeric_value.is_integer():
                raise ModelPredictionError(
                    f"Categorical feature '{name}' must be an integer category code."
                )
            category = int(numeric_value)
            if category not in CATEGORICAL_CODES[name]:
                raise ModelPredictionError(
                    f"Feature '{name}' has unsupported category {category}; "
                    f"expected one of {sorted(CATEGORICAL_CODES[name])}."
                )
            normalized[name] = category
        else:
            normalized[name] = numeric_value

    return pd.DataFrame([normalized], columns=expected_features)


def _load_pipeline() -> tuple[Pipeline, dict[str, Any]]:
    root = project_root()
    model_path = root / "artifacts" / MODEL_FILENAME
    metadata_path = root / "artifacts" / METADATA_FILENAME
    if not model_path.is_file():
        raise ModelPredictionError(
            f"Packaged model is missing: {model_path}. "
            "Run `python -m heart_disease.package_model` first."
        )
    if not metadata_path.is_file():
        raise ModelPredictionError(
            f"Model metadata is missing: {metadata_path}. "
            "Run `python -m heart_disease.package_model` first."
        )

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        model = joblib.load(model_path)
    except (OSError, EOFError, ValueError, json.JSONDecodeError, ImportError) as error:
        raise ModelPredictionError(
            f"Could not load the packaged model and metadata: {error}"
        ) from error

    if not isinstance(model, Pipeline):
        raise ModelPredictionError(
            f"Expected a scikit-learn Pipeline, found {type(model).__name__}."
        )
    if list(getattr(model, "feature_names_in_", [])) != EXPECTED_COLUMNS[:-1]:
        raise ModelPredictionError(
            "The packaged model feature names/order do not match the expected "
            "Step 2/3 schema."
        )
    if metadata.get("required_features_in_order") != EXPECTED_COLUMNS[:-1]:
        raise ModelPredictionError(
            "Model metadata feature names/order do not match the expected schema."
        )
    if not {"preprocessing", "classifier"}.issubset(model.named_steps):
        raise ModelPredictionError(
            "The packaged model is missing its preprocessing or classifier step."
        )
    if not hasattr(model.named_steps["preprocessing"], "transformers_"):
        raise ModelPredictionError(
            "The packaged preprocessing step is not fitted; prediction will not fit it."
        )
    return model, metadata


def predict_with_pipeline(
    example: dict[str, Any],
    input_source: str,
    model: Pipeline,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Validate and predict using an already-loaded fitted pipeline."""
    features = _validate_input(example)
    predicted_class = int(model.predict(features)[0])
    if predicted_class not in _TARGET_LABELS:
        raise ModelPredictionError(
            f"The model returned unexpected class {predicted_class}; "
            "expected 0 (absence) or 1 (presence)."
        )

    result: dict[str, Any] = {
        "selected_model": metadata["selected_model_name"],
        "input_source": input_source,
        "predicted_class": predicted_class,
        "predicted_label": _TARGET_LABELS[predicted_class],
        "target_encoding": "0 = absence; 1 = presence (original num values 1-4)",
        "medical_notice": (
            "This is an educational model output, not a medical diagnosis or "
            "medical advice. Seek a qualified clinician for health concerns."
        ),
    }

    predict_proba = getattr(model, "predict_proba", None)
    if callable(predict_proba):
        probabilities = predict_proba(features)[0]
        result["class_probabilities"] = {
            f"{int(class_label)} ({_TARGET_LABELS[int(class_label)]})": float(score)
            for class_label, score in zip(model.classes_, probabilities)
        }
        result["score_interpretation"] = (
            "These are the model's estimated class probabilities for this input; "
            "they are not a diagnosis and are not necessarily calibrated."
        )
    return result


def predict(example: dict[str, Any], input_source: str) -> dict[str, Any]:
    """Load the saved pipeline and predict one example."""
    model, metadata = _load_pipeline()
    return predict_with_pipeline(example, input_source, model, metadata)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Predict one heart-disease target class using the packaged fitted "
            "pipeline. No training or preprocessing fitting occurs."
        )
    )
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--input-json",
        help="One JSON object containing all 13 input features.",
    )
    input_group.add_argument(
        "--input-file",
        type=Path,
        help="Path to a UTF-8 JSON file containing one object with all features.",
    )
    args = parser.parse_args()

    example, input_source = _read_user_input(args)
    result = predict(example, input_source)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    try:
        main()
    except ModelPredictionError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
