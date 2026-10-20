"""API tests use the verified packaged model and project feature schema."""

from __future__ import annotations

from typing import Any

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from heart_disease.api import app
from heart_disease.feature_engineering import project_root

client = TestClient(app)
FEATURE_NAMES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]
CATEGORICAL_FEATURES = {
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal",
}
OBSERVED_CLEVELAND_ROW: dict[str, int | float] = {
    "age": 63.0,
    "sex": 1,
    "cp": 1,
    "trestbps": 145.0,
    "chol": 233.0,
    "fbs": 1,
    "restecg": 2,
    "thalach": 150.0,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 3,
    "ca": 0,
    "thal": 6,
}


def test_health_endpoint_reports_running_api() -> None:
    """The health route works without loading or requiring a model artifact."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_endpoint_returns_saved_model_result() -> None:
    """One valid project-schema input returns a class from the saved pipeline."""
    root = project_root()
    model_path = root / "artifacts" / "final_heart_disease_pipeline.joblib"
    data_path = root / "data" / "processed" / "heart_disease_cleaned.csv"
    if not model_path.is_file() or not data_path.is_file():
        pytest.skip(
            "This integration test needs the Git-ignored packaged model and "
            "cleaned Cleveland dataset available in the local workspace."
        )

    model = joblib.load(model_path)
    saved_features = list(model.feature_names_in_)
    assert saved_features == FEATURE_NAMES
    sample_frame = pd.read_csv(data_path, usecols=saved_features).iloc[0]
    example: dict[str, Any] = {
        name: (
            None
            if pd.isna(value)
            else int(value)
            if name in CATEGORICAL_FEATURES
            else value.item()
            if hasattr(value, "item")
            else value
        )
        for name, value in sample_frame.items()
    }

    response = client.post("/predict", json=example)

    assert response.status_code == 200, response.text
    result = response.json()
    assert result["predicted_class"] in model.classes_.tolist()
    assert result["predicted_label"] == {
        0: "absence",
        1: "presence",
    }[result["predicted_class"]]
    assert result["target_encoding"] == (
        "0 = absence; 1 = presence (original num values 1-4)"
    )
    assert "medical diagnosis" in result["medical_notice"]
    if callable(getattr(model, "predict_proba", None)):
        assert result["class_probabilities"]
    else:
        assert "class_probabilities" not in result


@pytest.mark.parametrize(
    ("change", "message_fragment"),
    [
        ({"remove": "age"}, "age"),
        ({"add": ("unexpected", 1)}, "Extra inputs are not permitted"),
        ({"set": ("cp", 99)}, "unsupported category"),
        ({"set": ("sex", 0.5)}, "Input should be a valid integer"),
    ],
)
def test_predict_endpoint_rejects_invalid_features(
    change: dict[str, Any],
    message_fragment: str,
) -> None:
    """Missing, extra, and invalid categorical inputs receive clear 422 errors."""
    example: dict[str, Any] = OBSERVED_CLEVELAND_ROW.copy()
    if "remove" in change:
        example.pop(change["remove"])
    if "add" in change:
        name, value = change["add"]
        example[name] = value
    if "set" in change:
        name, value = change["set"]
        example[name] = value

    response = client.post("/predict", json=example)

    assert response.status_code == 422
    assert message_fragment.lower() in response.text.lower()
