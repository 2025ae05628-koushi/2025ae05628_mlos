"""Monitoring tests use fixed request objects and do not load project data."""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest
from fastapi.testclient import TestClient

from heart_disease import api, predict
from heart_disease.monitoring import (
    FEATURE_ISSUES,
    HTTP_FAILURES,
    HTTP_REQUESTS,
    PREDICTIONS_SUCCEEDED,
    REQUEST_DURATION,
)

client = TestClient(api.app)
VALID_REQUEST: dict[str, int | float | None] = {
    "age": 63,
    "sex": 1,
    "cp": 1,
    "trestbps": 145,
    "chol": 233,
    "fbs": 1,
    "restecg": 2,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 3,
    "ca": 0,
    "thal": 6,
}
FAKE_RESULT = {
    "predicted_class": 1,
    "predicted_label": "presence",
    "class_probabilities": {"0 (absence)": 0.2, "1 (presence)": 0.8},
}


def _counter_value(counter: Any, *labels: str) -> float:
    """Read a labeled Prometheus counter for before/after assertions."""
    return counter.labels(*labels)._value.get()


@pytest.fixture
def fake_prediction(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use an in-memory stand-in instead of loading a real model artifact."""
    monkeypatch.setattr(api, "_load_model", lambda: (object(), {}))
    monkeypatch.setattr(
        predict,
        "predict_with_pipeline",
        lambda *_args, **_kwargs: FAKE_RESULT.copy(),
    )


def test_successful_prediction_metrics_and_log_exclude_input_values(
    fake_prediction: None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Success metrics/logs expose outcomes, not the submitted feature values."""
    before_success = PREDICTIONS_SUCCEEDED._value.get()
    before_requests = _counter_value(HTTP_REQUESTS, "POST", "/predict", "200")
    before_latency = REQUEST_DURATION.labels("POST", "/predict")._sum.get()
    caplog.set_level(logging.INFO, logger="uvicorn.error")

    response = client.post("/predict", json=VALID_REQUEST)

    assert response.status_code == 200
    assert PREDICTIONS_SUCCEEDED._value.get() == before_success + 1
    assert _counter_value(HTTP_REQUESTS, "POST", "/predict", "200") == (
        before_requests + 1
    )
    assert REQUEST_DURATION.labels("POST", "/predict")._sum.get() > before_latency
    event = next(
        json.loads(record.message)
        for record in caplog.records
        if record.name == "uvicorn.error"
        and json.loads(record.message).get("event") == "api_request"
    )
    assert event["predicted_class"] == 1
    assert event["predicted_label"] == "presence"
    assert {"age", "sex", "chol", "trestbps"}.isdisjoint(event)
    assert "233" not in json.dumps(event)
    assert "145" not in json.dumps(event)


def test_missing_invalid_and_out_of_training_range_inputs_are_counted(
    fake_prediction: None,
) -> None:
    """Known data-quality issues increment metrics without blocking valid inference."""
    before_null = _counter_value(FEATURE_ISSUES, "trestbps", "null")
    before_range = _counter_value(FEATURE_ISSUES, "age", "outside_training_range")
    before_missing = _counter_value(FEATURE_ISSUES, "age", "missing")
    before_category = _counter_value(FEATURE_ISSUES, "cp", "invalid_category")

    monitored = VALID_REQUEST.copy()
    monitored["age"] = 90
    monitored["trestbps"] = None
    assert client.post("/predict", json=monitored).status_code == 200

    missing = VALID_REQUEST.copy()
    missing.pop("age")
    assert client.post("/predict", json=missing).status_code == 422

    invalid_category = VALID_REQUEST.copy()
    invalid_category["cp"] = 99
    assert client.post("/predict", json=invalid_category).status_code == 422

    assert _counter_value(FEATURE_ISSUES, "trestbps", "null") == before_null + 1
    assert _counter_value(
        FEATURE_ISSUES,
        "age",
        "outside_training_range",
    ) == before_range + 1
    assert _counter_value(FEATURE_ISSUES, "age", "missing") == before_missing + 1
    assert _counter_value(
        FEATURE_ISSUES,
        "cp",
        "invalid_category",
    ) == before_category + 1
    assert _counter_value(
        HTTP_FAILURES,
        "/predict",
        "422",
        "invalid_input",
    ) >= 2


def test_model_load_failure_is_counted_and_logged(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Model availability failures increment API failure metrics without input."""
    before_failures = _counter_value(
        HTTP_FAILURES,
        "/predict",
        "503",
        "model_unavailable",
    )

    def fail_to_load() -> tuple[object, dict[str, Any]]:
        raise predict.ModelPredictionError("model unavailable for test")

    monkeypatch.setattr(api, "_load_model", fail_to_load)
    caplog.set_level(logging.INFO, logger="uvicorn.error")

    response = client.post("/predict", json=VALID_REQUEST)

    assert response.status_code == 503
    assert _counter_value(
        HTTP_FAILURES,
        "/predict",
        "503",
        "model_unavailable",
    ) == before_failures + 1
    events = [
        json.loads(record.message)
        for record in caplog.records
        if record.name == "uvicorn.error"
        and json.loads(record.message).get("event") == "api_request"
    ]
    assert events[-1]["failure_reason"] == "model_unavailable"
    assert "233" not in json.dumps(events[-1])


def test_metrics_endpoint_exposes_counters_and_latency_histogram() -> None:
    """Prometheus can scrape request, failure, prediction, and latency metrics."""
    response = client.get("/metrics")

    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert "heart_disease_api_requests_total" in response.text
    assert "heart_disease_api_request_failures_total" in response.text
    assert "heart_disease_predictions_succeeded_total" in response.text
    assert "heart_disease_api_request_duration_seconds_bucket" in response.text
    assert "heart_disease_input_feature_issues_total" in response.text
