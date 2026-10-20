"""FastAPI application serving predictions from the packaged fitted pipeline."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from functools import lru_cache
from typing import Annotated, Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt
from sklearn.pipeline import Pipeline
from starlette.requests import Request

from . import predict as prediction
from .monitoring import (
    HTTP_FAILURES,
    HTTP_REQUESTS,
    PREDICTIONS_SUCCEEDED,
    REQUEST_DURATION,
    observe_feature_values,
    observe_validation_errors,
)

StrictNumber = StrictFloat | StrictInt
_LOGGER = logging.getLogger("uvicorn.error")
_KNOWN_METHODS = {"DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"}


class PredictionRequest(BaseModel):
    """One Cleveland example with the exact feature names expected by the model."""

    model_config = ConfigDict(extra="forbid")

    age: Annotated[StrictNumber | None, Field()]
    sex: Annotated[StrictInt | None, Field()]
    cp: Annotated[StrictInt | None, Field()]
    trestbps: Annotated[StrictNumber | None, Field()]
    chol: Annotated[StrictNumber | None, Field()]
    fbs: Annotated[StrictInt | None, Field()]
    restecg: Annotated[StrictInt | None, Field()]
    thalach: Annotated[StrictNumber | None, Field()]
    exang: Annotated[StrictInt | None, Field()]
    oldpeak: Annotated[StrictNumber | None, Field()]
    slope: Annotated[StrictInt | None, Field()]
    ca: Annotated[StrictInt | None, Field()]
    thal: Annotated[StrictInt | None, Field()]


@lru_cache(maxsize=1)
def _load_model() -> tuple[Pipeline, dict[str, Any]]:
    """Load and validate the packaged pipeline once per API process."""
    return prediction._load_pipeline()


app = FastAPI(
    title="Heart Disease Prediction API",
    description=(
        "Educational predictions from the saved Cleveland heart-disease model. "
        "Outputs are not a medical diagnosis."
    ),
    version="1.0.0",
)


def _route_label(request: Request) -> str:
    """Return a bounded route label instead of an arbitrary request path."""
    route = request.scope.get("route")
    route_path = getattr(route, "path", None)
    return route_path if isinstance(route_path, str) else "unmatched"


def _method_label(method: str) -> str:
    """Keep metric labels bounded even for arbitrary HTTP method strings."""
    return method if method in _KNOWN_METHODS else "OTHER"


def _log_request(
    request: Request,
    route: str,
    status_code: int,
    latency_seconds: float,
    failure_reason: str | None,
) -> None:
    event: dict[str, Any] = {
        "event": "api_request",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": request.state.request_id,
        "method": request.method,
        "route": route,
        "status_code": status_code,
        "response_latency_ms": round(latency_seconds * 1000, 3),
    }
    if failure_reason:
        event["failure_reason"] = failure_reason
    prediction_output = getattr(request.state, "prediction_output", None)
    if prediction_output:
        event["predicted_class"] = prediction_output["predicted_class"]
        event["predicted_label"] = prediction_output["predicted_label"]
    _LOGGER.info(json.dumps(event, separators=(",", ":")))


@app.middleware("http")
async def monitor_requests(
    request: Request,
    call_next: Any,
) -> Response:
    """Record bounded request metrics and privacy-safe completion logs."""
    request.state.request_id = uuid4().hex
    started_at = time.perf_counter()
    method = _method_label(request.method)
    try:
        response = await call_next(request)
    except Exception:
        latency = time.perf_counter() - started_at
        route = _route_label(request)
        HTTP_REQUESTS.labels(
            method=method,
            route=route,
            status_code="500",
        ).inc()
        HTTP_FAILURES.labels(
            route=route,
            status_code="500",
            reason="unhandled_exception",
        ).inc()
        REQUEST_DURATION.labels(method=method, route=route).observe(latency)
        _log_request(request, route, 500, latency, "unhandled_exception")
        raise

    latency = time.perf_counter() - started_at
    route = _route_label(request)
    status_code = response.status_code
    status_label = str(status_code)
    HTTP_REQUESTS.labels(
        method=method,
        route=route,
        status_code=status_label,
    ).inc()
    REQUEST_DURATION.labels(method=method, route=route).observe(latency)

    failure_reason = getattr(request.state, "monitor_failure_reason", None)
    if status_code >= 400:
        failure_reason = failure_reason or "http_error"
        HTTP_FAILURES.labels(
            route=route,
            status_code=status_label,
            reason=failure_reason,
        ).inc()
    _log_request(request, route, status_code, latency, failure_reason)
    return response


@app.exception_handler(RequestValidationError)
async def monitor_validation_errors(
    request: Request,
    exception: RequestValidationError,
) -> JSONResponse:
    """Count invalid inputs without logging the submitted values."""
    request.state.monitor_failure_reason = "invalid_input"
    observe_validation_errors(exception.errors())
    return JSONResponse(
        status_code=422,
        content={"detail": jsonable_encoder(exception.errors())},
    )


@app.get("/health")
def health() -> dict[str, str]:
    """Report that the API process is responding."""
    return {"status": "ok"}


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    """Expose Prometheus counters and latency histogram for local scraping."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
def predict(
    request: PredictionRequest,
    raw_request: Request,
) -> dict[str, object]:
    """Return the saved model's prediction for one validated feature row."""
    example = request.model_dump()
    observe_feature_values(example)
    try:
        prediction._validate_input(example)
    except prediction.ModelPredictionError as error:
        raw_request.state.monitor_failure_reason = "invalid_input"
        raise HTTPException(status_code=422, detail=str(error)) from error

    try:
        model, metadata = _load_model()
    except prediction.ModelPredictionError as error:
        raw_request.state.monitor_failure_reason = "model_unavailable"
        raise HTTPException(
            status_code=503,
            detail=f"The packaged prediction model is unavailable: {error}",
        ) from error

    try:
        result = prediction.predict_with_pipeline(
            example,
            "API request",
            model,
            metadata,
        )
        PREDICTIONS_SUCCEEDED.inc()
        raw_request.state.prediction_output = result
        return result
    except prediction.ModelPredictionError as error:
        raw_request.state.monitor_failure_reason = "prediction_error"
        raise HTTPException(
            status_code=500,
            detail=f"The saved model could not produce a prediction: {error}",
        ) from error
