"""Low-cardinality Prometheus metrics and input-quality observations."""

from __future__ import annotations

from numbers import Real
from typing import Any

from prometheus_client import Counter, Histogram

from .predict import CATEGORICAL_CODES

HTTP_REQUESTS = Counter(
    "heart_disease_api_requests_total",
    "HTTP requests handled by the heart disease API.",
    ("method", "route", "status_code"),
)
HTTP_FAILURES = Counter(
    "heart_disease_api_request_failures_total",
    "HTTP requests that ended with an error response.",
    ("route", "status_code", "reason"),
)
PREDICTIONS_SUCCEEDED = Counter(
    "heart_disease_predictions_succeeded_total",
    "Prediction requests completed successfully.",
)
REQUEST_DURATION = Histogram(
    "heart_disease_api_request_duration_seconds",
    "End-to-end HTTP request latency in seconds.",
    ("method", "route"),
)
FEATURE_ISSUES = Counter(
    "heart_disease_input_feature_issues_total",
    "Observed missing, invalid, or out-of-training-range feature values.",
    ("feature", "issue"),
)

# Inclusive empirical minima/maxima from the exact 242-row training split:
# stratified split, random_state=42, test_size=0.2. These are monitoring
# references, not medical limits or automatic rejection thresholds.
NUMERICAL_TRAINING_RANGES = {
    "age": (29.0, 77.0),
    "trestbps": (94.0, 200.0),
    "chol": (126.0, 564.0),
    "thalach": (71.0, 202.0),
    "oldpeak": (0.0, 6.2),
}

def observe_feature_values(example: dict[str, Any]) -> None:
    """Count feature-quality observations without recording feature values."""
    for feature, value in example.items():
        if value is None:
            FEATURE_ISSUES.labels(feature=feature, issue="null").inc()
        elif feature in CATEGORICAL_CODES and value not in CATEGORICAL_CODES[feature]:
            FEATURE_ISSUES.labels(feature=feature, issue="invalid_category").inc()
        elif feature in NUMERICAL_TRAINING_RANGES and isinstance(value, Real):
            lower, upper = NUMERICAL_TRAINING_RANGES[feature]
            if value < lower or value > upper:
                FEATURE_ISSUES.labels(
                    feature=feature,
                    issue="outside_training_range",
                ).inc()


def observe_validation_errors(errors: list[dict[str, Any]]) -> None:
    """Count request validation failures using only known feature labels."""
    known_features = set(NUMERICAL_TRAINING_RANGES) | set(CATEGORICAL_CODES)
    for error in errors:
        location = error.get("loc", ())
        if not location or location[0] != "body":
            continue
        field = location[1] if len(location) > 1 else None
        feature = field if isinstance(field, str) and field in known_features else "unknown"
        error_type = error.get("type", "")
        if error_type == "missing":
            issue = "missing"
        elif error_type == "extra_forbidden":
            feature = "unknown"
            issue = "extra_field"
        else:
            issue = "invalid_value"
        FEATURE_ISSUES.labels(feature=feature, issue=issue).inc()
