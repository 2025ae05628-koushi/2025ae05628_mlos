"""Package the fitted Step 4 winner without training or replacing source files."""

from __future__ import annotations

import hashlib
import json
import math
import pickle
import platform
import re
import sys
import warnings
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from shutil import copyfileobj, copystat
from typing import Any

import joblib
import sklearn
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

if __package__:
    from .feature_engineering import (
        CATEGORICAL_FEATURES,
        EXPECTED_COLUMNS,
        NUMERICAL_FEATURES,
        TARGET_COLUMN,
        project_root,
    )
else:
    from feature_engineering import (
        CATEGORICAL_FEATURES,
        EXPECTED_COLUMNS,
        NUMERICAL_FEATURES,
        TARGET_COLUMN,
        project_root,
    )


PACKAGED_MODEL_FILENAME = "final_heart_disease_pipeline.joblib"
METADATA_FILENAME = "final_heart_disease_metadata.json"
EXPERIMENT_NAME = "heart-disease-classification"

_MODEL_CLASSES = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
}


class ModelPackagingError(Exception):
    """Explain why the selected fitted pipeline could not be packaged."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        for block in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _package_versions() -> dict[str, str]:
    packages = ("joblib", "mlflow", "numpy", "pandas", "scikit-learn")
    versions: dict[str, str] = {"python": platform.python_version()}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def _serialized_sklearn_version(
    load_warnings: list[warnings.WarningMessage],
) -> str:
    for warning in load_warnings:
        if isinstance(warning.message, InconsistentVersionWarning):
            original_version = getattr(
                warning.message, "original_sklearn_version", None
            )
            if original_version:
                return str(original_version)
            match = re.search(
                r"from version ([^\s]+) when using version",
                str(warning.message),
            )
            if match:
                return match.group(1)
    return sklearn.__version__


def _find_selected_mlflow_run(
    root: Path,
    selected_model: str,
    metrics: dict[str, float],
) -> tuple[str | None, str | None]:
    database = root / "mlflow.db"
    if not database.is_file():
        return None, "No local MLflow database was present during packaging."

    tracking_uri = f"sqlite:///{database.as_posix()}"
    try:
        client = MlflowClient(tracking_uri=tracking_uri)
        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment is None:
            return None, f"MLflow experiment '{EXPERIMENT_NAME}' was not found."
        runs = client.search_runs(
            [experiment.experiment_id],
            max_results=1000,
            order_by=["attributes.start_time DESC"],
        )
    except (MlflowException, OSError, ValueError) as error:
        return None, f"Could not read the local MLflow run history: {error}"

    for run in runs:
        if (
            run.info.status != "FINISHED"
            or run.data.params.get("model_name") != selected_model
            or run.data.tags.get("selection_outcome") != "selected"
        ):
            continue
        expected_run_metrics = {
            "cv_roc_auc_mean": metrics["cv_roc_auc_mean"],
            **{
                metric_name: value
                for metric_name, value in metrics.items()
                if metric_name.startswith("test_")
            },
        }
        if all(
            metric_name in run.data.metrics
            and math.isclose(
                run.data.metrics[metric_name],
                value,
                rel_tol=1e-9,
                abs_tol=1e-12,
            )
            for metric_name, value in expected_run_metrics.items()
        ):
            return run.info.run_id, None

    return (
        None,
        "No finished selected MLflow run with matching Step 4 metrics was found.",
    )


def main() -> None:
    """Copy the existing selected fitted pipeline and record its provenance."""
    root = project_root()
    artifact_directory = root / "artifacts"
    source_path = artifact_directory / "selected_model.joblib"
    packaged_path = artifact_directory / PACKAGED_MODEL_FILENAME
    metadata_path = artifact_directory / METADATA_FILENAME
    evaluation_path = root / "reports" / "model_evaluation.json"

    if not source_path.is_file():
        raise ModelPackagingError(
            f"The selected fitted Step 4 pipeline is missing: {source_path}. "
            "Restore or create the Step 4 selected model before packaging; "
            "this script will not retrain it."
        )
    if not evaluation_path.is_file():
        raise ModelPackagingError(
            f"Step 4 evaluation report is missing: {evaluation_path}. "
            "Restore the report before packaging so the selected model and "
            "metrics can be verified."
        )
    if packaged_path.exists() or metadata_path.exists():
        if packaged_path.is_file() and metadata_path.is_file():
            try:
                existing_metadata = json.loads(
                    metadata_path.read_text(encoding="utf-8")
                )
                source_hash = _sha256(source_path)
                if (
                    existing_metadata.get("artifact") == PACKAGED_MODEL_FILENAME
                    and existing_metadata.get("source_artifact_sha256") == source_hash
                    and existing_metadata.get("artifact_sha256") == source_hash
                    and _sha256(packaged_path) == source_hash
                ):
                    print(
                        "The packaged model already exists and matches the "
                        f"selected Step 4 artifact: {packaged_path}"
                    )
                    print(f"Metadata: {metadata_path}")
                    print(f"Verified SHA-256: {source_hash}")
                    return
            except (OSError, json.JSONDecodeError) as error:
                raise ModelPackagingError(
                    f"Could not verify existing package metadata "
                    f"'{metadata_path}': {error}"
                ) from error
        existing = [
            str(path)
            for path in (packaged_path, metadata_path)
            if path.exists()
        ]
        raise ModelPackagingError(
            "Existing packaged output does not match the selected source. "
            "Refusing to overwrite: "
            + ", ".join(existing)
            + ". Inspect those specific files before proceeding."
        )

    try:
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ModelPackagingError(
            f"Could not read Step 4 evaluation report '{evaluation_path}': {error}"
        ) from error

    selected_model = evaluation.get("selected_model")
    model_class = _MODEL_CLASSES.get(selected_model)
    if model_class is None:
        raise ModelPackagingError(
            f"Unexpected selected model in Step 4 report: {selected_model!r}."
        )

    try:
        with warnings.catch_warnings(record=True) as load_warnings:
            warnings.simplefilter("always", InconsistentVersionWarning)
            pipeline = joblib.load(source_path)
    except (OSError, EOFError, ValueError, pickle.UnpicklingError, ImportError) as error:
        raise ModelPackagingError(
            f"Could not load the existing selected pipeline '{source_path}': {error}"
        ) from error

    if not isinstance(pipeline, Pipeline):
        raise ModelPackagingError(
            f"Expected a fitted scikit-learn Pipeline in '{source_path}', "
            f"found {type(pipeline).__name__}."
        )
    if list(getattr(pipeline, "feature_names_in_", [])) != EXPECTED_COLUMNS[:-1]:
        raise ModelPackagingError(
            "The saved pipeline feature names/order do not match the verified "
            "Step 2/3 feature schema."
        )
    if not {"preprocessing", "classifier"}.issubset(pipeline.named_steps):
        raise ModelPackagingError(
            "The saved pipeline must contain both 'preprocessing' and "
            "'classifier' steps."
        )

    preprocessor = pipeline.named_steps["preprocessing"]
    classifier = pipeline.named_steps["classifier"]
    if not isinstance(classifier, model_class):
        raise ModelPackagingError(
            f"Step 4 selected '{selected_model}', but the saved pipeline "
            f"contains {type(classifier).__name__}."
        )
    if not hasattr(preprocessor, "transformers_") or not hasattr(
        classifier, "classes_"
    ):
        raise ModelPackagingError(
            "The selected model or preprocessing step is not fitted. "
            "The packaging script will not fit it."
        )
    if list(classifier.classes_) != [0, 1]:
        raise ModelPackagingError(
            f"Expected fitted target classes [0, 1], found "
            f"{list(classifier.classes_)}."
        )

    try:
        cv_auc = float(
            evaluation["cross_validation_training_only"][selected_model]["roc_auc"][
                "mean"
            ]
        )
        test_metrics = {
            name: float(value)
            for name, value in evaluation["held_out_test_metrics"][
                selected_model
            ].items()
            if name.startswith("test_")
        }
        classifier_parameters = {
            name: value
            for name, value in classifier.get_params(deep=False).items()
            if value is None or isinstance(value, (str, int, float, bool))
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ModelPackagingError(
            "The Step 4 report is missing the selected model's actual CV or "
            f"test metrics: {error}"
        ) from error

    if not test_metrics:
        raise ModelPackagingError(
            f"No held-out test metrics were recorded for '{selected_model}'."
        )

    selected_metrics = {"cv_roc_auc_mean": cv_auc, **test_metrics}
    mlflow_run_id, mlflow_note = _find_selected_mlflow_run(
        root,
        selected_model,
        selected_metrics,
    )
    if mlflow_note:
        print(f"MLflow provenance note: {mlflow_note}")

    source_hash = _sha256(source_path)
    metadata: dict[str, Any] = {
        "artifact": PACKAGED_MODEL_FILENAME,
        "artifact_format": "joblib serialized scikit-learn Pipeline",
        "artifact_sha256": source_hash,
        "source_artifact": "artifacts/selected_model.joblib",
        "source_artifact_sha256": source_hash,
        "selected_model_name": selected_model,
        "model_version": classifier_parameters,
        "reference_successful_mlflow_run_id": mlflow_run_id,
        "mlflow_run_relationship": (
            "The artifact bytes are copied from the selected Step 4 pipeline. "
            "The optional MLflow run ID identifies the newest finished, selected "
            "run with matching Step 4 metrics; it is lineage, not the source file."
        ),
        "dataset": evaluation.get("dataset"),
        "selection_rule": evaluation["split"]["selection_metric"],
        "selection_metric": "mean training-only cross-validation ROC-AUC",
        "selection_score": cv_auc,
        "held_out_test_metrics": test_metrics,
        "required_features_in_order": EXPECTED_COLUMNS[:-1],
        "feature_types": {
            "numeric": NUMERICAL_FEATURES,
            "categorical_integer_codes": CATEGORICAL_FEATURES,
        },
        "target_encoding": {
            "source_column": TARGET_COLUMN,
            "source_values": {
                "0": "absence",
                "1-4": "presence",
            },
            "model_classes": {
                "0": "absence",
                "1": "presence",
            },
        },
        "preprocessing": {
            "numeric": "median imputation, then standard scaling",
            "categorical": "most-frequent imputation, then one-hot encoding",
            "stored_with_classifier": True,
            "prediction_refits_preprocessing": False,
        },
        "package_versions": _package_versions(),
        "serialized_scikit_learn_version": _serialized_sklearn_version(
            load_warnings
        ),
    }

    packaged_created = False
    metadata_created = False
    try:
        with source_path.open("rb") as source_file, packaged_path.open("xb") as output_file:
            packaged_created = True
            copyfileobj(source_file, output_file)
        copystat(source_path, packaged_path)
        if _sha256(packaged_path) != source_hash:
            raise OSError("The packaged copy did not match the source SHA-256.")
        with metadata_path.open("x", encoding="utf-8") as metadata_file:
            metadata_created = True
            json.dump(metadata, metadata_file, indent=2)
            metadata_file.write("\n")
    except OSError as error:
        if metadata_created and metadata_path.is_file():
            metadata_path.unlink()
        if packaged_created and packaged_path.is_file():
            packaged_path.unlink()
        raise ModelPackagingError(
            f"Could not save the packaged model and metadata: {error}"
        ) from error

    print(f"Selected model: {selected_model}")
    print(f"Training-only CV ROC-AUC: {cv_auc:.6f}")
    print(f"Reference successful MLflow run: {mlflow_run_id or 'not available'}")
    print(f"Packaged fitted pipeline: {packaged_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Verified source/copy SHA-256: {source_hash}")
    for warning in load_warnings:
        print(f"Compatibility warning while reading source artifact: {warning.message}")


if __name__ == "__main__":
    try:
        main()
    except ModelPackagingError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
