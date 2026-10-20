"""Log the existing baseline training/evaluation workflow to local MLflow."""

import json
import sys
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
from mlflow.exceptions import MlflowException
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

if __package__:
    from .evaluate import CV_FOLDS, SCORING
    from .feature_engineering import (
        RANDOM_STATE,
        TEST_SIZE,
        FeatureEngineeringError,
        load_and_split,
        project_root,
    )
    from .train import build_candidate_models
else:
    from evaluate import CV_FOLDS, SCORING
    from feature_engineering import (
        RANDOM_STATE,
        TEST_SIZE,
        FeatureEngineeringError,
        load_and_split,
        project_root,
    )
    from train import build_candidate_models


EXPERIMENT_NAME = "heart-disease-classification"


class ExperimentTrackingError(Exception):
    """Explain an issue while recording the experiment locally."""


def json_safe(value: Any) -> Any:
    """Convert NumPy results and containers into JSON-compatible values."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def candidate_parameters(name: str, model: Any) -> dict[str, Any]:
    """Return simple, relevant model and preprocessing settings for MLflow."""
    parameters: dict[str, Any] = {
        "model_name": name,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "cross_validation_folds": CV_FOLDS,
        "target_source_column": "num",
        "target_mapping": "num=0 -> absence; num>0 -> presence",
        "numerical_features": "age,trestbps,chol,thalach,oldpeak",
        "categorical_features": "sex,cp,fbs,restecg,exang,slope,ca,thal",
        "numerical_imputation": "median",
        "numerical_scaling": "standard",
        "categorical_imputation": "most_frequent",
        "categorical_encoding": "one_hot",
        "selection_metric": "mean training-only cross-validation ROC-AUC",
    }
    classifier_parameters = model.get_params(deep=True)
    for key, value in classifier_parameters.items():
        if not key.startswith("classifier__"):
            continue
        if value is None or isinstance(value, (str, int, float, bool)):
            parameters[key] = value
    return parameters


def main() -> None:
    """Train, select by training-only CV, evaluate once, and log fresh runs."""
    root = project_root()
    tracking_database = root / "mlflow.db"
    artifact_directory = root / "mlartifacts"
    artifact_directory.mkdir(parents=True, exist_ok=True)
    tracking_uri = f"sqlite:///{tracking_database.as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)

    try:
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment is None:
            experiment_id = mlflow.create_experiment(
                EXPERIMENT_NAME,
                artifact_location=artifact_directory.as_uri(),
            )
            experiment = mlflow.get_experiment(experiment_id)
    except (MlflowException, OSError, ValueError) as error:
        raise ExperimentTrackingError(
            f"Could not initialize the local MLflow experiment using database "
            f"'{tracking_database}'. Details: {error}"
        ) from error
    if experiment is None:
        raise ExperimentTrackingError(
            f"MLflow did not return experiment '{EXPERIMENT_NAME}' after creation."
        )

    X_train, X_test, y_train, y_test, dataset = load_and_split()
    models = build_candidate_models()
    cross_validation = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    cv_results: dict[str, dict[str, Any]] = {}

    print(f"MLflow tracking URI: {tracking_uri}")
    print(f"MLflow experiment: {EXPERIMENT_NAME} (ID {experiment.experiment_id})")
    print(
        f"\nRunning {CV_FOLDS}-fold stratified cross-validation on "
        "training data only."
    )

    for name, model in models.items():
        scores = cross_validate(
            model,
            X_train,
            y_train,
            scoring=SCORING,
            cv=cross_validation,
            return_train_score=False,
            n_jobs=1,
            error_score="raise",
        )
        cv_results[name] = {}
        for metric_name in SCORING:
            fold_scores = scores[f"test_{metric_name}"]
            cv_results[name][metric_name] = {
                "fold_scores": fold_scores.tolist(),
                "mean": float(np.mean(fold_scores)),
                "standard_deviation": float(np.std(fold_scores, ddof=1)),
            }

        model.fit(X_train, y_train)
        cv_auc = cv_results[name]["roc_auc"]["mean"]
        print(
            f"{name}: training fit complete; mean training CV ROC-AUC="
            f"{cv_auc:.3f} +/- "
            f"{cv_results[name]['roc_auc']['standard_deviation']:.3f}"
        )

    selected_name = max(
        cv_results,
        key=lambda model_name: cv_results[model_name]["roc_auc"]["mean"],
    )
    selected_score = cv_results[selected_name]["roc_auc"]["mean"]
    selection_rationale = (
        "Selected by the highest mean ROC-AUC from stratified cross-validation "
        "on training data only; the held-out test set did not determine selection. "
        f"Selected model {selected_name} achieved mean CV ROC-AUC {selected_score:.6f}."
    )
    print(
        f"\nSelected {selected_name} using training-only CV ROC-AUC "
        f"({selected_score:.3f}). Now evaluating both candidates on the "
        "held-out test set once."
    )

    test_results: dict[str, dict[str, Any]] = {}
    for name, model in models.items():
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]
        test_results[name] = {
            "test_accuracy": float(accuracy_score(y_test, predictions)),
            "test_precision": float(
                precision_score(y_test, predictions, zero_division=0)
            ),
            "test_recall": float(recall_score(y_test, predictions, zero_division=0)),
            "test_roc_auc": float(roc_auc_score(y_test, probabilities)),
            "confusion_matrix": confusion_matrix(
                y_test, predictions, labels=[0, 1]
            ).tolist(),
            "classification_report": classification_report(
                y_test,
                predictions,
                labels=[0, 1],
                target_names=["absence", "presence"],
                output_dict=True,
                zero_division=0,
            ),
        }

    reports_directory = root / "reports"
    evaluation_report = reports_directory / "model_evaluation.json"
    figures_directory = reports_directory / "figures"
    if not evaluation_report.is_file():
        raise ExperimentTrackingError(
            f"Step 4 evaluation report is missing: {evaluation_report}. "
            "Run src\\heart_disease\\evaluate.py before tracking experiments."
        )

    for name in models:
        confusion_figure = (
            figures_directory / f"{name}_test_confusion_matrix.png"
        )
        if not confusion_figure.is_file():
            raise ExperimentTrackingError(
                f"Step 4 confusion matrix is missing: {confusion_figure}. "
                "Run src\\heart_disease\\evaluate.py before tracking experiments."
            )

    tracking_summary = {
        "experiment_name": EXPERIMENT_NAME,
        "dataset": "UCI processed Cleveland Heart Disease dataset",
        "dataset_rows": int(len(dataset)),
        "target_source_column": "num",
        "target_mapping": {
            "0": "absence of diagnosed disease",
            "1": "presence of disease for original num values 1 through 4",
        },
        "binary_target_counts": {
            str(int(label)): int(count)
            for label, count in dataset["num"].map(lambda value: int(value > 0))
            .value_counts()
            .sort_index()
            .items()
        },
        "selection_rule": selection_rationale,
        "selected_model": selected_name,
        "selected_mean_training_cv_roc_auc": selected_score,
        "cross_validation_training_only": cv_results,
        "held_out_test_metrics": test_results,
    }
    summary_path = reports_directory / "mlflow_tracking_summary.json"
    try:
        summary_path.write_text(
            json.dumps(json_safe(tracking_summary), indent=2),
            encoding="utf-8",
        )
    except OSError as error:
        raise ExperimentTrackingError(
            f"Could not save the MLflow tracking summary to '{summary_path}'. "
            f"Details: {error}"
        ) from error

    print("\nLogging one new MLflow run for each model...")
    for name, model in models.items():
        run_name = name.replace("_", " ").title() + " Baseline"
        run = None
        try:
            with mlflow.start_run(
                experiment_id=experiment.experiment_id,
                run_name=run_name,
            ) as active_run:
                run = active_run
                mlflow.log_params(candidate_parameters(name, model))
                mlflow.log_metric(
                    "training_accuracy",
                    float(accuracy_score(y_train, model.predict(X_train))),
                )
                for metric_name, result in cv_results[name].items():
                    mlflow.log_metric(f"cv_{metric_name}_mean", result["mean"])
                    mlflow.log_metric(
                        f"cv_{metric_name}_std", result["standard_deviation"]
                    )
                for metric_name, value in test_results[name].items():
                    if metric_name.startswith("test_"):
                        mlflow.log_metric(metric_name, value)

                mlflow.set_tag("model_family", name)
                mlflow.set_tag("selected_model", selected_name)
                mlflow.set_tag(
                    "selection_outcome",
                    "selected" if name == selected_name else "not_selected",
                )
                mlflow.set_tag("selection_rationale", selection_rationale)

                mlflow.log_artifact(str(evaluation_report), artifact_path="reports")
                mlflow.log_artifact(str(summary_path), artifact_path="reports")
                mlflow.log_artifact(
                    str(figures_directory / f"{name}_test_confusion_matrix.png"),
                    artifact_path="reports/figures",
                )
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    serialization_format="cloudpickle",
                )
        except MlflowException as error:
            run_id = run.info.run_id if run is not None else "not created"
            raise ExperimentTrackingError(
                f"Failed to log {name} to MLflow (run {run_id}) using "
                f"'{tracking_uri}'. Details: {error}"
            ) from error

        print(f"- {run_name}: run ID {run.info.run_id}")
        print(f"  Test accuracy={test_results[name]['test_accuracy']:.3f}; "
              f"test ROC-AUC={test_results[name]['test_roc_auc']:.3f}")

    print(
        f"\nCompleted. New runs were added to experiment '{EXPERIMENT_NAME}'. "
        "Previous runs were not deleted."
    )
    print(f"Tracking database: {tracking_database}")
    print(f"Run artifacts are stored under: {artifact_directory}")
    print(f"Tracking summary: {summary_path}")


if __name__ == "__main__":
    try:
        main()
    except (ExperimentTrackingError, FeatureEngineeringError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
