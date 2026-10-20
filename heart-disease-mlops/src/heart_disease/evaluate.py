"""Cross-validate training data and evaluate baseline models on the holdout."""

import json
import sys
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

if __package__:
    from .feature_engineering import (
        RANDOM_STATE,
        FeatureEngineeringError,
        load_and_split,
        project_root,
    )
    from .train import build_candidate_models
else:
    from feature_engineering import (
        RANDOM_STATE,
        FeatureEngineeringError,
        load_and_split,
        project_root,
    )
    from train import build_candidate_models


CV_FOLDS = 5
SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "roc_auc": "roc_auc",
}


def json_safe(value: Any) -> Any:
    """Convert NumPy values and nested containers to JSON-compatible values."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> None:
    """Run training-only cross-validation, then one final holdout evaluation."""
    X_train, X_test, y_train, y_test, dataset = load_and_split()
    candidate_models = build_candidate_models()
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cross_validation_results: dict[str, dict[str, Any]] = {}

    print(
        f"\nRunning {CV_FOLDS}-fold stratified cross-validation on training rows "
        "only. Each fold fits its own preprocessing pipeline."
    )
    for name, model in candidate_models.items():
        scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=SCORING,
            return_train_score=False,
            n_jobs=1,
            error_score="raise",
        )
        cross_validation_results[name] = {}
        print(f"\n{name.replace('_', ' ').title()} cross-validation:")
        for metric in SCORING:
            fold_scores = scores[f"test_{metric}"]
            mean_score = float(np.mean(fold_scores))
            std_score = float(np.std(fold_scores, ddof=1))
            cross_validation_results[name][metric] = {
                "fold_scores": fold_scores.tolist(),
                "mean": mean_score,
                "standard_deviation": std_score,
            }
            print(f"- {metric}: {mean_score:.3f} +/- {std_score:.3f}")

    selected_name = max(
        cross_validation_results,
        key=lambda name: cross_validation_results[name]["roc_auc"]["mean"],
    )
    print(
        "\nModel-selection rule: highest mean training-only cross-validation "
        f"ROC-AUC. Selected: {selected_name.replace('_', ' ').title()}."
    )
    print(
        "The test set was not used to select or tune the model. Now performing "
        "the final test-set evaluation once for each baseline."
    )

    artifact_directory = project_root() / "artifacts"
    candidate_path = artifact_directory / "candidate_models.joblib"
    if not candidate_path.is_file():
        raise FeatureEngineeringError(
            f"Trained candidate model file not found: {candidate_path}. "
            "Run src\\heart_disease\\train.py first."
        )
    try:
        fitted_models = joblib.load(candidate_path)
    except (OSError, EOFError, ValueError, KeyError) as error:
        raise FeatureEngineeringError(
            f"Could not load trained models from '{candidate_path}'. Details: {error}"
        ) from error
    if set(fitted_models) != set(candidate_models):
        raise FeatureEngineeringError(
            "Saved model candidates do not match the current training script. "
            "Run the training script again before evaluating."
        )

    final_metrics: dict[str, Any] = {}
    reports_directory = project_root() / "reports"
    figures_directory = reports_directory / "figures"
    figures_directory.mkdir(parents=True, exist_ok=True)

    for name in candidate_models:
        model = fitted_models[name]
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]
        confusion = confusion_matrix(y_test, predictions, labels=[0, 1])
        final_metrics[name] = {
            "test_accuracy": float(accuracy_score(y_test, predictions)),
            "test_precision": float(
                precision_score(y_test, predictions, zero_division=0)
            ),
            "test_recall": float(recall_score(y_test, predictions, zero_division=0)),
            "test_roc_auc": float(roc_auc_score(y_test, probabilities)),
            "confusion_matrix_labels": ["absence (0)", "presence (1)"],
            "confusion_matrix": confusion.tolist(),
            "classification_report": classification_report(
                y_test,
                predictions,
                labels=[0, 1],
                target_names=["absence", "presence"],
                output_dict=True,
                zero_division=0,
            ),
        }
        print(f"\n{name.replace('_', ' ').title()} held-out test metrics:")
        print(f"- accuracy: {final_metrics[name]['test_accuracy']:.3f}")
        print(f"- precision: {final_metrics[name]['test_precision']:.3f}")
        print(f"- recall: {final_metrics[name]['test_recall']:.3f}")
        print(f"- ROC-AUC: {final_metrics[name]['test_roc_auc']:.3f}")
        print(f"- confusion matrix (rows=true, columns=predicted):\n{confusion}")
        print(
            classification_report(
                y_test,
                predictions,
                labels=[0, 1],
                target_names=["absence", "presence"],
                zero_division=0,
            )
        )

        display = ConfusionMatrixDisplay(
            confusion_matrix=confusion,
            display_labels=["absence", "presence"],
        )
        display.plot(cmap="Blues", values_format="d")
        display.ax_.set_title(
            f"{name.replace('_', ' ').title()} - Held-out Test Set"
        )
        figure_path = (
            figures_directory / f"{name}_test_confusion_matrix.png"
        )
        display.figure_.savefig(figure_path, dpi=150, bbox_inches="tight")
        plt.close(display.figure_)
        final_metrics[name]["confusion_matrix_figure"] = str(figure_path)

    selected_model_path = artifact_directory / "selected_model.joblib"
    try:
        joblib.dump(fitted_models[selected_name], selected_model_path)
    except OSError as error:
        raise FeatureEngineeringError(
            f"Could not save selected model to '{selected_model_path}'. "
            f"Details: {error}"
        ) from error

    results = {
        "dataset": "UCI processed Cleveland Heart Disease dataset",
        "dataset_rows": int(len(dataset)),
        "target_source_column": "num",
        "target_mapping": {
            "0": "absence of diagnosed disease",
            "1": "presence of disease (original num values 1 through 4)",
        },
        "target_counts_original": {
            str(key): int(value)
            for key, value in dataset["num"].value_counts().sort_index().items()
        },
        "binary_target_counts": {
            str(int(key > 0)): int(value)
            for key, value in dataset["num"].map(lambda value: int(value > 0))
            .value_counts()
            .sort_index()
            .items()
        },
        "split": {
            "test_size": 0.2,
            "random_state": RANDOM_STATE,
            "cross_validation_folds": CV_FOLDS,
            "stratified": True,
            "selection_metric": "mean training-only cross-validation ROC-AUC",
        },
        "features": {
            "numerical": ["age", "trestbps", "chol", "thalach", "oldpeak"],
            "categorical": [
                "sex",
                "cp",
                "fbs",
                "restecg",
                "exang",
                "slope",
                "ca",
                "thal",
            ],
            "removed": [],
        },
        "preprocessing": {
            "numerical": "median imputation then standard scaling",
            "categorical": "most-frequent imputation then one-hot encoding",
            "fit_scope": "training folds for CV; full training split for final fitted models",
        },
        "cross_validation_training_only": cross_validation_results,
        "held_out_test_metrics": final_metrics,
        "selected_model": selected_name,
        "selected_model_artifact": str(selected_model_path),
        "note": (
            "Test metrics are for final reporting only; model selection uses "
            "training-only cross-validation ROC-AUC."
        ),
    }
    metrics_path = reports_directory / "model_evaluation.json"
    try:
        metrics_path.write_text(
            json.dumps(json_safe(results), indent=2),
            encoding="utf-8",
        )
    except OSError as error:
        raise FeatureEngineeringError(
            f"Could not save evaluation results to '{metrics_path}'. "
            f"Details: {error}"
        ) from error

    print(f"\nSelected fitted pipeline saved to: {selected_model_path}")
    print(f"Evaluation details saved to: {metrics_path}")
    print(
        "\nMetric guide: accuracy is the overall correct fraction; precision "
        "measures how often predicted presence is correct; recall measures how "
        "many actual presence cases are found; ROC-AUC measures ranking across "
        "thresholds (1 is best, 0.5 is chance). CV standard deviation describes "
        "variation between folds. These metrics do not establish clinical utility."
    )


if __name__ == "__main__":
    try:
        main()
    except FeatureEngineeringError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
