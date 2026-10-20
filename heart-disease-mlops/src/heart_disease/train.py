"""Train reproducible baseline classifiers without using the test set."""

import pickle
import sys

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

if __package__:
    from .feature_engineering import (
        RANDOM_STATE,
        FeatureEngineeringError,
        load_and_split,
        make_model_pipeline,
        project_root,
    )
else:
    from feature_engineering import (
        RANDOM_STATE,
        FeatureEngineeringError,
        load_and_split,
        make_model_pipeline,
        project_root,
    )


def build_candidate_models() -> dict[str, object]:
    """Create baseline models with fixed seeds and understandable settings."""
    return {
        "logistic_regression": make_model_pipeline(
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
        ),
        "random_forest": make_model_pipeline(
            RandomForestClassifier(
                n_estimators=300,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        ),
    }


def main() -> None:
    """Fit both candidates on identical training data and save fitted pipelines."""
    X_train, _, y_train, _, _ = load_and_split()
    models = build_candidate_models()

    for name, model in models.items():
        print(f"\nTraining {name.replace('_', ' ').title()} on training rows only...")
        model.fit(X_train, y_train)
        print(f"{name} training completed.")

    artifact_directory = project_root() / "artifacts"
    artifact_directory.mkdir(parents=True, exist_ok=True)
    model_path = artifact_directory / "candidate_models.joblib"
    try:
        joblib.dump(models, model_path)
    except (OSError, pickle.PickleError) as error:
        raise FeatureEngineeringError(
            f"Could not save candidate models to '{model_path}'. Details: {error}"
        ) from error

    print(f"\nSaved both fitted preprocessing/model pipelines to: {model_path}")
    print("The held-out test split was not used during training.")


if __name__ == "__main__":
    try:
        main()
    except FeatureEngineeringError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
