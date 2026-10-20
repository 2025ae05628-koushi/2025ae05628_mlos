"""Create a reproducible, leakage-safe split and preprocessing pipeline."""

import sys
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "num"
DATASET_FILENAME = "heart_disease_cleaned.csv"
EXPECTED_COLUMNS = [
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
    TARGET_COLUMN,
]
NUMERICAL_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal",
]
RANDOM_STATE = 42
TEST_SIZE = 0.2


class FeatureEngineeringError(Exception):
    """Explain why the cleaned dataset cannot be used for training."""


def project_root() -> Path:
    """Return the project root regardless of the current working directory."""
    return Path(__file__).resolve().parents[2]


def load_cleaned_dataset(dataset_path: Path | None = None) -> pd.DataFrame:
    """Load and validate Step 3's cleaned Cleveland CSV."""
    if dataset_path is None:
        dataset_path = project_root() / "data" / "processed" / DATASET_FILENAME
    if not dataset_path.is_file():
        raise FeatureEngineeringError(
            f"Cleaned dataset not found: {dataset_path}. Run the Step 3 "
            "preprocessing script first."
        )
    try:
        dataset = pd.read_csv(dataset_path)
    except (OSError, pd.errors.ParserError, ValueError) as error:
        raise FeatureEngineeringError(
            f"Could not read cleaned dataset '{dataset_path}'. Details: {error}"
        ) from error

    if dataset.shape != (303, len(EXPECTED_COLUMNS)):
        raise FeatureEngineeringError(
            f"Expected the Step 2/3 Cleveland dataset with shape (303, 14); "
            f"found {dataset.shape}. Check the input file before training."
        )
    if dataset.columns.tolist() != EXPECTED_COLUMNS:
        raise FeatureEngineeringError(
            f"Expected columns {EXPECTED_COLUMNS}; found {dataset.columns.tolist()}."
        )
    if dataset[TARGET_COLUMN].isna().any():
        raise FeatureEngineeringError(
            f"Target column '{TARGET_COLUMN}' contains missing values. "
            "Do not guess target labels; inspect the source data."
        )
    try:
        target_values = set(pd.to_numeric(dataset[TARGET_COLUMN], errors="raise").unique())
    except (TypeError, ValueError) as error:
        raise FeatureEngineeringError(
            f"Target '{TARGET_COLUMN}' must contain numeric UCI labels. Details: {error}"
        ) from error
    if not target_values or not target_values.issubset({0, 1, 2, 3, 4}):
        raise FeatureEngineeringError(
            f"Unexpected values in target '{TARGET_COLUMN}': {sorted(target_values)}. "
            "Expected the documented UCI values 0 through 4."
        )

    return dataset


def split_features_target(
    dataset: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Keep every feature and map documented disease severity to binary labels."""
    features = dataset.drop(columns=[TARGET_COLUMN])
    source_target = dataset[TARGET_COLUMN].astype(int)
    target = (source_target > 0).astype("int8")
    target.name = "disease_present"

    print(f"Target source column: {TARGET_COLUMN}")
    print("Verified target interpretation: 0 = absence; values 1 through 4 = presence.")
    print(
        "Explicit binary target mapping: num == 0 -> 0 (absence), "
        "num > 0 -> 1 (presence)."
    )
    print(
        "Original target counts:\n"
        f"{source_target.value_counts().sort_index().to_string()}"
    )
    print(
        "Binary target counts:\n"
        f"{target.value_counts().sort_index().rename(index={0: 'absence', 1: 'presence'}).to_string()}"
    )
    print(f"Numerical features to scale: {NUMERICAL_FEATURES}")
    print(f"Categorical code features to encode: {CATEGORICAL_FEATURES}")
    print("No input features are removed.")

    return features, target, source_target


def split_data(
    features: pd.DataFrame, target: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Make a fixed, stratified holdout split before fitting preprocessing."""
    if target.nunique() != 2 or target.value_counts().min() < 2:
        raise FeatureEngineeringError(
            "A stratified binary train/test split requires at least two examples "
            "of each target class."
        )
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=target,
        )
    except ValueError as error:
        raise FeatureEngineeringError(
            f"Could not create the stratified train/test split. Details: {error}"
        ) from error

    print(
        f"\nStratified split: {len(X_train)} training rows and "
        f"{len(X_test)} held-out test rows (test_size={TEST_SIZE}, "
        f"random_state={RANDOM_STATE})."
    )
    print(
        "Training class counts:\n"
        f"{y_train.value_counts().sort_index().rename(index={0: 'absence', 1: 'presence'}).to_string()}"
    )
    print(
        "Test class counts:\n"
        f"{y_test.value_counts().sort_index().rename(index={0: 'absence', 1: 'presence'}).to_string()}"
    )
    return X_train, X_test, y_train, y_test


def build_preprocessor() -> ColumnTransformer:
    """Build transformations; fitting happens only inside a training pipeline."""
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "one_hot_encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numerical", numerical_pipeline, NUMERICAL_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def make_model_pipeline(classifier: Any) -> Pipeline:
    """Combine training-only preprocessing with one classifier."""
    return Pipeline(
        steps=[
            ("preprocessing", build_preprocessor()),
            ("classifier", classifier),
        ]
    )


def load_and_split() -> tuple[
    pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame
]:
    """Load verified data, describe features, and make the reproducible split."""
    dataset = load_cleaned_dataset()
    features, target, _ = split_features_target(dataset)
    X_train, X_test, y_train, y_test = split_data(features, target)
    return X_train, X_test, y_train, y_test, dataset


if __name__ == "__main__":
    try:
        X_train, X_test, y_train, y_test, _ = load_and_split()
        print(
            f"\nFeature engineering check passed: X_train={X_train.shape}, "
            f"X_test={X_test.shape}, y_train={y_train.shape}, y_test={y_test.shape}."
        )
        print(
            "Imputers, encoder, and scaler are defined in a Pipeline and will "
            "learn their settings from training rows only."
        )
    except FeatureEngineeringError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
