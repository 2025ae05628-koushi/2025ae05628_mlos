"""Validate and create a typed, un-imputed copy of the UCI Cleveland data."""

import sys

import pandas as pd

if __package__:
    from .data_acquisition import (
        COLUMNS,
        DATASET_FILENAME,
        DatasetAcquisitionError,
        load_dataset,
        project_root,
    )
else:
    from data_acquisition import (
        COLUMNS,
        DATASET_FILENAME,
        DatasetAcquisitionError,
        load_dataset,
        project_root,
    )


TARGET_COLUMN = "num"
INTEGER_COLUMNS = [column for column in COLUMNS if column != "oldpeak"]
DOCUMENTED_CATEGORIES = {
    "sex": {0, 1},
    "cp": {1, 2, 3, 4},
    "fbs": {0, 1},
    "restecg": {0, 1, 2},
    "exang": {0, 1},
    "slope": {1, 2, 3},
    "ca": {0, 1, 2, 3},
    "thal": {3, 6, 7},
    "num": {0, 1, 2, 3, 4},
}
OUTPUT_FILENAME = "heart_disease_cleaned.csv"


def print_summary(title: str, dataset: pd.DataFrame) -> None:
    """Print comparable quality information before or after type correction."""
    print(f"\n{title}")
    print(f"Shape (rows, columns): {dataset.shape}")
    print(f"Data types:\n{dataset.dtypes.to_string()}")
    print(f"Missing values by column:\n{dataset.isna().sum().to_string()}")
    print(f"Duplicate rows: {dataset.duplicated().sum()}")
    print(
        f"Target '{TARGET_COLUMN}' values:\n"
        f"{dataset[TARGET_COLUMN].value_counts(dropna=False).sort_index().to_string()}"
    )


def validate_documented_values(dataset: pd.DataFrame) -> None:
    """Report documented category-code violations and stop rather than guessing."""
    invalid_found = False
    print("\nChecking documented categorical and target values:")
    for column, allowed_values in DOCUMENTED_CATEGORIES.items():
        observed_values = set(dataset[column].dropna().unique())
        invalid_values = observed_values - allowed_values
        if invalid_values:
            invalid_found = True
            print(f"- {column}: unexpected values {sorted(invalid_values)}")
        else:
            print(f"- {column}: no unexpected values")

    if invalid_found:
        raise DatasetAcquisitionError(
            "The raw data contains undocumented category or target values. "
            "No cleaned file was written; inspect the UCI field documentation "
            "before deciding how to handle them."
        )


def normalize_integer_types(dataset: pd.DataFrame) -> pd.DataFrame:
    """Convert integer-valued fields to nullable integers without filling nulls."""
    cleaned = dataset.copy()
    for column in COLUMNS:
        try:
            numeric_values = pd.to_numeric(cleaned[column], errors="raise")
        except (TypeError, ValueError) as error:
            raise DatasetAcquisitionError(
                f"Column '{column}' contains values that are not numeric. "
                f"Details: {error}"
            ) from error
        cleaned[column] = numeric_values

    for column in INTEGER_COLUMNS:
        non_missing = cleaned[column].dropna()
        if not non_missing.mod(1).eq(0).all():
            raise DatasetAcquisitionError(
                f"Column '{column}' contains non-integer values but is expected "
                "to contain integer-valued measurements or codes."
            )
        cleaned[column] = cleaned[column].astype("Int64")

    return cleaned


def main() -> None:
    """Validate the raw source and save a separate cleaned CSV copy."""
    root = project_root()
    raw_path = root / "data" / "raw" / DATASET_FILENAME
    output_directory = root / "data" / "processed"
    output_path = output_directory / OUTPUT_FILENAME

    if not raw_path.is_file():
        raise DatasetAcquisitionError(
            f"Raw dataset not found: {raw_path}. Run the Step 2 acquisition "
            "script before preprocessing."
        )

    raw_dataset = load_dataset(raw_path)
    print_summary("BEFORE: original raw dataset (read only)", raw_dataset)
    validate_documented_values(raw_dataset)

    duplicate_count = int(raw_dataset.duplicated().sum())
    if duplicate_count:
        print(
            "\nExact duplicate rows were found and will be preserved. The source "
            "has no patient identifier to establish whether identical clinical "
            "records represent the same person, so this script will not delete "
            "them automatically."
        )
    else:
        print(
            "\nNo exact duplicate rows were found; there are no duplicate rows "
            "to remove."
        )

    missing_count = int(raw_dataset.isna().sum().sum())
    if missing_count:
        print(
            f"\nThere are {missing_count} missing feature values. They will stay "
            "missing in this copy. Imputation statistics should be learned from "
            "the training split only during a later modeling step to avoid "
            "leakage from validation or test data."
        )
    else:
        print("\nNo missing values were found; no imputation is needed.")

    cleaned_dataset = normalize_integer_types(raw_dataset)
    output_directory.mkdir(parents=True, exist_ok=True)
    try:
        cleaned_dataset.to_csv(output_path, index=False, na_rep="")
    except OSError as error:
        raise DatasetAcquisitionError(
            f"Could not save cleaned dataset to '{output_path}'. "
            f"Check folder permissions and available disk space. Details: {error}"
        ) from error

    print_summary("AFTER: cleaned copy (raw source is unchanged)", cleaned_dataset)
    print(
        "\nCleaning decisions: documented codes were validated; integer-valued "
        "columns were given nullable integer types; missing values, duplicate "
        "rows (if any), feature values, and target labels were not silently "
        "changed."
    )
    print(f"\nCleaned dataset saved to: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except DatasetAcquisitionError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
