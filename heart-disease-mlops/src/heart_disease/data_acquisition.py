"""Download and inspect the official UCI Cleveland heart disease data."""

import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import pandas as pd

DATASET_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)
DATASET_FILENAME = "processed.cleveland.data"
COLUMNS = [
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
    "num",
]
EXPECTED_SHAPE = (303, len(COLUMNS))
EXPECTED_TARGET_VALUES = {0, 1, 2, 3, 4}


class DatasetAcquisitionError(Exception):
    """Explain a download or dataset-format error to the person running the script."""


def project_root() -> Path:
    """Return the project root, independent of the current working directory."""
    return Path(__file__).resolve().parents[2]


def download_dataset(destination: Path) -> None:
    """Download the official raw file without replacing an existing dataset."""
    temporary_path = destination.with_name(destination.name + ".download")

    try:
        with urlopen(DATASET_URL, timeout=30) as response:
            content = response.read()
        if not content:
            raise DatasetAcquisitionError(
                f"The UCI download was empty. Check the source URL: {DATASET_URL}"
            )
        temporary_path.write_bytes(content)
        temporary_path.replace(destination)
    except (URLError, TimeoutError, OSError) as error:
        if temporary_path.exists():
            temporary_path.unlink()
        raise DatasetAcquisitionError(
            "Could not download the UCI Cleveland dataset. Check your internet "
            f"connection and try again. Source: {DATASET_URL}. Details: {error}"
        ) from error

    print(f"Downloaded raw dataset from: {DATASET_URL}")


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    """Read the headerless Cleveland file without saving any modified data."""
    try:
        dataset = pd.read_csv(
            dataset_path,
            header=None,
            names=COLUMNS,
            na_values=["?"],
        )
    except (OSError, pd.errors.ParserError, ValueError) as error:
        raise DatasetAcquisitionError(
            f"Could not read dataset file '{dataset_path}'. "
            "Check that it is the original UCI processed Cleveland file. "
            f"Details: {error}"
        ) from error

    if dataset.shape != EXPECTED_SHAPE:
        raise DatasetAcquisitionError(
            f"Unexpected dataset shape {dataset.shape}; expected "
            f"{EXPECTED_SHAPE} for the UCI processed Cleveland version. "
            "Do not substitute a different dataset without checking the assignment."
        )

    observed_targets = set(dataset["num"].dropna().unique())
    if not observed_targets.issubset(EXPECTED_TARGET_VALUES):
        raise DatasetAcquisitionError(
            "Unexpected values were found in target column 'num': "
            f"{sorted(observed_targets)}. Expected values from 0 through 4."
        )

    return dataset


def main() -> None:
    """Obtain the source file if needed, then print a read-only data summary."""
    raw_directory = project_root() / "data" / "raw"
    dataset_path = raw_directory / DATASET_FILENAME
    raw_directory.mkdir(parents=True, exist_ok=True)

    if dataset_path.exists():
        print(f"Using existing raw dataset file: {dataset_path}")
    else:
        download_dataset(dataset_path)

    dataset = load_dataset(dataset_path)
    print(f"\nDataset file: {dataset_path}")
    print(f"Dataset shape (rows, columns): {dataset.shape}")
    print(f"\nColumn names:\n{dataset.columns.tolist()}")
    print(f"\nData types:\n{dataset.dtypes.to_string()}")
    print(f"\nFirst five rows:\n{dataset.head().to_string(index=False)}")
    print(f"\nMissing values by column:\n{dataset.isna().sum().to_string()}")
    print(f"\nDuplicate rows: {dataset.duplicated().sum()}")
    print(
        "\nObserved target values and counts for 'num':\n"
        f"{dataset['num'].value_counts(dropna=False).sort_index().to_string()}"
    )


if __name__ == "__main__":
    try:
        main()
    except DatasetAcquisitionError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
