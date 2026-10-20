"""Inspect the UCI Cleveland heart disease data and save exploratory plots."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from data_acquisition import (
    COLUMNS,
    DATASET_FILENAME,
    DatasetAcquisitionError,
    load_dataset,
    project_root,
)

TARGET_COLUMN = "num"
CONTINUOUS_COLUMNS = ["age", "trestbps", "chol", "thalach", "oldpeak"]


def load_processed_dataset(dataset_path: Path) -> pd.DataFrame:
    """Load a processed CSV and confirm it still matches the UCI data schema."""
    try:
        dataset = pd.read_csv(dataset_path)
    except (OSError, pd.errors.ParserError, ValueError) as error:
        raise DatasetAcquisitionError(
            f"Could not read processed dataset '{dataset_path}'. Details: {error}"
        ) from error

    if dataset.columns.tolist() != COLUMNS:
        raise DatasetAcquisitionError(
            f"Processed dataset columns do not match the expected UCI schema. "
            f"Expected {COLUMNS}; found {dataset.columns.tolist()}."
        )
    if dataset.shape != (303, len(COLUMNS)):
        raise DatasetAcquisitionError(
            f"Processed dataset has shape {dataset.shape}; expected "
            f"(303, {len(COLUMNS)}) for the Step 2 Cleveland dataset."
        )

    observed_targets = set(dataset[TARGET_COLUMN].dropna().unique())
    if not observed_targets.issubset({0, 1, 2, 3, 4}):
        raise DatasetAcquisitionError(
            f"Unexpected target values in '{TARGET_COLUMN}': "
            f"{sorted(observed_targets)}. Expected values from 0 through 4."
        )

    return dataset


def parse_arguments() -> argparse.Namespace:
    """Read input options so EDA can inspect raw or processed data."""
    parser = argparse.ArgumentParser(
        description=(
            "Print a data-quality summary and save plots for the UCI "
            "Cleveland heart disease dataset."
        )
    )
    parser.add_argument(
        "--input",
        default=f"data/raw/{DATASET_FILENAME}",
        help=(
            "Dataset path relative to the project root. Defaults to the "
            "Step 2 raw Cleveland file."
        ),
    )
    parser.add_argument(
        "--format",
        choices=("raw", "processed"),
        default="raw",
        help="Use 'raw' for the headerless UCI source or 'processed' for its CSV copy.",
    )
    return parser.parse_args()


def save_plots(dataset: pd.DataFrame, prefix: str, output_directory: Path) -> None:
    """Save continuous-feature histograms, correlation, target, and boxplots."""
    output_directory.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(3, 2, figsize=(12, 12))
    for axis, column in zip(axes.flat, CONTINUOUS_COLUMNS):
        dataset[column].dropna().hist(ax=axis, bins=15, edgecolor="black")
        axis.set_title(f"Distribution of {column}")
        axis.set_xlabel(column)
        axis.set_ylabel("Number of patient records")
    axes.flat[-1].set_visible(False)
    figure.suptitle("Cleveland Dataset: Continuous Feature Histograms")
    figure.tight_layout()
    histogram_path = output_directory / f"{prefix}_feature_histograms.png"
    figure.savefig(histogram_path, dpi=150, bbox_inches="tight")
    plt.close(figure)

    correlation = dataset[CONTINUOUS_COLUMNS].corr()
    figure, axis = plt.subplots(figsize=(9, 7))
    image = axis.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
    axis.set_xticks(range(len(CONTINUOUS_COLUMNS)), CONTINUOUS_COLUMNS, rotation=45)
    axis.set_yticks(range(len(CONTINUOUS_COLUMNS)), CONTINUOUS_COLUMNS)
    for row in range(len(CONTINUOUS_COLUMNS)):
        for column in range(len(CONTINUOUS_COLUMNS)):
            axis.text(
                column,
                row,
                f"{correlation.iloc[row, column]:.2f}",
                ha="center",
                va="center",
                color="black",
            )
    figure.colorbar(image, ax=axis, label="Pearson correlation")
    axis.set_title("Correlation Between Continuous Features")
    figure.tight_layout()
    correlation_path = output_directory / f"{prefix}_correlation_heatmap.png"
    figure.savefig(correlation_path, dpi=150, bbox_inches="tight")
    plt.close(figure)

    target_counts = dataset[TARGET_COLUMN].value_counts().sort_index()
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(target_counts.index.astype(str), target_counts.values)
    axis.set_title("Cleveland Dataset Target Class Distribution")
    axis.set_xlabel("num (0 = no disease; 1–4 = disease present)")
    axis.set_ylabel("Number of patient records")
    figure.tight_layout()
    target_path = output_directory / f"{prefix}_target_distribution.png"
    figure.savefig(target_path, dpi=150, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(11, 6))
    boxplot_data = [dataset[column].dropna() for column in CONTINUOUS_COLUMNS]
    axis.boxplot(boxplot_data, showfliers=True)
    axis.set_xticks(range(1, len(CONTINUOUS_COLUMNS) + 1), CONTINUOUS_COLUMNS)
    axis.set_title("Possible Outliers in Continuous Features")
    axis.set_xlabel("Continuous feature")
    axis.set_ylabel("Feature value (original units)")
    figure.tight_layout()
    boxplot_path = output_directory / f"{prefix}_feature_boxplots.png"
    figure.savefig(boxplot_path, dpi=150, bbox_inches="tight")
    plt.close(figure)

    print("\nSaved visualization files:")
    for path in (histogram_path, correlation_path, target_path, boxplot_path):
        print(f"- {path}")


def main() -> None:
    """Print an initial summary and save the requested exploratory plots."""
    arguments = parse_arguments()
    root = project_root()
    input_path = Path(arguments.input)
    if not input_path.is_absolute():
        input_path = root / input_path
    input_path = input_path.resolve()

    if not input_path.is_file():
        raise DatasetAcquisitionError(
            f"Dataset file not found: {input_path}. Run the Step 2 acquisition "
            "script first, or provide an existing file with --input."
        )

    dataset = (
        load_dataset(input_path)
        if arguments.format == "raw"
        else load_processed_dataset(input_path)
    )
    print(f"Input dataset: {input_path}")
    print(f"Dataset shape (rows, columns): {dataset.shape}")
    print(f"\nFirst five rows:\n{dataset.head().to_string(index=False)}")
    print(f"\nColumn names:\n{dataset.columns.tolist()}")
    print(f"\nData types:\n{dataset.dtypes.to_string()}")
    print(f"\nMissing values by column:\n{dataset.isna().sum().to_string()}")
    print(f"\nDuplicate rows: {dataset.duplicated().sum()}")
    print(f"\nNumerical summary statistics:\n{dataset.describe().T.to_string()}")
    print(
        "\nTarget column 'num' is UCI angiographic disease status: "
        "0 means no diagnosed disease; values 1 through 4 mean disease is present. "
        "No target conversion is performed."
    )
    print(
        "\nTarget value counts:\n"
        f"{dataset[TARGET_COLUMN].value_counts(dropna=False).sort_index().to_string()}"
    )

    prefix = "raw" if arguments.format == "raw" else "processed"
    save_plots(dataset, prefix, root / "reports" / "figures")


if __name__ == "__main__":
    try:
        main()
    except DatasetAcquisitionError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
