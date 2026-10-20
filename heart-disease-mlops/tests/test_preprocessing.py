"""Test Step 3 cleaning with the project's documented UCI column schema."""

import pandas as pd
import pytest

from heart_disease.data_acquisition import COLUMNS, DatasetAcquisitionError
from heart_disease.data_preprocessing import (
    DOCUMENTED_CATEGORIES,
    INTEGER_COLUMNS,
    normalize_integer_types,
    validate_documented_values,
)


@pytest.fixture
def cleveland_source_rows() -> pd.DataFrame:
    """Use observed rows from processed.cleveland.data, not invented columns."""
    rows = [
        [
            63.0, 1.0, 1.0, 145.0, 233.0, 1.0, 2.0, 150.0, 0.0, 2.3,
            3.0, 0.0, 6.0, 0,
        ],
        [
            67.0, 1.0, 4.0, 160.0, 286.0, 0.0, 2.0, 108.0, 1.0, 1.5,
            2.0, 3.0, 3.0, 2,
        ],
        [
            53.0, 0.0, 3.0, 128.0, 216.0, 0.0, 2.0, 115.0, 0.0, 0.0,
            1.0, 0.0, None, 0,
        ],
    ]
    return pd.DataFrame(rows, columns=COLUMNS)


def test_documented_category_codes_are_accepted(
    cleveland_source_rows: pd.DataFrame,
) -> None:
    """Actual UCI category and target codes pass the project's validation."""
    validate_documented_values(cleveland_source_rows)


def test_integer_columns_are_nullable_integers_and_missing_values_remain(
    cleveland_source_rows: pd.DataFrame,
) -> None:
    """Cleaning changes numeric types without filling a missing category."""
    cleaned = normalize_integer_types(cleveland_source_rows)

    assert cleaned.columns.tolist() == COLUMNS, (
        "Preprocessing must preserve the UCI source columns in their declared order."
    )
    assert all(
        str(cleaned[column].dtype) == "Int64"
        for column in INTEGER_COLUMNS
    ), "Documented integer-valued measurements and codes must use nullable integers."
    assert pd.api.types.is_float_dtype(cleaned["oldpeak"]), (
        "The continuous oldpeak measurement must remain a floating-point column."
    )
    assert pd.isna(cleaned.loc[2, "thal"]), (
        "Preprocessing must preserve missing source values instead of imputing them."
    )
    assert cleaned["num"].tolist() == [0, 2, 0], (
        "Preprocessing must preserve the original UCI target labels (0 through 4)."
    )


def test_undocumented_category_code_is_rejected(
    cleveland_source_rows: pd.DataFrame,
) -> None:
    """The validator stops instead of guessing how to interpret a new code."""
    column = "cp"
    invalid_value = max(DOCUMENTED_CATEGORIES[column]) + 1
    invalid_rows = cleveland_source_rows.copy()
    invalid_rows.loc[0, column] = invalid_value

    with pytest.raises(DatasetAcquisitionError, match="undocumented"):
        validate_documented_values(invalid_rows)


def test_non_integer_code_is_rejected(
    cleveland_source_rows: pd.DataFrame,
) -> None:
    """An integer-coded category cannot be silently rounded during cleaning."""
    invalid_rows = cleveland_source_rows.copy()
    invalid_rows.loc[0, "sex"] = 0.5

    with pytest.raises(DatasetAcquisitionError, match="non-integer"):
        normalize_integer_types(invalid_rows)
