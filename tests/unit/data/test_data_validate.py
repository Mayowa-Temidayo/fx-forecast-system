"""
Unit tests for data validation.
"""

from __future__ import annotations

import pandas as pd
import pytest

from fx_forecast.data.validate import validate_dataframe


@pytest.fixture
def valid_dataframe() -> pd.DataFrame:
    """Create a valid OHLCV dataset."""

    return pd.DataFrame(
        {
            "Open": [1.10, 1.20, 1.30],
            "High": [1.15, 1.25, 1.35],
            "Low": [1.05, 1.15, 1.25],
            "Close": [1.12, 1.22, 1.32],
            "Volume": [1000, 1100, 1200],
        },
        index=pd.date_range("2025-01-01", periods=3, freq="D"),
    )


def test_validate_valid_dataframe(
    valid_dataframe: pd.DataFrame,
) -> None:
    """A valid dataframe should pass validation."""

    validated = validate_dataframe(valid_dataframe)

    pd.testing.assert_frame_equal(valid_dataframe, validated)


def test_invalid_index_type(
    valid_dataframe: pd.DataFrame,
) -> None:
    """Index must be a DatetimeIndex."""

    df = valid_dataframe.copy()
    df.index = ["a", "b", "c"]

    with pytest.raises(TypeError, match="DatetimeIndex"):
        validate_dataframe(df)


def test_unsorted_index(
    valid_dataframe: pd.DataFrame,
) -> None:
    """Index must be sorted ascending."""

    df = valid_dataframe.sort_index(ascending=False)

    with pytest.raises(ValueError, match="ascending"):
        validate_dataframe(df)


def test_duplicate_index(
    valid_dataframe: pd.DataFrame,
) -> None:
    """Duplicate timestamps should fail."""

    df = valid_dataframe.copy()

    df.index = pd.DatetimeIndex(
        [
            "2025-01-01",
            "2025-01-01",
            "2025-01-03",
        ]
    )

    with pytest.raises(ValueError, match="Duplicate"):
        validate_dataframe(df)


def test_missing_required_column(
    valid_dataframe: pd.DataFrame,
) -> None:
    """Missing OHLCV columns should fail."""

    df = valid_dataframe.drop(columns=["Volume"])

    with pytest.raises(ValueError, match="Missing required columns"):
        validate_dataframe(df)


@pytest.mark.parametrize(
    "column",
    [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ],
)
def test_non_numeric_column(
    valid_dataframe: pd.DataFrame,
    column: str,
) -> None:
    """Numeric columns must contain numeric data."""

    df = valid_dataframe.copy()

    df[column] = "invalid"

    with pytest.raises(TypeError, match="numeric"):
        validate_dataframe(df)


@pytest.mark.parametrize(
    "column",
    [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ],
)
def test_empty_column(
    valid_dataframe: pd.DataFrame,
    column: str,
) -> None:
    """Columns containing only NaN should fail."""

    df = valid_dataframe.copy()

    df[column] = pd.NA

    with pytest.raises(ValueError, match="Empty columns"):
        validate_dataframe(df)
