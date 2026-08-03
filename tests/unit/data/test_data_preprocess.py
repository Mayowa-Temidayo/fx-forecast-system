"""
Unit tests for data preprocessing.
"""

from __future__ import annotations

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from fx_forecast.data.preprocess import preprocess_dataframe


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Create a valid OHLCV dataset."""

    return pd.DataFrame(
        {
            "Open": [1.10, 1.20, 1.30],
            "High": [1.15, 1.25, 1.35],
            "Low": [1.05, 1.15, 1.25],
            "Close": [1.12, 1.22, 1.32],
            "Volume": [1000, 1100, 1200],
        },
        index=pd.to_datetime(
            [
                "2025-01-01",
                "2025-01-02",
                "2025-01-03",
            ]
        ),
    )


def test_returns_dataframe(
    sample_dataframe: pd.DataFrame,
) -> None:
    """Preprocessing should return a DataFrame."""

    result = preprocess_dataframe(sample_dataframe)

    assert isinstance(result, pd.DataFrame)


def test_preserves_datetime_index(
    sample_dataframe: pd.DataFrame,
) -> None:
    """DatetimeIndex should be preserved."""

    result = preprocess_dataframe(sample_dataframe)

    assert isinstance(result.index, pd.DatetimeIndex)


def test_sorts_index() -> None:
    """Rows should be sorted by datetime."""

    df = pd.DataFrame(
        {
            "Open": [1.30, 1.10, 1.20],
            "High": [1.35, 1.15, 1.25],
            "Low": [1.25, 1.05, 1.15],
            "Close": [1.32, 1.12, 1.22],
            "Volume": [1200, 1000, 1100],
        },
        index=pd.to_datetime(
            [
                "2025-01-03",
                "2025-01-01",
                "2025-01-02",
            ]
        ),
    )

    result = preprocess_dataframe(df)

    assert result.index.is_monotonic_increasing


def test_removes_duplicate_index() -> None:
    """Duplicate timestamps should be removed."""

    df = pd.DataFrame(
        {
            "Open": [1.10, 1.20, 1.30],
            "High": [1.15, 1.25, 1.35],
            "Low": [1.05, 1.15, 1.25],
            "Close": [1.12, 1.22, 1.32],
            "Volume": [1000, 1100, 1200],
        },
        index=pd.to_datetime(
            [
                "2025-01-01",
                "2025-01-01",
                "2025-01-03",
            ]
        ),
    )

    result = preprocess_dataframe(df)

    assert not result.index.has_duplicates


def test_fills_missing_values(
    sample_dataframe: pd.DataFrame,
) -> None:
    """Missing values should be filled."""

    df = sample_dataframe.copy()

    df.loc[df.index[1], "Open"] = pd.NA

    result = preprocess_dataframe(df)

    assert result["Open"].isna().sum() == 0


def test_drops_remaining_missing_rows(
    sample_dataframe: pd.DataFrame,
) -> None:
    """No missing values should remain."""

    df = sample_dataframe.copy()

    df.loc[df.index[0], :] = pd.NA

    result = preprocess_dataframe(df)

    assert result.isna().sum().sum() == 0


def test_standardizes_column_names(
    sample_dataframe: pd.DataFrame,
) -> None:
    """Column names should be standardized."""

    df = sample_dataframe.copy()

    df.columns = [
        " open ",
        "HIGH",
        "low",
        "close price",
        "volume",
    ]

    result = preprocess_dataframe(df)

    assert list(result.columns) == [
        "Open",
        "High",
        "Low",
        "Close_Price",
        "Volume",
    ]


def test_original_dataframe_not_modified(
    sample_dataframe: pd.DataFrame,
) -> None:
    """Original dataframe should remain unchanged."""

    original = sample_dataframe.copy(deep=True)

    preprocess_dataframe(sample_dataframe)

    assert_frame_equal(
        sample_dataframe,
        original,
    )


def test_clean_dataframe_unchanged(
    sample_dataframe: pd.DataFrame,
) -> None:
    """Already clean datasets should remain unchanged."""

    result = preprocess_dataframe(sample_dataframe)

    assert_frame_equal(
        sample_dataframe,
        result,
        check_freq=False,
    )
