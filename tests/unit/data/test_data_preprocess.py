"""Tests for dataframe preprocessing."""

from __future__ import annotations

import pandas as pd
import pytest

from fx_forecast.data.preprocess import preprocess_dataframe


def test_preprocess_normalizes_column_names() -> None:
    """Column names should be normalized into canonical form."""

    df = pd.DataFrame(
        {
            " open ": [100.0, 101.0],
            "HIGH PRICE": [105.0, 106.0],
            "close": [103.0, 104.0],
        },
        index=pd.to_datetime(
            ["2025-01-02", "2025-01-01"],
        ),
    )

    result = preprocess_dataframe(df)

    assert set(result.columns) == {
        "Open",
        "High_Price",
        "Close",
    }


def test_preprocess_normalizes_datetime_index() -> None:
    """The dataframe index should be a sorted DatetimeIndex."""

    df = pd.DataFrame(
        {"Close": [102.0, 101.0]},
        index=pd.Index(
            ["2025-01-02", "2025-01-01"],
        ),
    )

    result = preprocess_dataframe(df)

    assert isinstance(result.index, pd.DatetimeIndex)
    assert result.index.name == "Date"
    assert result.index.tolist() == [
        pd.Timestamp("2025-01-01"),
        pd.Timestamp("2025-01-02"),
    ]


def test_preprocess_removes_duplicate_timestamps() -> None:
    """Duplicate timestamps should be reduced to one observation."""

    index = pd.to_datetime(
        [
            "2025-01-01",
            "2025-01-01",
            "2025-01-02",
        ]
    )

    df = pd.DataFrame(
        {"Close": [100.0, 101.0, 102.0]},
        index=index,
    )

    result = preprocess_dataframe(df)

    assert len(result) == 2
    assert not result.index.has_duplicates
    assert result.loc[pd.Timestamp("2025-01-01"), "Close"] == 100.0


def test_preprocess_sorts_chronologically() -> None:
    """Observations should be sorted in ascending date order."""

    index = pd.to_datetime(
        [
            "2025-01-03",
            "2025-01-01",
            "2025-01-02",
        ]
    )

    df = pd.DataFrame(
        {"Close": [103.0, 101.0, 102.0]},
        index=index,
    )

    result = preprocess_dataframe(df)

    assert result.index.is_monotonic_increasing
    assert result["Close"].tolist() == [101.0, 102.0, 103.0]


def test_preprocess_converts_numeric_object_columns() -> None:
    """Numeric values stored as strings should become numeric."""

    df = pd.DataFrame(
        {
            "Close": ["100.5", "101.5"],
            "Volume": ["1000", "2000"],
        },
        index=pd.to_datetime(
            ["2025-01-01", "2025-01-02"],
        ),
    )

    result = preprocess_dataframe(df)

    assert pd.api.types.is_numeric_dtype(result["Close"])
    assert pd.api.types.is_numeric_dtype(result["Volume"])
    assert result["Close"].tolist() == [100.5, 101.5]
    assert result["Volume"].tolist() == [1000, 2000]


def test_preprocess_drops_missing_rows() -> None:
    """Rows containing missing values should be removed."""

    df = pd.DataFrame(
        {
            "Close": [100.0, None, 102.0],
            "Volume": [1000, 2000, 3000],
        },
        index=pd.to_datetime(
            [
                "2025-01-01",
                "2025-01-02",
                "2025-01-03",
            ]
        ),
    )

    result = preprocess_dataframe(df)

    assert len(result) == 2
    assert result["Close"].tolist() == [100.0, 102.0]


def test_preprocess_does_not_mutate_input() -> None:
    """Preprocessing should not mutate the original dataframe."""

    df = pd.DataFrame(
        {
            " close ": [100.0, 101.0],
        },
        index=pd.to_datetime(
            ["2025-01-01", "2025-01-02"],
        ),
    )

    original_columns = df.columns.tolist()
    original_index = df.index.copy()

    preprocess_dataframe(df)

    assert df.columns.tolist() == original_columns
    assert df.index.equals(original_index)


def test_preprocess_rejects_invalid_datetime_index() -> None:
    """Invalid index values should raise a datetime conversion error."""

    df = pd.DataFrame(
        {"Close": [100.0]},
        index=pd.Index(["not-a-date"]),
    )

    with pytest.raises(ValueError):
        preprocess_dataframe(df)


def test_preprocess_enforces_canonical_column_order() -> None:
    """Market columns should follow the canonical project order."""

    df = pd.DataFrame(
        {
            "Close": [103.0, 104.0],
            "Volume": [1000, 1100],
            "Open": [100.0, 101.0],
            "Low": [99.0, 100.0],
            "High": [105.0, 106.0],
        },
        index=pd.to_datetime(
            ["2025-01-01", "2025-01-02"],
        ),
    )

    result = preprocess_dataframe(df)

    assert result.columns.tolist() == [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]
