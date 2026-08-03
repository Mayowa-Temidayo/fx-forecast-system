"""
Unit tests for calendar feature engineering.
"""

from __future__ import annotations

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from fx_forecast.features.calendar import create_calendar_features


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Create a sample market DataFrame."""

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


def test_create_calendar_features_returns_dataframe(
    sample_dataframe: pd.DataFrame,
) -> None:
    """Function should return a DataFrame."""

    result = create_calendar_features(sample_dataframe)

    assert isinstance(result, pd.DataFrame)


def test_original_columns_preserved(
    sample_dataframe: pd.DataFrame,
) -> None:
    """Original market columns should remain unchanged."""

    result = create_calendar_features(sample_dataframe)

    expected = sample_dataframe.columns.tolist()

    assert result.columns[: len(expected)].tolist() == expected


def test_all_calendar_features_created(
    sample_dataframe: pd.DataFrame,
) -> None:
    """All expected calendar features should exist."""

    result = create_calendar_features(sample_dataframe)

    expected_features = [
        "day_of_week",
        "day_of_month",
        "day_of_year",
        "week_of_year",
        "month",
        "quarter",
        "year",
        "is_month_start",
        "is_month_end",
        "is_quarter_start",
        "is_quarter_end",
        "is_year_start",
        "is_year_end",
    ]

    for feature in expected_features:
        assert feature in result.columns


def test_original_dataframe_not_modified(
    sample_dataframe: pd.DataFrame,
) -> None:
    """Input DataFrame should not be modified."""

    original = sample_dataframe.copy(deep=True)

    _ = create_calendar_features(sample_dataframe)

    assert_frame_equal(sample_dataframe, original)


def test_requires_datetime_index() -> None:
    """Non-datetime index should raise TypeError."""

    df = pd.DataFrame(
        {
            "Open": [1],
            "High": [1],
            "Low": [1],
            "Close": [1],
            "Volume": [1],
        }
    )

    with pytest.raises(TypeError, match="DatetimeIndex"):
        create_calendar_features(df)
