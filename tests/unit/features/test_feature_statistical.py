"""
Unit tests for statistical feature engineering.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fx_forecast.features.statistical import create_statistical_features


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Create a sample market dataset."""

    return pd.DataFrame(
        {
            "Open": [1.10, 1.12, 1.15, 1.18, 1.20, 1.22],
            "High": [1.12, 1.14, 1.17, 1.20, 1.22, 1.24],
            "Low": [1.08, 1.10, 1.13, 1.16, 1.18, 1.20],
            "Close": [1.11, 1.13, 1.16, 1.19, 1.21, 1.23],
            "Volume": [1000, 1100, 1200, 1300, 1400, 1500],
        },
        index=pd.date_range("2025-01-01", periods=6, freq="D"),
    )


def test_statistical_features_created(
    sample_dataframe: pd.DataFrame,
) -> None:
    """Expected statistical features should be added."""

    result = create_statistical_features(
        sample_dataframe,
        windows=(3,),
    )

    expected = [
        "return",
        "log_return",
        "rolling_mean_3",
        "rolling_std_3",
        "rolling_min_3",
        "rolling_max_3",
    ]

    for column in expected:
        assert column in result.columns


def test_missing_close_column() -> None:
    """Missing Close column should raise ValueError."""

    df = pd.DataFrame(
        {
            "Open": [1.0],
        }
    )

    with pytest.raises(ValueError, match="Close"):
        create_statistical_features(df)


def test_return_contains_nan_first_row(
    sample_dataframe: pd.DataFrame,
) -> None:
    """First return must be NaN."""

    result = create_statistical_features(sample_dataframe)

    assert np.isnan(result["return"].iloc[0])


def test_log_return_contains_nan_first_row(
    sample_dataframe: pd.DataFrame,
) -> None:
    """First log return must be NaN."""

    result = create_statistical_features(sample_dataframe)

    assert np.isnan(result["log_return"].iloc[0])


def test_rolling_mean(
    sample_dataframe: pd.DataFrame,
) -> None:
    """Rolling mean should produce valid values."""

    result = create_statistical_features(
        sample_dataframe,
        windows=(3,),
    )

    expected = (1.11 + 1.13 + 1.16) / 3

    assert result["rolling_mean_3"].iloc[2] == pytest.approx(expected)


def test_rolling_min(
    sample_dataframe: pd.DataFrame,
) -> None:
    """Rolling minimum should be correct."""

    result = create_statistical_features(
        sample_dataframe,
        windows=(3,),
    )

    assert result["rolling_min_3"].iloc[2] == pytest.approx(1.11)


def test_rolling_max(
    sample_dataframe: pd.DataFrame,
) -> None:
    """Rolling maximum should be correct."""

    result = create_statistical_features(
        sample_dataframe,
        windows=(3,),
    )

    assert result["rolling_max_3"].iloc[2] == pytest.approx(1.16)


def test_original_dataframe_not_modified(
    sample_dataframe: pd.DataFrame,
) -> None:
    """Feature generation must not modify the input DataFrame."""

    original = sample_dataframe.copy(deep=True)

    create_statistical_features(sample_dataframe)

    pd.testing.assert_frame_equal(sample_dataframe, original)
