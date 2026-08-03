"""
Unit tests for target engineering.
"""

from __future__ import annotations

import pandas as pd
import pandas.testing as pdt

from fx_forecast.features.target import create_target_features


def sample_prices() -> pd.DataFrame:
    """Sample closing price data."""

    return pd.DataFrame(
        {
            "Close": [
                1.10,
                1.11,
                1.12,
                1.13,
                1.14,
                1.15,
                1.16,
                1.17,
                1.18,
                1.19,
            ]
        },
        index=pd.date_range("2025-01-01", periods=10, freq="D"),
    )


def test_returns_dataframe() -> None:
    """Should return a DataFrame."""

    result = create_target_features(sample_prices())

    assert isinstance(result, pd.DataFrame)


def test_preserves_index() -> None:
    """Index should remain unchanged."""

    df = sample_prices()

    result = create_target_features(df)

    pdt.assert_index_equal(result.index, df.index)


def test_creates_target_columns() -> None:
    """All target columns should be created."""

    result = create_target_features(sample_prices())

    for column in [
        "target",
        "target_3d",
        "target_5d",
    ]:
        assert column in result.columns


def test_creates_future_return_columns() -> None:
    """Future return columns should exist."""

    result = create_target_features(sample_prices())

    for column in [
        "future_return_1d",
        "future_return_3d",
        "future_return_5d",
    ]:
        assert column in result.columns


def test_target_shift_is_correct() -> None:
    """Next-day target should equal tomorrow's Close."""

    result = create_target_features(sample_prices())

    assert result["target"].iloc[0] == result["Close"].iloc[1]


def test_last_values_are_nan() -> None:
    """Trailing rows should contain NaN targets."""

    result = create_target_features(sample_prices())

    assert pd.isna(result["target"].iloc[-1])
    assert pd.isna(result["target_3d"].iloc[-3])
    assert pd.isna(result["target_5d"].iloc[-5])


def test_original_dataframe_not_modified() -> None:
    """Input DataFrame should remain unchanged."""

    df = sample_prices()
    original = df.copy(deep=True)

    create_target_features(df)

    pdt.assert_frame_equal(df, original)


def test_future_returns_are_numeric() -> None:
    """Future return columns should be numeric."""

    result = create_target_features(sample_prices())

    assert pd.api.types.is_numeric_dtype(result["future_return_1d"])
    assert pd.api.types.is_numeric_dtype(result["future_return_3d"])
    assert pd.api.types.is_numeric_dtype(result["future_return_5d"])


def test_output_has_more_columns() -> None:
    """Output should contain additional engineered columns."""

    df = sample_prices()

    result = create_target_features(df)

    assert result.shape[1] > df.shape[1]
