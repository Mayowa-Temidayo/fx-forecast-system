"""
Unit tests for technical indicator features.
"""

from __future__ import annotations

import pandas as pd
import pytest

from fx_forecast.features.technical import create_technical_features


@pytest.fixture
def sample_prices() -> pd.DataFrame:
    """Create sample market prices."""

    index = pd.date_range(
        "2025-01-01",
        periods=10,
        freq="D",
    )

    return pd.DataFrame(
        {
            "Open": [1.10 + i * 0.01 for i in range(10)],
            "High": [1.11 + i * 0.01 for i in range(10)],
            "Low": [1.09 + i * 0.01 for i in range(10)],
            "Close": [1.10 + i * 0.01 for i in range(10)],
            "Volume": [100] * 10,
        },
        index=index,
    )


def test_creates_simple_moving_averages(
    sample_prices: pd.DataFrame,
) -> None:
    """Simple moving averages should be created."""

    result = create_technical_features(sample_prices)

    for column in [
        "sma_5",
        "sma_10",
        "sma_20",
    ]:
        assert column in result.columns


def test_creates_exponential_moving_averages(
    sample_prices: pd.DataFrame,
) -> None:
    """Exponential moving averages should be created."""

    result = create_technical_features(sample_prices)

    for column in [
        "ema_5",
        "ema_10",
        "ema_20",
    ]:
        assert column in result.columns


def test_sma_5_is_calculated_correctly(
    sample_prices: pd.DataFrame,
) -> None:
    """The five-period SMA should use the previous five closing prices."""

    result = create_technical_features(sample_prices)

    expected = sample_prices["Close"].iloc[:5].mean()

    assert result["sma_5"].iloc[4] == pytest.approx(expected)


def test_ema_features_are_numeric(
    sample_prices: pd.DataFrame,
) -> None:
    """Generated EMA features should be numeric."""

    result = create_technical_features(sample_prices)

    for column in [
        "ema_5",
        "ema_10",
        "ema_20",
    ]:
        assert pd.api.types.is_numeric_dtype(result[column])


def test_sma_features_are_numeric(
    sample_prices: pd.DataFrame,
) -> None:
    """Generated SMA features should be numeric."""

    result = create_technical_features(sample_prices)

    for column in [
        "sma_5",
        "sma_10",
        "sma_20",
    ]:
        assert pd.api.types.is_numeric_dtype(result[column])


def test_original_dataframe_is_not_modified(
    sample_prices: pd.DataFrame,
) -> None:
    """Technical feature engineering should not mutate the input."""

    original = sample_prices.copy(deep=True)

    create_technical_features(sample_prices)

    pd.testing.assert_frame_equal(
        sample_prices,
        original,
    )


def test_missing_close_column_raises() -> None:
    """A missing Close column should raise ValueError."""

    df = pd.DataFrame(
        {
            "Open": [1.0, 2.0],
            "High": [1.1, 2.1],
            "Low": [0.9, 1.9],
        }
    )

    with pytest.raises(
        ValueError,
        match="Column 'Close' not found.",
    ):
        create_technical_features(df)
