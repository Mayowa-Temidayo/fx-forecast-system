"""
Unit tests for technical indicators.
"""

from __future__ import annotations

import pandas as pd
import pytest

from fx_forecast.features.technical import (
    create_technical_features,
)


@pytest.fixture
def sample_prices() -> pd.DataFrame:
    """Create sample OHLCV data."""

    dates = pd.date_range(
        "2025-01-01",
        periods=10,
        freq="D",
    )

    return pd.DataFrame(
        {
            "Open": [1.10, 1.11, 1.12, 1.13, 1.14, 1.15, 1.16, 1.17, 1.18, 1.19],
            "High": [1.11, 1.12, 1.13, 1.14, 1.15, 1.16, 1.17, 1.18, 1.19, 1.20],
            "Low": [1.09, 1.10, 1.11, 1.12, 1.13, 1.14, 1.15, 1.16, 1.17, 1.18],
            "Close": [1.10, 1.11, 1.12, 1.13, 1.14, 1.15, 1.16, 1.17, 1.18, 1.19],
            "Volume": [100] * 10,
        },
        index=dates,
    )


def test_returns_dataframe(
    sample_prices: pd.DataFrame,
) -> None:
    """Output should be a DataFrame."""

    result = create_technical_features(sample_prices)

    assert isinstance(result, pd.DataFrame)


def test_preserves_index(
    sample_prices: pd.DataFrame,
) -> None:
    """Datetime index should remain unchanged."""

    result = create_technical_features(sample_prices)

    pd.testing.assert_index_equal(
        result.index,
        sample_prices.index,
    )


def test_creates_sma(
    sample_prices: pd.DataFrame,
) -> None:
    """SMA feature should exist."""

    result = create_technical_features(sample_prices)

    assert "sma_5" in result.columns


def test_creates_ema(
    sample_prices: pd.DataFrame,
) -> None:
    """EMA feature should exist."""

    result = create_technical_features(sample_prices)

    assert "ema_5" in result.columns


def test_creates_rolling_std(
    sample_prices: pd.DataFrame,
) -> None:
    """Rolling volatility feature should exist."""

    result = create_technical_features(sample_prices)

    assert "rolling_std_5" in result.columns


def test_original_dataframe_not_modified(
    sample_prices: pd.DataFrame,
) -> None:
    """Input dataframe should remain unchanged."""

    original_columns = sample_prices.columns.tolist()

    create_technical_features(sample_prices)

    assert sample_prices.columns.tolist() == original_columns


def test_new_columns_are_numeric(
    sample_prices: pd.DataFrame,
) -> None:
    """Generated features should be numeric."""

    result = create_technical_features(sample_prices)

    for column in [
        "sma_5",
        "ema_5",
        "rolling_std_5",
    ]:
        assert pd.api.types.is_numeric_dtype(result[column])


def test_output_has_more_columns(
    sample_prices: pd.DataFrame,
) -> None:
    """Feature engineering should add columns."""

    result = create_technical_features(sample_prices)

    assert result.shape[1] > sample_prices.shape[1]
