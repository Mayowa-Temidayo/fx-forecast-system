"""Unit tests for forecast evaluation metrics."""

from __future__ import annotations

import pandas as pd
import pytest

from fx_forecast.evaluation.metrics import (
    directional_accuracy,
    mean_absolute_error,
    mean_absolute_percentage_error,
    root_mean_squared_error,
)


@pytest.fixture
def actual() -> pd.Series:
    """Return sample actual values."""
    return pd.Series([100.0, 110.0, 120.0])


@pytest.fixture
def predicted() -> pd.Series:
    """Return sample predicted values."""
    return pd.Series([102.0, 108.0, 123.0])


def test_mean_absolute_error(
    actual: pd.Series,
    predicted: pd.Series,
) -> None:
    """MAE should calculate the average absolute error."""
    assert mean_absolute_error(actual, predicted) == pytest.approx(2.333333)


def test_root_mean_squared_error(
    actual: pd.Series,
    predicted: pd.Series,
) -> None:
    """RMSE should calculate the root mean squared error."""
    assert root_mean_squared_error(actual, predicted) == pytest.approx(2.380476)


def test_mean_absolute_percentage_error(
    actual: pd.Series,
    predicted: pd.Series,
) -> None:
    """MAPE should calculate percentage error."""
    assert mean_absolute_percentage_error(actual, predicted) == pytest.approx(2.106061)


def test_directional_accuracy(
    actual: pd.Series,
    predicted: pd.Series,
) -> None:
    """Directional accuracy should compare price movement direction."""
    assert directional_accuracy(actual, predicted) == pytest.approx(100.0)


def test_rejects_empty_inputs() -> None:
    """Metrics should reject empty inputs."""
    actual = pd.Series(dtype=float)
    predicted = pd.Series(dtype=float)

    with pytest.raises(ValueError, match="must not be empty"):
        mean_absolute_error(actual, predicted)


def test_rejects_mismatched_lengths() -> None:
    """Metrics should reject mismatched input lengths."""
    actual = pd.Series([100.0, 110.0])
    predicted = pd.Series([100.0])

    with pytest.raises(ValueError, match="same number of rows"):
        mean_absolute_error(actual, predicted)


def test_rejects_mismatched_indexes() -> None:
    """Metrics should reject mismatched indexes."""
    actual = pd.Series(
        [100.0, 110.0],
        index=pd.date_range("2025-01-01", periods=2),
    )
    predicted = pd.Series(
        [100.0, 110.0],
        index=pd.date_range("2025-02-01", periods=2),
    )

    with pytest.raises(ValueError, match="identical indexes"):
        mean_absolute_error(actual, predicted)


def test_mape_rejects_zero_actual() -> None:
    """MAPE should reject zero actual values."""
    actual = pd.Series([100.0, 0.0])
    predicted = pd.Series([101.0, 1.0])

    with pytest.raises(ValueError, match="actual values contain zero"):
        mean_absolute_percentage_error(actual, predicted)
