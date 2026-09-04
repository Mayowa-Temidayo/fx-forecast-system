"""Unit tests for time-series backtesting."""

from __future__ import annotations

import pandas as pd
import pytest

from fx_forecast.evaluation.backtest import BacktestResult, run_backtest
from fx_forecast.models.naive import NaiveForecastModel


def _sample_data() -> tuple[pd.DataFrame, pd.Series]:
    """Create deterministic time-series test data."""
    index = pd.date_range("2025-01-01", periods=6)

    X = pd.DataFrame(
        {"feature": range(6)},
        index=index,
    )

    y = pd.Series(
        [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
        index=index,
        name="target",
    )

    return X, y


def test_backtest_returns_result() -> None:
    """Backtest should return a BacktestResult."""
    X, y = _sample_data()

    result = run_backtest(
        NaiveForecastModel(),
        X,
        y,
        min_train_size=3,
    )

    assert isinstance(result, BacktestResult)


def test_backtest_returns_expected_number_of_predictions() -> None:
    """Backtest should predict every observation after training starts."""
    X, y = _sample_data()

    result = run_backtest(
        NaiveForecastModel(),
        X,
        y,
        min_train_size=3,
    )

    assert len(result.predicted) == 3
    assert len(result.actual) == 3


def test_backtest_preserves_prediction_index() -> None:
    """Backtest predictions should preserve their original timestamps."""
    X, y = _sample_data()

    result = run_backtest(
        NaiveForecastModel(),
        X,
        y,
        min_train_size=3,
    )

    assert result.predicted.index.equals(y.index[3:])


def test_backtest_uses_only_past_data() -> None:
    """Each prediction should use only observations available at that time."""
    X, y = _sample_data()

    result = run_backtest(
        NaiveForecastModel(),
        X,
        y,
        min_train_size=3,
    )

    assert result.predicted.tolist() == [102.0, 103.0, 104.0]


def test_actual_values_are_aligned() -> None:
    """Actual values should align with predictions."""
    X, y = _sample_data()

    result = run_backtest(
        NaiveForecastModel(),
        X,
        y,
        min_train_size=3,
    )

    pd.testing.assert_series_equal(
        result.actual,
        y.iloc[3:].rename("actual"),
    )


def test_rejects_empty_inputs() -> None:
    """Backtesting should reject empty inputs."""
    X = pd.DataFrame()
    y = pd.Series(dtype=float)

    with pytest.raises(ValueError, match="must not be empty"):
        run_backtest(
            NaiveForecastModel(),
            X,
            y,
            min_train_size=2,
        )


def test_rejects_mismatched_lengths() -> None:
    """Backtesting should reject mismatched input lengths."""
    X, y = _sample_data()

    with pytest.raises(ValueError, match="same number of rows"):
        run_backtest(
            NaiveForecastModel(),
            X.iloc[:-1],
            y,
            min_train_size=2,
        )


def test_rejects_mismatched_indexes() -> None:
    """Backtesting should reject mismatched indexes."""
    X, y = _sample_data()

    y = y.copy()
    y.index = pd.date_range("2026-01-01", periods=6)

    with pytest.raises(ValueError, match="identical indexes"):
        run_backtest(
            NaiveForecastModel(),
            X,
            y,
            min_train_size=2,
        )


def test_rejects_invalid_minimum_training_size() -> None:
    """Backtesting should reject an invalid minimum training size."""
    X, y = _sample_data()

    with pytest.raises(ValueError, match="at least 1"):
        run_backtest(
            NaiveForecastModel(),
            X,
            y,
            min_train_size=0,
        )


def test_rejects_training_size_larger_than_dataset() -> None:
    """Backtesting should reject a training size covering the dataset."""
    X, y = _sample_data()

    with pytest.raises(ValueError, match="smaller than the dataset"):
        run_backtest(
            NaiveForecastModel(),
            X,
            y,
            min_train_size=len(X),
        )
