"""Integration tests for the backtesting pipeline."""

from __future__ import annotations

import pandas as pd

from fx_forecast.evaluation.backtest import run_backtest
from fx_forecast.evaluation.metrics import (
    directional_accuracy,
    mean_absolute_error,
    root_mean_squared_error,
)
from fx_forecast.models.naive import NaiveForecastModel


def test_backtest_predictions_can_be_evaluated() -> None:
    """Test backtesting followed by forecast evaluation."""
    index = pd.date_range("2025-01-01", periods=8)

    X = pd.DataFrame(
        {"feature": range(8)},
        index=index,
    )

    y = pd.Series(
        [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0],
        index=index,
        name="target",
    )

    result = run_backtest(
        NaiveForecastModel(),
        X,
        y,
        min_train_size=4,
    )

    mae = mean_absolute_error(result.actual, result.predicted)
    rmse = root_mean_squared_error(result.actual, result.predicted)
    accuracy = directional_accuracy(result.actual, result.predicted)

    assert len(result.predicted) == 4
    assert mae == 1.0
    assert rmse == 1.0
    assert accuracy == 100.0
