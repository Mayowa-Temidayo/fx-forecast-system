"""Time-series backtesting utilities."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from fx_forecast.models.base import BaseForecastModel


@dataclass(frozen=True)
class BacktestResult:
    """Results produced by a historical backtest."""

    actual: pd.Series
    predicted: pd.Series


def run_backtest(
    model: BaseForecastModel,
    X: pd.DataFrame,
    y: pd.Series,
    min_train_size: int,
) -> BacktestResult:
    """Run an expanding-window one-step-ahead backtest."""
    _validate_inputs(X, y, min_train_size)

    predictions: list[float] = []
    actual: list[float] = []

    for position in range(min_train_size, len(X)):
        X_train = X.iloc[:position]
        y_train = y.iloc[:position]
        X_test = X.iloc[[position]]

        model.fit(X_train, y_train)
        prediction = model.predict(X_test).iloc[0]

        predictions.append(float(prediction))
        actual.append(float(y.iloc[position]))

    return BacktestResult(
        actual=y.iloc[min_train_size:].rename("actual"),
        predicted=pd.Series(
            predictions,
            index=y.index[min_train_size:],
            name="prediction",
        ),
    )


def _validate_inputs(
    X: pd.DataFrame,
    y: pd.Series,
    min_train_size: int,
) -> None:
    """Validate backtesting inputs."""
    if X.empty or y.empty:
        raise ValueError("X and y must not be empty.")

    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of rows.")

    if not X.index.equals(y.index):
        raise ValueError("X and y must have identical indexes.")

    if min_train_size < 1:
        raise ValueError("min_train_size must be at least 1.")

    if min_train_size >= len(X):
        raise ValueError("min_train_size must be smaller than the dataset size.")
