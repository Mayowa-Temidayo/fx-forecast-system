"""Forecast evaluation metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def mean_absolute_error(
    actual: pd.Series,
    predicted: pd.Series,
) -> float:
    """Calculate mean absolute error."""
    _validate_inputs(actual, predicted)

    return float(np.mean(np.abs(actual - predicted)))


def root_mean_squared_error(
    actual: pd.Series,
    predicted: pd.Series,
) -> float:
    """Calculate root mean squared error."""
    _validate_inputs(actual, predicted)

    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mean_absolute_percentage_error(
    actual: pd.Series,
    predicted: pd.Series,
) -> float:
    """Calculate mean absolute percentage error."""
    _validate_inputs(actual, predicted)

    if (actual == 0).any():
        raise ValueError("MAPE cannot be calculated when actual values contain zero.")

    return float(np.mean(np.abs((actual - predicted) / actual)) * 100)


def directional_accuracy(
    actual: pd.Series,
    predicted: pd.Series,
) -> float:
    """Calculate the percentage of predictions with the correct direction."""
    _validate_inputs(actual, predicted)

    actual_direction = np.sign(actual.diff())
    predicted_direction = np.sign(predicted.diff())

    valid = actual_direction.notna() & predicted_direction.notna()

    return float((actual_direction[valid] == predicted_direction[valid]).mean() * 100)


def _validate_inputs(
    actual: pd.Series,
    predicted: pd.Series,
) -> None:
    """Validate metric inputs."""
    if actual.empty or predicted.empty:
        raise ValueError("Actual and predicted values must not be empty.")

    if len(actual) != len(predicted):
        raise ValueError(
            "Actual and predicted values must contain the same number of rows."
        )

    if not actual.index.equals(predicted.index):
        raise ValueError("Actual and predicted values must have identical indexes.")
