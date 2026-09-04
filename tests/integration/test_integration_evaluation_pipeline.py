"""Integration tests for the evaluation pipeline."""

from __future__ import annotations

import pandas as pd

from fx_forecast.evaluation.metrics import (
    directional_accuracy,
    mean_absolute_error,
    root_mean_squared_error,
)
from fx_forecast.models.naive import NaiveForecastModel
from fx_forecast.training.trainer import ModelTrainer


def test_training_predictions_can_be_evaluated() -> None:
    """Test model training followed by forecast evaluation."""
    index = pd.date_range("2025-01-01", periods=10)

    X_train = pd.DataFrame(
        {"feature": range(6)},
        index=index[:6],
    )

    y_train = pd.Series(
        [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
        index=index[:6],
        name="target",
    )

    X_validation = pd.DataFrame(
        {"feature": range(6, 10)},
        index=index[6:],
    )

    y_validation = pd.Series(
        [106.0, 107.0, 108.0, 109.0],
        index=index[6:],
        name="target",
    )

    result = ModelTrainer(
        NaiveForecastModel(),
    ).train(
        X_train,
        y_train,
        X_validation,
    )

    mae = mean_absolute_error(
        y_validation,
        result.validation_predictions,
    )

    rmse = root_mean_squared_error(
        y_validation,
        result.validation_predictions,
    )

    accuracy = directional_accuracy(
        y_validation,
        result.validation_predictions,
    )

    assert mae == 2.5
    assert rmse > 0
    assert accuracy == 0.0
