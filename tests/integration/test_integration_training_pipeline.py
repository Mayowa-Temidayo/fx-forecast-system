"""Integration tests for the training pipeline."""

from __future__ import annotations

import pandas as pd

from fx_forecast.data.split import split_time_series
from fx_forecast.models.naive import NaiveForecastModel
from fx_forecast.training.trainer import ModelTrainer


def test_split_and_train_pipeline() -> None:
    """Test chronological splitting followed by model training."""
    index = pd.date_range("2025-01-01", periods=10)

    X = pd.DataFrame(
        {"feature": range(10)},
        index=index,
    )

    y = pd.Series(
        [100.0 + i for i in range(10)],
        index=index,
        name="target",
    )

    split = split_time_series(
        X,
        y,
        train_size=0.6,
        validation_size=0.2,
    )

    result = ModelTrainer(
        NaiveForecastModel(),
    ).train(
        split.X_train,
        split.y_train,
        split.X_validation,
    )

    assert len(split.X_train) == 6
    assert len(split.X_validation) == 2
    assert len(split.X_test) == 2

    assert len(result.validation_predictions) == 2
    assert result.validation_predictions.index.equals(
        split.X_validation.index,
    )

    assert result.validation_predictions.iloc[0] == 105.0
    assert result.validation_predictions.iloc[1] == 105.0
