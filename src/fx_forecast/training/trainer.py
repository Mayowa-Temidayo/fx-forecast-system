"""Model training orchestration."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from fx_forecast.models.base import BaseForecastModel


@dataclass(frozen=True)
class TrainingResult:
    """Result produced after training a forecasting model."""

    model: BaseForecastModel
    validation_predictions: pd.Series


class ModelTrainer:
    """Train forecasting models and generate validation predictions."""

    def __init__(self, model: BaseForecastModel) -> None:
        """Initialize the trainer."""
        self.model = model

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_validation: pd.DataFrame,
    ) -> TrainingResult:
        """Fit the model and generate validation predictions."""
        if X_train.empty or y_train.empty:
            raise ValueError("Training data must not be empty.")

        if len(X_train) != len(y_train):
            raise ValueError(
                "X_train and y_train must contain the same number of rows."
            )

        if X_validation.empty:
            raise ValueError("Validation data must not be empty.")

        self.model.fit(X_train, y_train)

        predictions = self.model.predict(X_validation)

        return TrainingResult(
            model=self.model,
            validation_predictions=predictions,
        )
