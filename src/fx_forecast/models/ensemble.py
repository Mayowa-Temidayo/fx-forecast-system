"""Ensemble forecasting models."""

from __future__ import annotations

import pandas as pd

from fx_forecast.models.base import BaseForecastModel


class WeightedEnsembleModel(BaseForecastModel):
    """Combine predictions from multiple forecasting models using weights."""

    def __init__(
        self,
        models: dict[str, BaseForecastModel],
        weights: dict[str, float] | None = None,
    ) -> None:
        """Initialize the weighted ensemble."""
        if not models:
            raise ValueError("At least one model is required.")

        self.models = models
        self.weights = weights or self._equal_weights()
        self._validate_weights()

    @property
    def name(self) -> str:
        """Return the model name."""
        return "weighted_ensemble"

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> WeightedEnsembleModel:
        """Fit every model in the ensemble."""
        for model in self.models.values():
            model.fit(X, y)

        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Generate weighted ensemble predictions."""
        predictions = []

        for name, model in self.models.items():
            prediction = model.predict(X)
            predictions.append(prediction * self.weights[name])

        result = sum(predictions)

        return pd.Series(
            result,
            index=X.index,
            name="prediction",
        )

    def _equal_weights(self) -> dict[str, float]:
        """Create equal weights for all models."""
        weight = 1.0 / len(self.models)

        return {name: weight for name in self.models}

    def _validate_weights(self) -> None:
        """Validate ensemble weights."""
        if set(self.weights) != set(self.models):
            raise ValueError("Weights must match the ensemble models.")

        if any(weight < 0 for weight in self.weights.values()):
            raise ValueError("Weights must be non-negative.")

        if sum(self.weights.values()) != 1.0:
            raise ValueError("Weights must sum to 1.0.")
