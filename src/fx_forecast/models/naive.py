"""
Naive forecasting baseline.
"""

from __future__ import annotations

import pandas as pd

from fx_forecast.models.base import BaseForecastModel


class NaiveForecastModel(BaseForecastModel):
    """Forecast using the most recently observed target value."""

    def __init__(self) -> None:
        """Initialize the model."""
        self._last_value: float | None = None

    @property
    def name(self) -> str:
        """Return the model name."""
        return "naive"

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> NaiveForecastModel:
        """Fit the model using the final observed target value."""

        if X.empty or y.empty:
            raise ValueError("Training data must not be empty.")

        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of rows.")

        self._last_value = float(y.iloc[-1])

        return self

    def predict(
        self,
        X: pd.DataFrame,
    ) -> pd.Series:
        """Generate naive predictions."""

        if self._last_value is None:
            raise RuntimeError("Model must be fitted before prediction.")

        if X.empty:
            raise ValueError("Prediction data must not be empty.")

        return pd.Series(
            self._last_value,
            index=X.index,
            name="prediction",
        )
