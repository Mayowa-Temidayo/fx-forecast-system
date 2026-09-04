"""
Classical forecasting models.
"""

from __future__ import annotations

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from fx_forecast.models.base import BaseForecastModel


class ARIMAForecastModel(BaseForecastModel):
    """ARIMA forecasting model."""

    def __init__(
        self,
        order: tuple[int, int, int] = (1, 1, 1),
    ) -> None:
        """Initialize the ARIMA model."""

        self.order = order
        self._model = None
        self._fitted_model = None

    @property
    def name(self) -> str:
        """Return the model name."""

        return "arima"

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> ARIMAForecastModel:
        """Fit ARIMA to the target series."""

        if X.empty or y.empty:
            raise ValueError("Training data must not be empty.")

        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of rows.")

        self._model = ARIMA(y, order=self.order)
        self._fitted_model = self._model.fit()

        return self

    def predict(
        self,
        X: pd.DataFrame,
    ) -> pd.Series:
        """Generate ARIMA forecasts."""

        if self._fitted_model is None:
            raise RuntimeError("Model must be fitted before prediction.")

        if X.empty:
            raise ValueError("Prediction data must not be empty.")

        forecast = self._fitted_model.forecast(steps=len(X))

        return pd.Series(
            forecast.to_numpy(),
            index=X.index,
            name="prediction",
        )
