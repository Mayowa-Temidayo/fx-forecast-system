"""Gradient boosting forecasting models."""

from __future__ import annotations

from typing import Any

import pandas as pd
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from fx_forecast.models.base import BaseForecastModel


class XGBoostModel(BaseForecastModel):
    """XGBoost regression model for FX forecasting."""

    def __init__(self, **params: Any) -> None:
        """Initialize the XGBoost model."""
        self.model = XGBRegressor(**params)
        self._fitted = False

    @property
    def name(self) -> str:
        """Return the model name."""
        return "xgboost"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> XGBoostModel:
        """Fit the model."""
        self.model.fit(X, y)
        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Generate predictions."""
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction.")

        predictions = self.model.predict(X)

        return pd.Series(
            predictions,
            index=X.index,
            name="prediction",
        )


class LightGBMModel(BaseForecastModel):
    """LightGBM regression model for FX forecasting."""

    def __init__(self, **params: Any) -> None:
        """Initialize the LightGBM model."""
        self.model = LGBMRegressor(**params)
        self._fitted = False

    @property
    def name(self) -> str:
        """Return the model name."""
        return "lightgbm"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> LightGBMModel:
        """Fit the model."""
        self.model.fit(X, y)
        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Generate predictions."""
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction.")

        predictions = self.model.predict(X)

        return pd.Series(
            predictions,
            index=X.index,
            name="prediction",
        )
