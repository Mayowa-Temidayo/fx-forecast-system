"""
Base forecasting model interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseForecastModel(ABC):
    """Abstract interface for all forecasting models."""

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> BaseForecastModel:
        """Fit the model on training data."""
        raise NotImplementedError

    @abstractmethod
    def predict(
        self,
        X: pd.DataFrame,
    ) -> pd.Series:
        """Generate predictions."""
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the model name."""
        raise NotImplementedError
