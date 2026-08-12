"""FX data provider contract."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from fx_forecast.data.validate import DataSchema


class FXProvider(ABC):
    """Base contract for FX data providers."""

    @property
    @abstractmethod
    def schema(self) -> DataSchema:
        """Return the provider's dataframe schema."""

    @abstractmethod
    def fetch(
        self,
        pair: str,
        start: str,
        end: str | None = None,
    ) -> pd.DataFrame:
        """Fetch FX observations."""
