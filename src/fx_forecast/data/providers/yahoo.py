"""Yahoo Finance FX provider."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import yfinance as yf

from fx_forecast.data.providers.base import FXProvider
from fx_forecast.data.validate import DataSchema


class YahooFXProvider(FXProvider):
    """Fetch FX data from Yahoo Finance."""

    @property
    def schema(self) -> DataSchema:
        """Return the Yahoo Finance OHLCV schema."""

        return DataSchema(
            required_columns=(
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
            ),
            numeric_columns=(
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
            ),
        )

    def fetch(
        self,
        pair: str,
        start: str,
        end: str | None = None,
    ) -> pd.DataFrame:
        """Download historical FX data."""

        symbol = {
            "USD/NGN": "USDNGN=X",
            "EUR/NGN": "EURNGN=X",
        }.get(pair, pair)

        end = end or datetime.today().strftime("%Y-%m-%d")

        df = yf.download(
            symbol,
            start=start,
            end=end,
            interval="1d",
            progress=False,
            auto_adjust=True,
        )

        if df is None or df.empty:
            raise ValueError(f"No data returned for {pair}")

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.columns.name = None

        return df
