"""
Data fetching utilities.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

from fx_forecast.data.io import save_dataframe
from fx_forecast.utils.logger import logger


class DataFetcher:
    """Download historical FX data from Yahoo Finance."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir

    def download(
        self,
        symbol: str,
        start: str,
        end: str | None = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Download historical price data."""

        if end is None:
            end = datetime.today().strftime("%Y-%m-%d")

        logger.info(f"Downloading {symbol} ({start} → {end})")

        df = yf.download(
            tickers=symbol,
            start=start,
            end=end,
            interval=interval,
            progress=False,
            auto_adjust=True,
        )

        if df is None:
            raise RuntimeError(f"Yahoo Finance returned None for {symbol}")

        if df.empty:
            raise ValueError(f"No data returned for {symbol}")

        filename = symbol.replace("=", "_")

        save_dataframe(
            df=df,
            path=self.output_dir / filename,
        )

        logger.success(f"{symbol}: {len(df)} rows downloaded")

        return df
