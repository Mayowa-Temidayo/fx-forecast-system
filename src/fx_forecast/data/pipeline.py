"""
End-to-end data pipeline.
"""

from __future__ import annotations

import pandas as pd

from fx_forecast.config.paths import PROCESSED_DATA_DIR, RAW_DATA_DIR
from fx_forecast.data.fetch import DataFetcher
from fx_forecast.data.io import save_dataframe
from fx_forecast.data.preprocess import preprocess_dataframe
from fx_forecast.data.validate import validate_dataframe
from fx_forecast.utils.logger import logger


def run_pipeline(
    symbol: str,
    start: str,
    end: str | None = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Download, validate, preprocess and save market data.
    """

    logger.info(f"Starting pipeline for {symbol}")

    fetcher = DataFetcher(RAW_DATA_DIR)

    df = fetcher.download(
        symbol=symbol,
        start=start,
        end=end,
        interval=interval,
    )

    df = validate_dataframe(df)
    df = preprocess_dataframe(df)

    output_path = PROCESSED_DATA_DIR / f"{symbol.replace('=', '_')}.csv"

    save_dataframe(
        df=df,
        path=output_path,
    )

    logger.success(f"Pipeline completed for {symbol}")

    return df
