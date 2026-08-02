from __future__ import annotations

print("A. fetch.py loaded")

from pathlib import Path

print("B. pathlib")

import pandas as pd

print("C. pandas")

import yfinance as yf

print("D. yfinance")

from fx_forecast.config.paths import RAW_DATA_DIR

print("E. paths")

from fx_forecast.utils.logger import logger

print("F. logger")


def download_fx_data(
    ticker: str,
    start_date: str,
    interval: str = "1d",
    overwrite: bool = False,
) -> Path:
    print("download_fx_data called")
    return RAW_DATA_DIR / "test.parquet"
