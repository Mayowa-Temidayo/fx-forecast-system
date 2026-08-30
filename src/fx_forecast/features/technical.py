"""
Technical indicator features.
"""

from __future__ import annotations

import pandas as pd

from fx_forecast.utils.logger import logger


def create_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create technical indicator features.

    Features
    --------
    Simple Moving Averages
        - sma_5
        - sma_10
        - sma_20

    Exponential Moving Averages
        - ema_5
        - ema_10
        - ema_20
    """

    data = df.copy()

    if "Close" not in data.columns:
        raise ValueError("Column 'Close' not found.")

    close = data["Close"]

    # Simple Moving Averages
    data["sma_5"] = close.rolling(window=5).mean()
    data["sma_10"] = close.rolling(window=10).mean()
    data["sma_20"] = close.rolling(window=20).mean()

    # Exponential Moving Averages
    data["ema_5"] = close.ewm(span=5, adjust=False).mean()
    data["ema_10"] = close.ewm(span=10, adjust=False).mean()
    data["ema_20"] = close.ewm(span=20, adjust=False).mean()

    logger.success("Technical indicators created.")

    return data
