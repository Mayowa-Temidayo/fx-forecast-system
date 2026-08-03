"""
Target engineering utilities.
"""

from __future__ import annotations

import pandas as pd

from fx_forecast.utils.logger import logger


def create_target_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create forecasting targets.

    Generated columns
    -----------------
    target
        Next-day closing price.

    target_3d
        Closing price 3 days ahead.

    target_5d
        Closing price 5 days ahead.

    future_return_1d
    future_return_3d
    future_return_5d
    """

    data = df.copy()

    close = data["Close"]

    # Future prices
    data["target"] = close.shift(-1)
    data["target_3d"] = close.shift(-3)
    data["target_5d"] = close.shift(-5)

    # Future returns
    data["future_return_1d"] = (data["target"] - close) / close

    data["future_return_3d"] = (data["target_3d"] - close) / close

    data["future_return_5d"] = (data["target_5d"] - close) / close

    logger.success("Target features created.")

    return data
