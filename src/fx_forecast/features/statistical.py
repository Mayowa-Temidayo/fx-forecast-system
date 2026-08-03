"""
Statistical feature engineering.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fx_forecast.utils.logger import logger


def create_statistical_features(
    df: pd.DataFrame,
    windows: tuple[int, ...] = (5, 10, 20),
) -> pd.DataFrame:
    """
    Create statistical features from closing prices.

    Parameters
    ----------
    df : pd.DataFrame
        Market data.
    windows : tuple[int, ...]
        Rolling window sizes.

    Returns
    -------
    pd.DataFrame
        DataFrame with statistical features.
    """

    if "Close" not in df.columns:
        raise ValueError("Column 'Close' not found.")

    result = df.copy()

    close = result["Close"]

    # Returns
    result["return"] = close.pct_change()
    result["log_return"] = np.log(close / close.shift(1))

    for window in windows:
        rolling = close.rolling(window)

        result[f"rolling_mean_{window}"] = rolling.mean()
        result[f"rolling_std_{window}"] = rolling.std()
        result[f"rolling_min_{window}"] = rolling.min()
        result[f"rolling_max_{window}"] = rolling.max()

    logger.success("Statistical features created.")

    return result
