"""
Data validation utilities.
"""

from __future__ import annotations

import pandas as pd

from fx_forecast.utils.logger import logger

REQUIRED_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
]


def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate a downloaded market dataset.

    Returns
    -------
    pd.DataFrame
        The validated DataFrame.
    """

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Index must be a DatetimeIndex.")

    if not df.index.is_monotonic_increasing:
        raise ValueError("Index must be sorted in ascending order.")

    if df.index.has_duplicates:
        raise ValueError("Duplicate timestamps found.")

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    numeric_columns = ["Open", "High", "Low", "Close", "Volume"]

    for column in numeric_columns:
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise TypeError(f"{column} must be numeric.")

    empty_columns: list[str] = []

    for column in df.columns:
        if bool(df[column].isna().all()):
            empty_columns.append(column)

    if empty_columns:
        raise ValueError(f"Empty columns detected: {empty_columns}")

    logger.success("Dataset validation passed.")

    return df
