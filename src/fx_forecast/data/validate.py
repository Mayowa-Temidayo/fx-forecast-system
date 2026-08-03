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

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    empty_columns: list[str] = []

    for column in REQUIRED_COLUMNS:
        if bool(df[column].isna().all()):
            empty_columns.append(column)

    if empty_columns:
        raise ValueError(f"Empty columns detected: {empty_columns}")

    for column in REQUIRED_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise TypeError(f"{column} must be numeric.")

    logger.success("Dataset validation passed.")

    return df
