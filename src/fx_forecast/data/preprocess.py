"""Data preprocessing and canonicalization utilities."""

from __future__ import annotations

import pandas as pd

from fx_forecast.utils.logger import logger

CANONICAL_COLUMN_ORDER = (
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
)


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and canonicalize a validated dataframe."""

    rows_before = len(df)

    result = df.copy()

    # Normalize the dataframe index.
    if not isinstance(result.index, pd.DatetimeIndex):
        result.index = pd.to_datetime(result.index, errors="raise")

    result.index.name = "Date"

    # Normalize column names.
    result.columns = (
        result.columns.astype(str)
        .str.strip()
        .str.replace(" ", "_", regex=False)
        .str.title()
    )

    # Sort observations chronologically.
    result = result.sort_index()

    # Remove duplicate timestamps.
    result = result.loc[~result.index.duplicated(keep="first")]

    # Convert numeric-looking object columns.
    for column in result.columns:
        if result[column].dtype == "object":
            try:
                result[column] = pd.to_numeric(result[column])
            except (TypeError, ValueError):
                pass

    # Remove rows containing missing values.
    result = result.dropna()

    # Enforce the canonical market-column order when available.
    ordered_columns = [
        column for column in CANONICAL_COLUMN_ORDER if column in result.columns
    ]

    remaining_columns = [
        column for column in result.columns if column not in ordered_columns
    ]

    result = result.loc[:, ordered_columns + remaining_columns]

    rows_after = len(result)

    logger.success(f"Preprocessing complete ({rows_before} → {rows_after} rows).")

    return result
