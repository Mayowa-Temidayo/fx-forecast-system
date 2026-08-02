"""
Data preprocessing utilities.
"""

from __future__ import annotations

import pandas as pd

from fx_forecast.utils.logger import logger


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean a validated market dataset.

    Steps
    -----
    1. Sort index
    2. Remove duplicate timestamps
    3. Standardize column names
    4. Forward-fill missing values
    5. Drop remaining missing rows
    """

    df = df.copy()

    # Sort by datetime index
    df = df.sort_index()

    # Remove duplicate timestamps
    df = df.loc[~df.index.duplicated(keep="first")]

    # Standardize column names
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace(" ", "_", regex=False)
        .str.title()
    )

    # Fill missing values
    # 1. Forward-fill gaps
    # 2. Backward-fill any remaining leading gaps
    df = df.ffill().bfill()

    # Remove any rows that still contain missing values
    df = df.dropna()

    rows_before = len(df)

    # Fill missing values
    df = df.ffill().bfill()

    # Remove any rows that still contain missing values
    df = df.dropna()

    rows_after = len(df)

    logger.success(f"Preprocessing complete ({rows_before} → {rows_after} rows).")
    return df
