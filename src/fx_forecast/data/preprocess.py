"""Data preprocessing utilities."""

from __future__ import annotations

import pandas as pd

from fx_forecast.utils.logger import logger


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean a validated market dataset."""

    rows_before = len(df)

    df = df.copy()

    df = df.sort_index()

    df = df.loc[~df.index.duplicated(keep="first")]

    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace(" ", "_", regex=False)
        .str.title()
    )

    df = df.ffill().bfill()
    df = df.dropna()

    rows_after = len(df)

    logger.success(f"Preprocessing complete ({rows_before} → {rows_after} rows).")

    return df
