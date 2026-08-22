"""Data preprocessing and canonicalization utilities."""

from __future__ import annotations

import pandas as pd

from fx_forecast.utils.logger import logger


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and canonicalize a validated dataframe.

    The preprocessing stage is provider-agnostic.

    The canonical dataframe produced here has:
    - a DatetimeIndex;
    - an index named ``Date``;
    - chronologically sorted observations;
    - unique timestamps;
    - normalized column names;
    - numeric-looking object columns converted to numeric;
    - no missing values.
    """

    rows_before = len(df)

    result = df.copy()

    # Normalize the index.
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

    # Sort chronologically.
    result = result.sort_index()

    # Remove duplicate timestamps while keeping the first observation.
    result = result.loc[~result.index.duplicated(keep="first")]

    # Convert numeric-looking object columns to numeric.
    #
    # Non-numeric columns such as Pair remain unchanged.
    for column in result.columns:
        if result[column].dtype == "object":
            try:
                result[column] = pd.to_numeric(result[column])
            except (TypeError, ValueError):
                pass

    # Do not invent financial observations through mean/median/mode
    # imputation. Rows containing missing values are removed.
    result = result.dropna()

    rows_after = len(result)

    logger.success(f"Preprocessing complete ({rows_before} -> {rows_after} rows).")

    return result
