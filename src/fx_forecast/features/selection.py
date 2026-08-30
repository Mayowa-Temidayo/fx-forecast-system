"""
Feature selection utilities.
"""

from __future__ import annotations

import pandas as pd

from fx_forecast.utils.logger import logger

TARGET_COLUMN = "target"

EXCLUDED_COLUMNS = {
    "target",
    "target_3d",
    "target_5d",
    "future_return_1d",
    "future_return_3d",
    "future_return_5d",
}


def select_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split engineered data into features and target."""

    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column: '{TARGET_COLUMN}'")

    excluded = [column for column in EXCLUDED_COLUMNS if column in df.columns]

    X = df.drop(columns=excluded).copy()
    y = df.loc[:, TARGET_COLUMN].copy()

    if X.empty:
        raise ValueError("No feature columns available.")

    logger.success(f"Selected {X.shape[1]} feature(s).")

    return X, pd.Series(y, name=TARGET_COLUMN)
