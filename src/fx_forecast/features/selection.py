"""
Feature selection utilities.
"""

from __future__ import annotations

import pandas as pd

from fx_forecast.utils.logger import logger

TARGET_COLUMN = "target"

EXCLUDED_COLUMNS = {
    TARGET_COLUMN,
}


def select_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split an engineered dataset into a feature matrix (X)
    and target vector (y).

    Parameters
    ----------
    df : pd.DataFrame
        Engineered feature dataset.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        Feature matrix and target vector.
    """

    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column: '{TARGET_COLUMN}'")

    X = df.drop(columns=list(EXCLUDED_COLUMNS)).copy()
    y = df.loc[:, TARGET_COLUMN].copy()

    if X.empty:
        raise ValueError("No feature columns available.")

    logger.success(f"Selected {X.shape[1]} feature(s).")

    return X, pd.Series(y, name=TARGET_COLUMN)
