"""
Time-series dataset splitting utilities.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TimeSeriesSplit:
    """Chronological train, validation, and test datasets."""

    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_test: pd.DataFrame

    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series


def split_time_series(
    X: pd.DataFrame,
    y: pd.Series,
    train_size: float = 0.70,
    validation_size: float = 0.15,
) -> TimeSeriesSplit:
    """
    Split feature data chronologically into train, validation, and test sets.

    Parameters
    ----------
    X
        Feature matrix.
    y
        Target series.
    train_size
        Fraction of observations assigned to training.
    validation_size
        Fraction of observations assigned to validation.

    Returns
    -------
    TimeSeriesSplit
        Chronological train, validation, and test datasets.
    """

    if X.empty or y.empty:
        raise ValueError("X and y must not be empty.")

    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of rows.")

    if not 0 < train_size < 1:
        raise ValueError("train_size must be between 0 and 1.")

    if not 0 < validation_size < 1:
        raise ValueError("validation_size must be between 0 and 1.")

    if train_size + validation_size >= 1:
        raise ValueError("train_size + validation_size must be less than 1.")

    if not X.index.equals(y.index):
        raise ValueError("X and y must have identical indexes.")

    train_end = int(len(X) * train_size)
    validation_end = train_end + int(len(X) * validation_size)

    if train_end == 0 or validation_end >= len(X):
        raise ValueError("Split sizes produce an empty dataset.")

    return TimeSeriesSplit(
        X_train=X.iloc[:train_end].copy(),
        X_validation=X.iloc[train_end:validation_end].copy(),
        X_test=X.iloc[validation_end:].copy(),
        y_train=y.iloc[:train_end].copy(),
        y_validation=y.iloc[train_end:validation_end].copy(),
        y_test=y.iloc[validation_end:].copy(),
    )
