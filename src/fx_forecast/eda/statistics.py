"""
EDA statistical utilities.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from fx_forecast.utils.logger import logger


def descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Descriptive statistics for numeric columns.
    """

    numeric = df.select_dtypes(include=np.number)

    if numeric.empty:
        raise ValueError("No numeric columns found.")

    logger.success("Descriptive statistics generated.")

    return numeric.describe().T


def skewness(df: pd.DataFrame) -> pd.Series:
    """
    Compute skewness.
    """

    numeric = df.select_dtypes(include=np.number)

    logger.success("Skewness calculated.")

    result = numeric.skew()

    assert isinstance(result, pd.Series)

    return result


def kurtosis(df: pd.DataFrame) -> pd.Series:
    """
    Compute kurtosis.
    """

    numeric = df.select_dtypes(include=np.number)

    logger.success("Kurtosis calculated.")

    result = numeric.kurt()
    assert isinstance(result, pd.Series)
    return result


def correlation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pearson correlation matrix.
    """

    numeric = df.select_dtypes(include=np.number)

    logger.success("Correlation matrix generated.")

    return numeric.corr()


def generate_statistics(df: pd.DataFrame) -> dict[str, Any]:
    """
    Run all statistical summaries.
    """

    logger.info("Generating statistical analysis.")

    return {
        "describe": descriptive_statistics(df),
        "skewness": skewness(df),
        "kurtosis": kurtosis(df),
        "correlation": correlation(df),
    }
