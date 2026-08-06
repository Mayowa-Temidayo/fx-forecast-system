"""
EDA summary utilities.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from fx_forecast.utils.logger import logger


def dataset_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a high-level dataset summary.
    """

    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    memory_mb = float(df.memory_usage(deep=True).sum()) / (1024**2)

    summary = pd.DataFrame(
        {
            "Metric": [
                "Rows",
                "Columns",
                "Missing Values",
                "Duplicate Rows",
                "Memory (MB)",
            ],
            "Value": [
                len(df),
                df.shape[1],
                int(df.isna().sum().sum()),
                int(df.duplicated().sum()),
                round(memory_mb, 2),
            ],
        }
    )

    logger.success("Dataset summary generated.")

    return summary


def missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize missing values by column.
    """

    missing = df.isna().sum()

    percentage = (missing / len(df)) * 100

    summary = pd.DataFrame(
        {
            "missing": missing,
            "percentage": percentage,
        }
    )

    summary["percentage"] = summary["percentage"].round(2)

    logger.success("Missing-value summary generated.")

    return summary


def data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize column data types.
    """

    summary = df.dtypes.astype(str).rename("dtype").to_frame()

    logger.success("Data-type summary generated.")

    return summary


def memory_usage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Memory usage of the dataset.
    """

    memory_kb = float(df.memory_usage(deep=True).sum()) / 1024

    summary = pd.DataFrame(
        {
            "memory_kb": [round(memory_kb, 2)],
        }
    )

    logger.success("Memory usage calculated.")

    return summary


def generate_summary(df: pd.DataFrame) -> dict[str, Any]:
    """
    Generate all summary outputs.
    """

    logger.info("Generating dataset summary.")

    return {
        "dataset": dataset_summary(df),
        "missing": missing_values(df),
        "dtypes": data_types(df),
        "memory": memory_usage(df),
    }
