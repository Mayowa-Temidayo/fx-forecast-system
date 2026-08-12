"""Data validation utilities."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from fx_forecast.utils.logger import logger


@dataclass(frozen=True)
class DataSchema:
    """Describe the required structure of a provider dataset."""

    required_columns: tuple[str, ...]
    numeric_columns: tuple[str, ...] = ()


def validate_dataframe(
    df: pd.DataFrame,
    schema: DataSchema,
) -> pd.DataFrame:
    """Validate a dataframe against a provider-specific schema."""

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Index must be a DatetimeIndex.")

    if df.empty:
        raise ValueError("Dataset must not be empty.")

    if not df.index.is_monotonic_increasing:
        raise ValueError("Index must be sorted in ascending order.")

    if df.index.has_duplicates:
        raise ValueError("Duplicate timestamps found.")

    missing = [column for column in schema.required_columns if column not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    empty_columns = [
        column for column in schema.required_columns if bool(df[column].isna().all())
    ]

    if empty_columns:
        raise ValueError(f"Empty columns detected: {empty_columns}")

    for column in schema.numeric_columns:
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise TypeError(f"{column} must be numeric.")

    logger.success("Dataset validation passed.")

    return df
