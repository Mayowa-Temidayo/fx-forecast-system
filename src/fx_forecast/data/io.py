"""
Data I/O utilities.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fx_forecast.utils.logger import logger

DEFAULT_FORMAT = "csv"  # Change to "parquet" later if desired.


def ensure_directory(path: Path) -> None:
    """Create directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def save_dataframe(
    df: pd.DataFrame,
    path: Path,
    file_format: str = DEFAULT_FORMAT,
) -> None:
    """Save a DataFrame."""

    ensure_directory(path.parent)

    if file_format == "parquet":
        path = path.with_suffix(".parquet")
        df.to_parquet(path, index=True)
    elif file_format == "csv":
        path = path.with_suffix(".csv")
        df.to_csv(path, index=True)
    else:
        raise ValueError(f"Unsupported format: {file_format}")

    logger.success(f"Saved dataset -> {path}")


def load_dataframe(
    path: Path,
    file_format: str = DEFAULT_FORMAT,
) -> pd.DataFrame:
    """Load a DataFrame."""

    if file_format == "parquet":
        path = path.with_suffix(".parquet")
        return pd.read_parquet(path)

    if file_format == "csv":
        path = path.with_suffix(".csv")
        return pd.read_csv(path, index_col=0)

    raise ValueError(f"Unsupported format: {file_format}")
