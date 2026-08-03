"""
Data I/O utilities.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fx_forecast.utils.logger import logger

DEFAULT_FORMAT = "csv"


def ensure_directory(path: Path) -> None:
    """Create a directory if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)


def save_dataframe(
    df: pd.DataFrame,
    path: Path,
    file_format: str = DEFAULT_FORMAT,
) -> None:
    """
    Save a DataFrame to disk.

    Parameters
    ----------
    df
        DataFrame to save.
    path
        Destination path without relying on the caller to provide
        the correct suffix.
    file_format
        Either "csv" or "parquet".
    """

    file_format = file_format.lower()

    ensure_directory(path.parent)

    match file_format:
        case "csv":
            path = path.with_suffix(".csv")
            df.to_csv(path, index=True)

        case "parquet":
            path = path.with_suffix(".parquet")
            df.to_parquet(path, index=True)

        case _:
            raise ValueError(f"Unsupported file format: {file_format}")

    logger.success(f"Saved dataset -> {path}")


def load_dataframe(
    path: Path,
    file_format: str = DEFAULT_FORMAT,
) -> pd.DataFrame:
    """
    Load a DataFrame from disk.

    Parameters
    ----------
    path
        File path without requiring the correct suffix.
    file_format
        Either "csv" or "parquet".

    Returns
    -------
    pd.DataFrame
    """

    file_format = file_format.lower()

    match file_format:
        case "csv":
            path = path.with_suffix(".csv")

        case "parquet":
            path = path.with_suffix(".parquet")

        case _:
            raise ValueError(f"Unsupported file format: {file_format}")

    if not path.exists():
        raise FileNotFoundError(path)

    logger.info(f"Loading dataset <- {path}")

    match file_format:
        case "csv":
            df = pd.read_csv(
                path,
                index_col=0,
                parse_dates=[0],
            )

            df.index.name = None

            return df

        case "parquet":
            return pd.read_parquet(path)

    # Defensive fallback (should never be reached)
    raise RuntimeError("Unexpected file format.")
