"""
Unit tests for data I/O helpers.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from fx_forecast.data.io import (
    load_dataframe,
    save_dataframe,
)


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Create a sample OHLCV DataFrame."""

    return pd.DataFrame(
        {
            "Open": [1.10, 1.20, 1.30],
            "High": [1.15, 1.25, 1.35],
            "Low": [1.05, 1.15, 1.25],
            "Close": [1.12, 1.22, 1.32],
            "Volume": [1000, 1100, 1200],
        },
        index=pd.date_range("2025-01-01", periods=3, freq="D"),
    )


def test_save_and_load_csv(
    tmp_path: Path,
    sample_dataframe: pd.DataFrame,
) -> None:
    """Saving and loading a CSV should preserve the dataset."""

    filepath = tmp_path / "sample.csv"

    save_dataframe(sample_dataframe, filepath)

    loaded = load_dataframe(filepath)

    assert_frame_equal(
        sample_dataframe,
        loaded,
        check_freq=False,
    )


def test_save_creates_file(
    tmp_path: Path,
    sample_dataframe: pd.DataFrame,
) -> None:
    """Saving should create the destination file."""

    filepath = tmp_path / "created.csv"

    save_dataframe(sample_dataframe, filepath)

    assert filepath.exists()


def test_load_missing_file(tmp_path: Path) -> None:
    """Loading a missing file should raise FileNotFoundError."""

    filepath = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError):
        load_dataframe(filepath)


@pytest.mark.parametrize("extension", [".csv"])
def test_supported_extensions(
    tmp_path: Path,
    sample_dataframe: pd.DataFrame,
    extension: str,
) -> None:
    """Supported formats should save and load successfully."""

    filepath = tmp_path / f"dataset{extension}"

    save_dataframe(sample_dataframe, filepath)

    loaded = load_dataframe(filepath)

    assert_frame_equal(
        sample_dataframe,
        loaded,
        check_freq=False,
    )
