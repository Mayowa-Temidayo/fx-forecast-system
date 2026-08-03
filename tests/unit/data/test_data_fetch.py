"""
Unit tests for data fetching.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from fx_forecast.data.fetch import DataFetcher


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Return a valid market dataframe."""

    index = pd.date_range("2025-01-01", periods=3)

    return pd.DataFrame(
        {
            "Open": [1.10, 1.20, 1.30],
            "High": [1.15, 1.25, 1.35],
            "Low": [1.05, 1.15, 1.25],
            "Close": [1.12, 1.22, 1.32],
            "Volume": [1000, 1100, 1200],
        },
        index=index,
    )


def test_download_success(
    tmp_path: Path,
    sample_dataframe: pd.DataFrame,
) -> None:
    """
    Download should return a dataframe and save it.
    """

    fetcher = DataFetcher(tmp_path)

    with patch(
        "fx_forecast.data.fetch.yf.download",
        return_value=sample_dataframe,
    ):
        df = fetcher.download(
            symbol="EURUSD=X",
            start="2025-01-01",
            end="2025-01-10",
        )

    pd.testing.assert_frame_equal(df, sample_dataframe)

    assert (tmp_path / "EURUSD_X.csv").exists()


def test_download_none(tmp_path: Path) -> None:
    """
    None returned from yfinance should raise RuntimeError.
    """

    fetcher = DataFetcher(tmp_path)

    with patch(
        "fx_forecast.data.fetch.yf.download",
        return_value=None,
    ):
        with pytest.raises(RuntimeError, match="returned None"):
            fetcher.download(
                "EURUSD=X",
                "2025-01-01",
                "2025-01-10",
            )


def test_download_empty_dataframe(tmp_path: Path) -> None:
    """
    Empty dataframe should raise ValueError.
    """

    fetcher = DataFetcher(tmp_path)

    with patch(
        "fx_forecast.data.fetch.yf.download",
        return_value=pd.DataFrame(),
    ):
        with pytest.raises(ValueError, match="No data returned"):
            fetcher.download(
                "EURUSD=X",
                "2025-01-01",
                "2025-01-10",
            )


def test_multiindex_columns_are_flattened(tmp_path: Path) -> None:
    """
    MultiIndex columns from yfinance should be flattened.
    """

    index = pd.date_range("2025-01-01", periods=2)

    columns = pd.MultiIndex.from_product(
        [
            ["Open", "High", "Low", "Close", "Volume"],
            ["EURUSD=X"],
        ],
        names=["Price", "Ticker"],
    )

    df = pd.DataFrame(
        [
            [1.10, 1.15, 1.05, 1.12, 1000],
            [1.20, 1.25, 1.15, 1.22, 1100],
        ],
        index=index,
        columns=columns,
    )

    fetcher = DataFetcher(tmp_path)

    with patch(
        "fx_forecast.data.fetch.yf.download",
        return_value=df,
    ):
        result = fetcher.download(
            "EURUSD=X",
            "2025-01-01",
            "2025-01-10",
        )

    assert list(result.columns) == [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    assert result.columns.name is None


def test_symbol_filename_conversion(
    tmp_path: Path,
    sample_dataframe: pd.DataFrame,
) -> None:
    """
    '=' should be replaced with '_' when saving.
    """

    fetcher = DataFetcher(tmp_path)

    with patch(
        "fx_forecast.data.fetch.yf.download",
        return_value=sample_dataframe,
    ):
        fetcher.download(
            "EURUSD=X",
            "2025-01-01",
            "2025-01-10",
        )

    assert (tmp_path / "EURUSD_X.csv").exists()
