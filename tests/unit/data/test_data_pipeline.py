"""
Unit tests for the data pipeline.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from fx_forecast.data.pipeline import run_pipeline


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


@patch("fx_forecast.data.pipeline.save_dataframe")
@patch("fx_forecast.data.pipeline.preprocess_dataframe")
@patch("fx_forecast.data.pipeline.validate_dataframe")
@patch("fx_forecast.data.pipeline.DataFetcher")
def test_run_pipeline_success(
    mock_fetcher_cls: MagicMock,
    mock_validate: MagicMock,
    mock_preprocess: MagicMock,
    mock_save: MagicMock,
    sample_dataframe: pd.DataFrame,
) -> None:
    """
    Pipeline should execute every stage successfully.
    """

    fetcher = mock_fetcher_cls.return_value
    fetcher.download.return_value = sample_dataframe

    mock_validate.return_value = sample_dataframe
    mock_preprocess.return_value = sample_dataframe

    result = run_pipeline(
        symbol="EURUSD=X",
        start="2025-01-01",
        end="2025-01-10",
    )

    fetcher.download.assert_called_once_with(
        symbol="EURUSD=X",
        start="2025-01-01",
        end="2025-01-10",
        interval="1d",
    )

    mock_validate.assert_called_once_with(sample_dataframe)
    mock_preprocess.assert_called_once_with(sample_dataframe)
    mock_save.assert_called_once()

    pd.testing.assert_frame_equal(result, sample_dataframe)


@patch("fx_forecast.data.pipeline.DataFetcher")
def test_pipeline_propagates_fetch_error(
    mock_fetcher_cls: MagicMock,
) -> None:
    """
    Fetch failures should propagate.
    """

    fetcher = mock_fetcher_cls.return_value
    fetcher.download.side_effect = RuntimeError("Download failed")

    with pytest.raises(RuntimeError, match="Download failed"):
        run_pipeline(
            "EURUSD=X",
            "2025-01-01",
        )


@patch("fx_forecast.data.pipeline.validate_dataframe")
@patch("fx_forecast.data.pipeline.DataFetcher")
def test_pipeline_propagates_validation_error(
    mock_fetcher_cls: MagicMock,
    mock_validate: MagicMock,
    sample_dataframe: pd.DataFrame,
) -> None:
    """
    Validation failures should propagate.
    """

    fetcher = mock_fetcher_cls.return_value
    fetcher.download.return_value = sample_dataframe

    mock_validate.side_effect = ValueError("Validation failed")

    with pytest.raises(ValueError, match="Validation failed"):
        run_pipeline(
            "EURUSD=X",
            "2025-01-01",
        )


@patch("fx_forecast.data.pipeline.preprocess_dataframe")
@patch("fx_forecast.data.pipeline.validate_dataframe")
@patch("fx_forecast.data.pipeline.DataFetcher")
def test_pipeline_propagates_preprocess_error(
    mock_fetcher_cls: MagicMock,
    mock_validate: MagicMock,
    mock_preprocess: MagicMock,
    sample_dataframe: pd.DataFrame,
) -> None:
    """
    Preprocessing failures should propagate.
    """

    fetcher = mock_fetcher_cls.return_value
    fetcher.download.return_value = sample_dataframe

    mock_validate.return_value = sample_dataframe
    mock_preprocess.side_effect = ValueError("Preprocess failed")

    with pytest.raises(ValueError, match="Preprocess failed"):
        run_pipeline(
            "EURUSD=X",
            "2025-01-01",
        )


@patch("fx_forecast.data.pipeline.save_dataframe")
@patch("fx_forecast.data.pipeline.preprocess_dataframe")
@patch("fx_forecast.data.pipeline.validate_dataframe")
@patch("fx_forecast.data.pipeline.DataFetcher")
def test_processed_data_is_saved(
    mock_fetcher_cls: MagicMock,
    mock_validate: MagicMock,
    mock_preprocess: MagicMock,
    mock_save: MagicMock,
    sample_dataframe: pd.DataFrame,
) -> None:
    """
    Pipeline should save the processed dataframe.
    """

    fetcher = mock_fetcher_cls.return_value
    fetcher.download.return_value = sample_dataframe

    mock_validate.return_value = sample_dataframe
    mock_preprocess.return_value = sample_dataframe

    run_pipeline(
        symbol="EURUSD=X",
        start="2025-01-01",
    )

    _, kwargs = mock_save.call_args

    pd.testing.assert_frame_equal(kwargs["df"], sample_dataframe)

    assert kwargs["path"].name == "EURUSD_X.csv"
