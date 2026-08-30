"""Tests for FX data ingestion orchestration."""

from unittest.mock import MagicMock, patch

from fx_forecast.data.ingestion import ingest_all


@patch("fx_forecast.data.ingestion.run_pipeline")
@patch("fx_forecast.data.ingestion.get_provider")
def test_ingest_all_runs_each_provider_and_pair(
    mock_get_provider: MagicMock,
    mock_run_pipeline: MagicMock,
) -> None:
    """Ingestion should run every provider/pair combination."""

    provider = MagicMock()
    mock_get_provider.return_value = provider

    ingest_all(
        providers=["yahoo", "cbn"],
        pairs=["USD/NGN", "EUR/NGN"],
        start="2025-01-01",
    )

    assert mock_get_provider.call_count == 2
    assert mock_run_pipeline.call_count == 4
