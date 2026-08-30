"""Integration tests for the data pipeline."""

from __future__ import annotations

import pandas as pd

from fx_forecast.data.pipeline import run_pipeline
from fx_forecast.data.providers.base import FXProvider
from fx_forecast.data.validate import DataSchema


class FakeProvider(FXProvider):
    """Small in-memory provider for integration testing."""

    @property
    def schema(self) -> DataSchema:
        """Return the expected test schema."""
        return DataSchema(
            required_columns=("Open", "High", "Low", "Close", "Volume"),
            numeric_columns=("Open", "High", "Low", "Close", "Volume"),
        )

    def fetch(
        self,
        pair: str,
        start: str,
        end: str | None = None,
    ) -> pd.DataFrame:
        """Return representative FX observations."""
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


def test_data_pipeline_integrates_all_stages(tmp_path, monkeypatch) -> None:
    """Pipeline should fetch, validate, preprocess and save data."""

    monkeypatch.setattr(
        "fx_forecast.data.pipeline.PROCESSED_DATA_DIR",
        tmp_path,
    )

    result = run_pipeline(
        provider=FakeProvider(),
        pair="USD/NGN",
        start="2025-01-01",
        provider_name="fake",
    )

    output = tmp_path / "fake" / "USD_NGN.csv"

    assert not result.empty
    assert output.exists()
