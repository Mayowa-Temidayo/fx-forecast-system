"""End-to-end FX data pipeline."""

from __future__ import annotations

import pandas as pd

from fx_forecast.config.paths import PROCESSED_DATA_DIR
from fx_forecast.data.io import save_dataframe
from fx_forecast.data.preprocess import preprocess_dataframe
from fx_forecast.data.providers.base import FXProvider
from fx_forecast.data.validate import validate_dataframe
from fx_forecast.utils.logger import logger


def run_pipeline(
    provider: FXProvider,
    pair: str,
    start: str,
    end: str | None = None,
    provider_name: str = "unknown",
) -> pd.DataFrame:
    """Fetch, validate, preprocess and save FX market data."""

    logger.info(f"Starting pipeline for {provider_name}/{pair}")

    df = provider.fetch(
        pair=pair,
        start=start,
        end=end,
    )

    df = validate_dataframe(
        df,
        schema=provider.schema,
    )

    df = preprocess_dataframe(df)

    provider_dir = PROCESSED_DATA_DIR / provider_name
    provider_dir.mkdir(parents=True, exist_ok=True)

    output_path = provider_dir / pair.replace("/", "_")

    save_dataframe(
        df=df,
        path=output_path,
    )

    logger.success(f"Pipeline completed for {provider_name}/{pair}")

    return df
