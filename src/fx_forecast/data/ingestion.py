"""FX data ingestion orchestration."""

from fx_forecast.data.pipeline import run_pipeline
from fx_forecast.data.providers.registry import get_provider


def ingest_all(
    providers: list[str],
    pairs: list[str],
    start: str,
    end: str | None = None,
) -> None:
    """Run the data pipeline for every provider and pair."""

    for provider_name in providers:
        provider = get_provider(provider_name)

        for pair in pairs:
            run_pipeline(
                provider=provider,
                pair=pair,
                start=start,
                end=end,
                provider_name=provider_name,
            )
