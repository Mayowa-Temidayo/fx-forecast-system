"""FX provider registry."""

from fx_forecast.data.providers.aboki import AbokiProvider
from fx_forecast.data.providers.base import FXProvider
from fx_forecast.data.providers.cbn import CBNProvider
from fx_forecast.data.providers.yahoo import YahooFXProvider


def get_provider(
    name: str,
    aboki_api_url: str | None = None,
    aboki_api_key: str | None = None,
) -> FXProvider:
    """Create an FX provider by name."""

    if name == "yahoo":
        return YahooFXProvider()

    if name == "cbn":
        return CBNProvider()

    if name == "aboki":
        if not aboki_api_url or not aboki_api_key:
            raise ValueError("AbokiFX credentials are required.")

        return AbokiProvider(
            api_url=aboki_api_url,
            api_key=aboki_api_key,
        )

    raise ValueError(f"Unknown FX provider: {name}")
