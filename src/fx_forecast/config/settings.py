"""
Application settings.
"""

from __future__ import annotations

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from fx_forecast.config.constants import (
    DEFAULT_INTERVAL,
    DEFAULT_PAIRS,
    DEFAULT_START_DATE,
    RANDOM_SEED,
)


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="FX_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    currency_pairs: list[str] = DEFAULT_PAIRS

    start_date: str = DEFAULT_START_DATE

    interval: str = DEFAULT_INTERVAL

    random_seed: int = RANDOM_SEED

    aboki_api_url: str | None = None

    aboki_api_key: SecretStr | None = None


settings = Settings()
