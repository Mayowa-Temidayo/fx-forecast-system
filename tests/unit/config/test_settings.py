"""Tests for application settings."""

from __future__ import annotations

from pydantic import SecretStr

from fx_forecast.config.constants import (
    DEFAULT_INTERVAL,
    DEFAULT_PAIRS,
    DEFAULT_START_DATE,
    RANDOM_SEED,
)
from fx_forecast.config.settings import Settings


def test_settings_use_project_defaults() -> None:
    """Settings should use project defaults when no environment values exist."""

    settings = Settings()

    assert settings.currency_pairs == DEFAULT_PAIRS
    assert settings.start_date == DEFAULT_START_DATE
    assert settings.interval == DEFAULT_INTERVAL
    assert settings.random_seed == RANDOM_SEED


def test_settings_aboki_configuration_is_optional() -> None:
    """Aboki configuration should be optional."""

    settings = Settings()

    assert settings.aboki_api_url is None
    assert settings.aboki_api_key is None


def test_settings_load_aboki_configuration(
    monkeypatch,
) -> None:
    """Aboki configuration should be loaded from environment variables."""

    monkeypatch.setenv(
        "FX_ABOKI_API_URL",
        "https://example.com/api",
    )
    monkeypatch.setenv(
        "FX_ABOKI_API_KEY",
        "test-api-key",
    )

    settings = Settings()

    assert settings.aboki_api_url == "https://example.com/api"
    assert isinstance(settings.aboki_api_key, SecretStr)
    assert settings.aboki_api_key.get_secret_value() == "test-api-key"


def test_settings_load_custom_interval(
    monkeypatch,
) -> None:
    """The download interval should be configurable."""

    monkeypatch.setenv("FX_INTERVAL", "1h")

    settings = Settings()

    assert settings.interval == "1h"


def test_settings_load_custom_start_date(
    monkeypatch,
) -> None:
    """The start date should be configurable."""

    monkeypatch.setenv(
        "FX_START_DATE",
        "2020-01-01",
    )

    settings = Settings()

    assert settings.start_date == "2020-01-01"
