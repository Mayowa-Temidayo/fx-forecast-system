"""Tests for the AbokiFX data provider."""

from unittest.mock import Mock, patch

import pandas as pd
import pytest

from fx_forecast.data.providers.aboki import AbokiProvider


@pytest.fixture
def provider() -> AbokiProvider:
    """Create a test AbokiFX provider."""
    return AbokiProvider(
        api_url="https://example.com/api/fx",
        api_key="test-key",
    )


@pytest.fixture
def aboki_payload() -> dict[str, list[dict[str, object]]]:
    """Return representative AbokiFX API data."""
    return {
        "data": [
            {
                "date": "2025-01-03",
                "buy": 1540.0,
                "sell": 1550.0,
            },
            {
                "date": "2025-01-02",
                "buy": 1530.0,
                "sell": 1540.0,
            },
        ]
    }


def test_fetch_usd_ngn(
    provider: AbokiProvider,
    aboki_payload: dict[str, list[dict[str, object]]],
) -> None:
    """USD/NGN should return normalized observations."""

    response = Mock()
    response.json.return_value = aboki_payload

    with patch(
        "fx_forecast.data.providers.aboki.requests.get",
        return_value=response,
    ) as mock_get:
        result = provider.fetch(
            pair="USD/NGN",
            start="2025-01-01",
            end="2025-01-03",
        )

    mock_get.assert_called_once()

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2
    assert list(result.columns) == [
        "Date",
        "Pair",
        "Buy",
        "Sell",
        "Close",
    ]
    assert bool(result["Pair"].eq("USD/NGN").all())


def test_fetch_eur_ngn(
    provider: AbokiProvider,
    aboki_payload: dict[str, list[dict[str, object]]],
) -> None:
    """EUR/NGN should be accepted."""

    response = Mock()
    response.json.return_value = aboki_payload

    with patch(
        "fx_forecast.data.providers.aboki.requests.get",
        return_value=response,
    ):
        result = provider.fetch(
            pair="EUR/NGN",
            start="2025-01-01",
            end="2025-01-03",
        )

    assert bool(result["Pair"].eq("EUR/NGN").all())


def test_fetch_sends_expected_request(
    provider: AbokiProvider,
    aboki_payload: dict[str, list[dict[str, object]]],
) -> None:
    """Fetch should send pair, date range, and authentication details."""

    response = Mock()
    response.json.return_value = aboki_payload

    with patch(
        "fx_forecast.data.providers.aboki.requests.get",
        return_value=response,
    ) as mock_get:
        provider.fetch(
            pair="USD/NGN",
            start="2025-01-01",
            end="2025-01-03",
        )

    mock_get.assert_called_once_with(
        "https://example.com/api/fx",
        params={
            "pair": "USD/NGN",
            "start": "2025-01-01",
            "end": "2025-01-03",
        },
        headers={
            "Authorization": "Bearer test-key",
            "Accept": "application/json",
        },
        timeout=30,
    )


def test_normalize_calculates_close(
    aboki_payload: dict[str, list[dict[str, object]]],
) -> None:
    """Close should be the midpoint of buy and sell."""

    result = AbokiProvider._normalize(
        aboki_payload,
        "USD/NGN",
    )

    assert result["Close"].tolist() == [1535.0, 1545.0]


def test_normalize_sorts_by_date(
    aboki_payload: dict[str, list[dict[str, object]]],
) -> None:
    """Normalized observations should be sorted chronologically."""

    result = AbokiProvider._normalize(
        aboki_payload,
        "USD/NGN",
    )

    assert result["Date"].tolist() == [
        pd.Timestamp("2025-01-02"),
        pd.Timestamp("2025-01-03"),
    ]


def test_unsupported_pair_raises(
    provider: AbokiProvider,
) -> None:
    """Unsupported currency pairs should be rejected."""

    with pytest.raises(ValueError, match="Unsupported AbokiFX pair"):
        provider.fetch(
            pair="GBP/NGN",
            start="2025-01-01",
        )


def test_invalid_payload_type_raises() -> None:
    """A non-object API response should raise TypeError."""

    with pytest.raises(
        TypeError,
        match="response must be a JSON object",
    ):
        AbokiProvider._normalize(
            payload=[],
            pair="USD/NGN",
        )


def test_missing_data_list_raises() -> None:
    """A response without a valid data list should raise ValueError."""

    with pytest.raises(
        ValueError,
        match="does not contain a valid data list",
    ):
        AbokiProvider._normalize(
            payload={},
            pair="USD/NGN",
        )


def test_missing_required_fields_raises() -> None:
    """Missing date, buy, or sell fields should raise ValueError."""

    payload = {
        "data": [
            {
                "date": "2025-01-03",
                "buy": 1540.0,
            }
        ]
    }

    with pytest.raises(
        ValueError,
        match="missing required fields",
    ):
        AbokiProvider._normalize(
            payload=payload,
            pair="USD/NGN",
        )


def test_invalid_numeric_value_raises() -> None:
    """Invalid buy/sell values should raise during normalization."""

    payload = {
        "data": [
            {
                "date": "2025-01-03",
                "buy": "invalid",
                "sell": 1550.0,
            }
        ]
    }

    with pytest.raises(ValueError):
        AbokiProvider._normalize(
            payload=payload,
            pair="USD/NGN",
        )


def test_http_error_is_propagated(
    provider: AbokiProvider,
) -> None:
    """HTTP errors should be propagated to the caller."""

    response = Mock()
    response.raise_for_status.side_effect = RuntimeError("HTTP failure")

    with patch(
        "fx_forecast.data.providers.aboki.requests.get",
        return_value=response,
    ):
        with pytest.raises(RuntimeError, match="HTTP failure"):
            provider.fetch(
                pair="USD/NGN",
                start="2025-01-01",
            )
