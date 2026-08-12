"""Unit tests for the CBN FX provider."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pandas as pd
import pytest

from fx_forecast.data.providers.cbn import CBNProvider


@pytest.fixture
def cbn_html() -> str:
    """Return representative CBN table HTML."""

    return """
    <table>
        <tr>
            <th>Currency</th>
            <th>Rate Date</th>
            <th>Central Rate</th>
        </tr>
        <tr>
            <td>US DOLLAR</td>
            <td>2025-01-02</td>
            <td>1550.25</td>
        </tr>
        <tr>
            <td>EURO</td>
            <td>2025-01-02</td>
            <td>1610.50</td>
        </tr>
        <tr>
            <td>US DOLLAR</td>
            <td>2025-01-03</td>
            <td>1555.75</td>
        </tr>
    </table>
    """


def test_fetch_usd_ngn(cbn_html: str) -> None:
    """USD/NGN should return CBN dollar observations."""

    response = Mock()
    response.text = cbn_html

    provider = CBNProvider()

    with patch(
        "fx_forecast.data.providers.cbn.requests.get",
        return_value=response,
    ):
        result = provider.fetch(
            pair="USD/NGN",
            start="2025-01-01",
            end="2025-01-03",
        )

    assert list(result.columns) == ["Date", "Pair", "Close"]
    assert len(result) == 2
    assert result["Pair"].unique().tolist() == ["USD/NGN"]
    assert result["Close"].tolist() == [1550.25, 1555.75]


def test_fetch_eur_ngn(cbn_html: str) -> None:
    """EUR/NGN should return CBN euro observations."""

    response = Mock()
    response.text = cbn_html

    provider = CBNProvider()

    with patch(
        "fx_forecast.data.providers.cbn.requests.get",
        return_value=response,
    ):
        result = provider.fetch(
            pair="EUR/NGN",
            start="2025-01-01",
            end="2025-01-03",
        )

    assert len(result) == 1
    assert result["Pair"].unique().tolist() == ["EUR/NGN"]
    assert result["Close"].iloc[0] == 1610.50


def test_fetch_filters_date_range(cbn_html: str) -> None:
    """CBN observations should respect the requested date range."""

    response = Mock()
    response.text = cbn_html

    provider = CBNProvider()

    with patch(
        "fx_forecast.data.providers.cbn.requests.get",
        return_value=response,
    ):
        result = provider.fetch(
            pair="USD/NGN",
            start="2025-01-03",
            end="2025-01-03",
        )

    assert len(result) == 1
    assert result["Close"].iloc[0] == 1555.75


def test_fetch_raises_for_http_error(cbn_html: str) -> None:
    """HTTP failures should propagate."""

    response = Mock()
    response.text = cbn_html
    response.raise_for_status.side_effect = RuntimeError("HTTP failure")

    provider = CBNProvider()

    with patch(
        "fx_forecast.data.providers.cbn.requests.get",
        return_value=response,
    ):
        with pytest.raises(RuntimeError, match="HTTP failure"):
            provider.fetch(
                pair="USD/NGN",
                start="2025-01-01",
            )


def test_normalize_returns_sorted_data() -> None:
    """Normalized CBN observations should be sorted chronologically."""

    df = pd.DataFrame(
        {
            "Currency": ["US DOLLAR", "US DOLLAR"],
            "Rate Date": ["2025-01-03", "2025-01-02"],
            "Central Rate": [1555.75, 1550.25],
        }
    )

    result = CBNProvider._normalize(
        df=df,
        pair="USD/NGN",
        start="2025-01-01",
        end=None,
    )

    assert result["Date"].is_monotonic_increasing
    assert result["Close"].tolist() == [1550.25, 1555.75]
