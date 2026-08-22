"""Tests for the Yahoo Finance FX provider."""

from unittest.mock import patch

import pandas as pd
import pytest

from fx_forecast.data.providers.yahoo import YahooFXProvider


@pytest.fixture
def yahoo_dataframe() -> pd.DataFrame:
    """Return representative Yahoo Finance data."""

    index = pd.date_range("2025-01-01", periods=2, freq="D")

    return pd.DataFrame(
        {
            "Open": [1500.0, 1510.0],
            "High": [1520.0, 1530.0],
            "Low": [1490.0, 1500.0],
            "Close": [1510.0, 1520.0],
            "Volume": [1000, 1100],
        },
        index=index,
    )


def test_fetch_usd_ngn(
    yahoo_dataframe: pd.DataFrame,
) -> None:
    """USD/NGN should map to the Yahoo USDNGN symbol."""

    with patch(
        "fx_forecast.data.providers.yahoo.yf.download",
        return_value=yahoo_dataframe,
    ) as mock_download:
        result = YahooFXProvider().fetch(
            pair="USD/NGN",
            start="2025-01-01",
            end="2025-01-03",
        )

    mock_download.assert_called_once_with(
        "USDNGN=X",
        start="2025-01-01",
        end="2025-01-03",
        interval="1d",
        progress=False,
        auto_adjust=True,
    )

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2


def test_fetch_eur_ngn(
    yahoo_dataframe: pd.DataFrame,
) -> None:
    """EUR/NGN should map to the Yahoo EURNGN symbol."""

    with patch(
        "fx_forecast.data.providers.yahoo.yf.download",
        return_value=yahoo_dataframe,
    ) as mock_download:
        YahooFXProvider().fetch(
            pair="EUR/NGN",
            start="2025-01-01",
            end="2025-01-03",
        )

    mock_download.assert_called_once_with(
        "EURNGN=X",
        start="2025-01-01",
        end="2025-01-03",
        interval="1d",
        progress=False,
        auto_adjust=True,
    )


def test_fetch_uses_custom_interval(
    yahoo_dataframe: pd.DataFrame,
) -> None:
    """Yahoo downloads should use the configured interval."""

    with patch(
        "fx_forecast.data.providers.yahoo.yf.download",
        return_value=yahoo_dataframe,
    ) as mock_download:
        YahooFXProvider(interval="1h").fetch(
            pair="USD/NGN",
            start="2025-01-01",
            end="2025-01-03",
        )

    mock_download.assert_called_once_with(
        "USDNGN=X",
        start="2025-01-01",
        end="2025-01-03",
        interval="1h",
        progress=False,
        auto_adjust=True,
    )


def test_fetch_preserves_unknown_symbol(
    yahoo_dataframe: pd.DataFrame,
) -> None:
    """Unknown pairs should be passed through as Yahoo symbols."""

    with patch(
        "fx_forecast.data.providers.yahoo.yf.download",
        return_value=yahoo_dataframe,
    ) as mock_download:
        YahooFXProvider().fetch(
            pair="GBPUSD=X",
            start="2025-01-01",
            end="2025-01-03",
        )

    mock_download.assert_called_once_with(
        "GBPUSD=X",
        start="2025-01-01",
        end="2025-01-03",
        interval="1d",
        progress=False,
        auto_adjust=True,
    )


def test_fetch_raises_when_no_data() -> None:
    """An empty Yahoo response should raise ValueError."""

    with patch(
        "fx_forecast.data.providers.yahoo.yf.download",
        return_value=pd.DataFrame(),
    ):
        with pytest.raises(
            ValueError,
            match="No data returned for USD/NGN",
        ):
            YahooFXProvider().fetch(
                pair="USD/NGN",
                start="2025-01-01",
                end="2025-01-03",
            )


def test_fetch_flattens_multiindex_columns() -> None:
    """MultiIndex Yahoo columns should be flattened."""

    index = pd.date_range("2025-01-01", periods=2, freq="D")

    columns = pd.MultiIndex.from_tuples(
        [
            ("Open", "USDNGN=X"),
            ("High", "USDNGN=X"),
            ("Low", "USDNGN=X"),
            ("Close", "USDNGN=X"),
            ("Volume", "USDNGN=X"),
        ]
    )

    dataframe = pd.DataFrame(
        [
            [1500.0, 1520.0, 1490.0, 1510.0, 1000],
            [1510.0, 1530.0, 1500.0, 1520.0, 1100],
        ],
        index=index,
        columns=columns,
    )

    with patch(
        "fx_forecast.data.providers.yahoo.yf.download",
        return_value=dataframe,
    ):
        result = YahooFXProvider().fetch(
            pair="USD/NGN",
            start="2025-01-01",
            end="2025-01-03",
        )

    assert isinstance(result.columns, pd.Index)
    assert list(result.columns) == [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]
    assert result.columns.name is None


def test_fetch_preserves_normal_columns(
    yahoo_dataframe: pd.DataFrame,
) -> None:
    """Normal Yahoo columns should remain unchanged."""

    with patch(
        "fx_forecast.data.providers.yahoo.yf.download",
        return_value=yahoo_dataframe,
    ):
        result = YahooFXProvider().fetch(
            pair="USD/NGN",
            start="2025-01-01",
            end="2025-01-03",
        )

    assert list(result.columns) == [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]
    assert result.columns.name is None


def test_fetch_uses_default_end_date(
    yahoo_dataframe: pd.DataFrame,
) -> None:
    """Fetch should supply today's date when end is omitted."""

    with patch(
        "fx_forecast.data.providers.yahoo.yf.download",
        return_value=yahoo_dataframe,
    ) as mock_download:
        YahooFXProvider().fetch(
            pair="USD/NGN",
            start="2025-01-01",
        )

    call = mock_download.call_args

    assert call is not None
    assert call.kwargs["start"] == "2025-01-01"
    assert call.kwargs["end"] == pd.Timestamp.today().strftime("%Y-%m-%d")
    assert call.kwargs["interval"] == "1d"
    assert call.kwargs["progress"] is False
    assert call.kwargs["auto_adjust"] is True
