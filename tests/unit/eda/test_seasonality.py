"""
Tests for seasonality.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import pandas.testing as pdt
import pytest
from matplotlib.figure import Figure

from fx_forecast.eda.seasonality import (
    generate_seasonality_analysis,
    monthly_returns,
    plot_monthly_returns,
    plot_weekday_returns,
    weekday_returns,
)


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Sample market dataset."""

    dates = pd.date_range(
        "2024-01-01",
        periods=10,
        freq="D",
    )

    return pd.DataFrame(
        {
            "Close": [
                100,
                101,
                102,
                101,
                103,
                104,
                105,
                104,
                106,
                107,
            ]
        },
        index=dates,
    )


def test_monthly_returns_returns_dataframe(
    sample_dataframe: pd.DataFrame,
) -> None:
    result = monthly_returns(sample_dataframe)

    assert isinstance(result, pd.DataFrame)


def test_monthly_returns_contains_expected_column(
    sample_dataframe: pd.DataFrame,
) -> None:
    result = monthly_returns(sample_dataframe)

    assert "average_return" in result.columns


def test_monthly_returns_missing_close_column() -> None:
    with pytest.raises(ValueError):
        monthly_returns(pd.DataFrame({"Open": [1, 2]}))


def test_monthly_returns_non_datetime_index() -> None:
    df = pd.DataFrame({"Close": [1, 2, 3]})

    with pytest.raises(ValueError):
        monthly_returns(df)


def test_weekday_returns_returns_dataframe(
    sample_dataframe: pd.DataFrame,
) -> None:
    result = weekday_returns(sample_dataframe)

    assert isinstance(result, pd.DataFrame)


def test_weekday_returns_contains_expected_column(
    sample_dataframe: pd.DataFrame,
) -> None:
    result = weekday_returns(sample_dataframe)

    assert "average_return" in result.columns


def test_weekday_returns_missing_close_column() -> None:
    with pytest.raises(ValueError):
        weekday_returns(pd.DataFrame({"Open": [1, 2]}))


def test_weekday_returns_non_datetime_index() -> None:
    df = pd.DataFrame({"Close": [1, 2, 3]})

    with pytest.raises(ValueError):
        weekday_returns(df)


def test_plot_monthly_returns_returns_figure(
    sample_dataframe: pd.DataFrame,
) -> None:
    fig = plot_monthly_returns(sample_dataframe)

    assert isinstance(fig, Figure)

    plt.close(fig)


def test_plot_weekday_returns_returns_figure(
    sample_dataframe: pd.DataFrame,
) -> None:
    fig = plot_weekday_returns(sample_dataframe)

    assert isinstance(fig, Figure)

    plt.close(fig)


def test_generate_seasonality_analysis_returns_dictionary(
    sample_dataframe: pd.DataFrame,
) -> None:
    result = generate_seasonality_analysis(sample_dataframe)

    assert isinstance(result, dict)

    plt.close(result["monthly_plot"])
    plt.close(result["weekday_plot"])


def test_generate_seasonality_analysis_expected_keys(
    sample_dataframe: pd.DataFrame,
) -> None:
    result = generate_seasonality_analysis(sample_dataframe)

    assert set(result.keys()) == {
        "monthly_returns",
        "weekday_returns",
        "monthly_plot",
        "weekday_plot",
    }

    plt.close(result["monthly_plot"])
    plt.close(result["weekday_plot"])


def test_generate_seasonality_analysis_output_types(
    sample_dataframe: pd.DataFrame,
) -> None:
    result = generate_seasonality_analysis(sample_dataframe)

    assert isinstance(result["monthly_returns"], pd.DataFrame)
    assert isinstance(result["weekday_returns"], pd.DataFrame)
    assert isinstance(result["monthly_plot"], Figure)
    assert isinstance(result["weekday_plot"], Figure)

    plt.close(result["monthly_plot"])
    plt.close(result["weekday_plot"])


def test_original_dataframe_not_modified(
    sample_dataframe: pd.DataFrame,
) -> None:
    original = sample_dataframe.copy(deep=True)

    result = generate_seasonality_analysis(sample_dataframe)

    pdt.assert_frame_equal(
        sample_dataframe,
        original,
    )

    plt.close(result["monthly_plot"])
    plt.close(result["weekday_plot"])
