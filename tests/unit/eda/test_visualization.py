"""
Tests for visualization.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.figure
import pandas as pd
import pandas.testing as pdt
import pytest

from fx_forecast.eda.visualization import (
    generate_visualizations,
    plot_close_price,
    plot_correlation_heatmap,
    plot_histogram,
    plot_returns,
)


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Sample market dataset."""

    return pd.DataFrame(
        {
            "Close": [1.20, 1.35, 1.28, 1.42, 1.50],
            "return": [0.01, -0.02, 0.03, 0.01, -0.01],
            "Volume": [100, 150, 120, 180, 170],
        }
    )


def test_plot_close_price_returns_figure(
    sample_dataframe: pd.DataFrame,
) -> None:
    fig = plot_close_price(sample_dataframe)

    assert isinstance(fig, matplotlib.figure.Figure)


def test_plot_close_price_saves_file(
    sample_dataframe: pd.DataFrame,
    tmp_path: Path,
) -> None:
    plot_close_price(sample_dataframe, tmp_path)

    assert (tmp_path / "close_price.png").exists()


def test_plot_close_price_missing_close_column() -> None:
    with pytest.raises(ValueError):
        plot_close_price(pd.DataFrame({"Open": [1, 2]}))


def test_plot_returns_returns_figure(
    sample_dataframe: pd.DataFrame,
) -> None:
    fig = plot_returns(sample_dataframe)

    assert isinstance(fig, matplotlib.figure.Figure)


def test_plot_returns_saves_file(
    sample_dataframe: pd.DataFrame,
    tmp_path: Path,
) -> None:
    plot_returns(sample_dataframe, tmp_path)

    assert (tmp_path / "returns.png").exists()


def test_plot_returns_missing_column() -> None:
    with pytest.raises(ValueError):
        plot_returns(pd.DataFrame({"Close": [1, 2]}))


def test_plot_histogram_returns_figure(
    sample_dataframe: pd.DataFrame,
) -> None:
    fig = plot_histogram(sample_dataframe, "Close")

    assert isinstance(fig, matplotlib.figure.Figure)


def test_plot_histogram_saves_file(
    sample_dataframe: pd.DataFrame,
    tmp_path: Path,
) -> None:
    plot_histogram(
        sample_dataframe,
        "Close",
        tmp_path,
    )

    assert (tmp_path / "Close_histogram.png").exists()


def test_plot_histogram_missing_column() -> None:
    with pytest.raises(ValueError):
        plot_histogram(
            pd.DataFrame({"A": [1, 2]}),
            "Close",
        )


def test_plot_correlation_heatmap_returns_figure(
    sample_dataframe: pd.DataFrame,
) -> None:
    fig = plot_correlation_heatmap(sample_dataframe)

    assert isinstance(fig, matplotlib.figure.Figure)


def test_plot_correlation_heatmap_saves_file(
    sample_dataframe: pd.DataFrame,
    tmp_path: Path,
) -> None:
    plot_correlation_heatmap(
        sample_dataframe,
        tmp_path,
    )

    assert (tmp_path / "correlation_heatmap.png").exists()


def test_plot_correlation_heatmap_no_numeric_columns() -> None:
    df = pd.DataFrame({"A": ["x", "y"]})

    with pytest.raises(ValueError):
        plot_correlation_heatmap(df)


def test_generate_visualizations_returns_dictionary(
    sample_dataframe: pd.DataFrame,
) -> None:
    result = generate_visualizations(sample_dataframe)

    assert isinstance(result, dict)


def test_generate_visualizations_expected_keys(
    sample_dataframe: pd.DataFrame,
) -> None:
    result = generate_visualizations(sample_dataframe)

    assert set(result.keys()) == {
        "close_price",
        "returns",
        "histogram",
        "heatmap",
    }


def test_generate_visualizations_output_types(
    sample_dataframe: pd.DataFrame,
) -> None:
    result = generate_visualizations(sample_dataframe)

    assert all(isinstance(fig, matplotlib.figure.Figure) for fig in result.values())


def test_original_dataframe_not_modified(
    sample_dataframe: pd.DataFrame,
) -> None:
    original = sample_dataframe.copy(deep=True)

    generate_visualizations(sample_dataframe)

    pdt.assert_frame_equal(
        sample_dataframe,
        original,
    )
