"""
Tests for distributions.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pandas.testing as pdt
import pytest
from matplotlib.figure import Figure

from fx_forecast.eda.distributions import (
    generate_distributions,
    plot_boxplots,
    plot_feature_distributions,
)


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Sample numeric dataset."""

    return pd.DataFrame(
        {
            "Open": [1, 2, 3, 4, 5],
            "High": [2, 3, 4, 5, 6],
            "Low": [0, 1, 2, 3, 4],
            "Close": [1.2, 2.1, 3.4, 4.6, 5.8],
            "Volume": [100, 200, 300, 400, 500],
        }
    )


def test_plot_feature_distributions_returns_dictionary(
    sample_dataframe: pd.DataFrame,
) -> None:
    result = plot_feature_distributions(sample_dataframe)

    assert isinstance(result, dict)

    for fig in result.values():
        plt.close(fig)


def test_plot_feature_distributions_returns_figures(
    sample_dataframe: pd.DataFrame,
) -> None:
    result = plot_feature_distributions(sample_dataframe)

    assert all(isinstance(fig, Figure) for fig in result.values())

    for fig in result.values():
        plt.close(fig)


def test_plot_feature_distributions_saves_png_files(
    sample_dataframe: pd.DataFrame,
    tmp_path: Path,
) -> None:
    result = plot_feature_distributions(
        sample_dataframe,
        output_dir=tmp_path,
    )

    assert len(list(tmp_path.glob("*_distribution.png"))) == len(result)

    for fig in result.values():
        plt.close(fig)


def test_plot_feature_distributions_no_numeric_columns() -> None:
    df = pd.DataFrame({"A": ["x", "y", "z"]})

    with pytest.raises(ValueError):
        plot_feature_distributions(df)


def test_plot_boxplots_returns_dictionary(
    sample_dataframe: pd.DataFrame,
) -> None:
    result = plot_boxplots(sample_dataframe)

    assert isinstance(result, dict)

    for fig in result.values():
        plt.close(fig)


def test_plot_boxplots_returns_figures(
    sample_dataframe: pd.DataFrame,
) -> None:
    result = plot_boxplots(sample_dataframe)

    assert all(isinstance(fig, Figure) for fig in result.values())

    for fig in result.values():
        plt.close(fig)


def test_plot_boxplots_saves_png_files(
    sample_dataframe: pd.DataFrame,
    tmp_path: Path,
) -> None:
    result = plot_boxplots(
        sample_dataframe,
        output_dir=tmp_path,
    )

    assert len(list(tmp_path.glob("*_boxplot.png"))) == len(result)

    for fig in result.values():
        plt.close(fig)


def test_plot_boxplots_no_numeric_columns() -> None:
    df = pd.DataFrame({"A": ["x", "y"]})

    with pytest.raises(ValueError):
        plot_boxplots(df)


def test_generate_distributions_returns_dictionary(
    sample_dataframe: pd.DataFrame,
) -> None:
    result = generate_distributions(sample_dataframe)

    assert isinstance(result, dict)

    for fig in result["histograms"].values():
        plt.close(fig)

    for fig in result["boxplots"].values():
        plt.close(fig)


def test_generate_distributions_expected_keys(
    sample_dataframe: pd.DataFrame,
) -> None:
    result = generate_distributions(sample_dataframe)

    assert set(result.keys()) == {
        "histograms",
        "boxplots",
    }

    for fig in result["histograms"].values():
        plt.close(fig)

    for fig in result["boxplots"].values():
        plt.close(fig)


def test_generate_distributions_output_types(
    sample_dataframe: pd.DataFrame,
) -> None:
    result = generate_distributions(sample_dataframe)

    assert isinstance(result["histograms"], dict)
    assert isinstance(result["boxplots"], dict)

    for fig in result["histograms"].values():
        plt.close(fig)

    for fig in result["boxplots"].values():
        plt.close(fig)


def test_original_dataframe_not_modified(
    sample_dataframe: pd.DataFrame,
) -> None:
    original = sample_dataframe.copy(deep=True)

    result = generate_distributions(sample_dataframe)

    pdt.assert_frame_equal(
        sample_dataframe,
        original,
    )

    for fig in result["histograms"].values():
        plt.close(fig)

    for fig in result["boxplots"].values():
        plt.close(fig)
