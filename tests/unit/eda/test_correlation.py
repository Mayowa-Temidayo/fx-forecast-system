"""
Tests for correlation.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import pandas.testing as pdt
import pytest
from matplotlib.figure import Figure

from fx_forecast.eda.correlation import (
    DEFAULT_THRESHOLD,
    EXCLUDED_COLUMNS,
    correlation_matrix,
    generate_correlation_analysis,
    highly_correlated_features,
    plot_correlation_heatmap,
)


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Sample engineered dataset."""

    return pd.DataFrame(
        {
            "Open": [1, 2, 3, 4, 5],
            "High": [2, 3, 4, 5, 6],
            "Low": [0, 1, 2, 3, 4],
            "Close": [1.5, 2.5, 3.5, 4.5, 5.5],
            "Volume": [100, 200, 300, 400, 500],
            "return": [0.01, 0.02, -0.01, 0.03, 0.02],
            "target": [1, 0, 0, 1, 0],
        }
    )


def test_correlation_matrix_returns_dataframe(
    sample_dataframe: pd.DataFrame,
) -> None:
    result = correlation_matrix(sample_dataframe)

    assert isinstance(result, pd.DataFrame)


def test_correlation_matrix_square(
    sample_dataframe: pd.DataFrame,
) -> None:
    result = correlation_matrix(sample_dataframe)

    assert result.shape[0] == result.shape[1]


def test_target_columns_removed(
    sample_dataframe: pd.DataFrame,
) -> None:
    result = correlation_matrix(sample_dataframe)

    for column in EXCLUDED_COLUMNS:
        assert column not in result.columns


def test_no_numeric_columns() -> None:
    df = pd.DataFrame(
        {
            "A": ["x", "y"],
            "B": ["a", "b"],
        }
    )

    with pytest.raises(ValueError):
        correlation_matrix(df)


def test_highly_correlated_features_returns_dataframe(
    sample_dataframe: pd.DataFrame,
) -> None:
    corr = correlation_matrix(sample_dataframe)

    result = highly_correlated_features(corr)

    assert isinstance(result, pd.DataFrame)


def test_highly_correlated_features_columns(
    sample_dataframe: pd.DataFrame,
) -> None:
    corr = correlation_matrix(sample_dataframe)

    result = highly_correlated_features(corr)

    assert list(result.columns) == [
        "feature_1",
        "feature_2",
        "correlation",
    ]


def test_plot_correlation_heatmap_returns_figure(
    sample_dataframe: pd.DataFrame,
) -> None:
    corr = correlation_matrix(sample_dataframe)

    fig = plot_correlation_heatmap(corr)

    assert isinstance(fig, Figure)

    plt.close(fig)


def test_generate_correlation_analysis_returns_dictionary(
    sample_dataframe: pd.DataFrame,
) -> None:
    result = generate_correlation_analysis(sample_dataframe)

    assert isinstance(result, dict)

    plt.close(result["heatmap"])


def test_generate_correlation_analysis_expected_keys(
    sample_dataframe: pd.DataFrame,
) -> None:
    result = generate_correlation_analysis(sample_dataframe)

    assert set(result.keys()) == {
        "matrix",
        "high_correlations",
        "heatmap",
    }

    plt.close(result["heatmap"])


def test_generate_correlation_analysis_output_types(
    sample_dataframe: pd.DataFrame,
) -> None:
    result = generate_correlation_analysis(sample_dataframe)

    assert isinstance(result["matrix"], pd.DataFrame)
    assert isinstance(result["high_correlations"], pd.DataFrame)
    assert isinstance(result["heatmap"], Figure)

    plt.close(result["heatmap"])


def test_generate_correlation_analysis_custom_method(
    sample_dataframe: pd.DataFrame,
) -> None:
    result = generate_correlation_analysis(
        sample_dataframe,
        method="spearman",
    )

    assert isinstance(result["matrix"], pd.DataFrame)

    plt.close(result["heatmap"])


def test_generate_correlation_analysis_custom_threshold(
    sample_dataframe: pd.DataFrame,
) -> None:
    result = generate_correlation_analysis(
        sample_dataframe,
        threshold=0.50,
    )

    assert isinstance(result["high_correlations"], pd.DataFrame)

    plt.close(result["heatmap"])


def test_default_threshold_range() -> None:
    assert 0 < DEFAULT_THRESHOLD < 1


def test_original_dataframe_not_modified(
    sample_dataframe: pd.DataFrame,
) -> None:
    original = sample_dataframe.copy(deep=True)

    result = generate_correlation_analysis(sample_dataframe)

    pdt.assert_frame_equal(
        sample_dataframe,
        original,
    )

    plt.close(result["heatmap"])
