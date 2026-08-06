"""
Tests for statistics.py
"""

from __future__ import annotations

import pandas as pd
import pandas.testing as pdt
import pytest

from fx_forecast.eda.statistics import (
    correlation,
    descriptive_statistics,
    generate_statistics,
    kurtosis,
    skewness,
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


def test_descriptive_statistics_returns_dataframe(
    sample_dataframe: pd.DataFrame,
) -> None:
    result = descriptive_statistics(sample_dataframe)

    assert isinstance(result, pd.DataFrame)


def test_descriptive_statistics_contains_numeric_columns(
    sample_dataframe: pd.DataFrame,
) -> None:
    result = descriptive_statistics(sample_dataframe)

    assert "Open" in result.index
    assert "Close" in result.index


def test_descriptive_statistics_no_numeric_columns() -> None:
    df = pd.DataFrame({"A": ["x", "y", "z"]})

    with pytest.raises(ValueError):
        descriptive_statistics(df)


def test_skewness_returns_series(
    sample_dataframe: pd.DataFrame,
) -> None:
    result = skewness(sample_dataframe)

    assert isinstance(result, pd.Series)


def test_kurtosis_returns_series(
    sample_dataframe: pd.DataFrame,
) -> None:
    result = kurtosis(sample_dataframe)

    assert isinstance(result, pd.Series)


def test_correlation_returns_dataframe(
    sample_dataframe: pd.DataFrame,
) -> None:
    result = correlation(sample_dataframe)

    assert isinstance(result, pd.DataFrame)


def test_correlation_is_square(
    sample_dataframe: pd.DataFrame,
) -> None:
    result = correlation(sample_dataframe)

    assert result.shape[0] == result.shape[1]


def test_generate_statistics_returns_dictionary(
    sample_dataframe: pd.DataFrame,
) -> None:
    result = generate_statistics(sample_dataframe)

    assert isinstance(result, dict)


def test_generate_statistics_contains_expected_sections(
    sample_dataframe: pd.DataFrame,
) -> None:
    result = generate_statistics(sample_dataframe)

    expected = {
        "describe",
        "skewness",
        "kurtosis",
        "correlation",
    }

    assert expected == set(result.keys())


def test_original_dataframe_not_modified(
    sample_dataframe: pd.DataFrame,
) -> None:
    original = sample_dataframe.copy(deep=True)

    generate_statistics(sample_dataframe)

    pdt.assert_frame_equal(
        sample_dataframe,
        original,
    )
