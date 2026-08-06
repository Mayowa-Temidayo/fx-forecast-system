"""
Tests for summary.py
"""

from __future__ import annotations

import pandas as pd
import pandas.testing as pdt
import pytest

from fx_forecast.eda.summary import (
    data_types,
    dataset_summary,
    generate_summary,
    memory_usage,
    missing_values,
)


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Sample dataset."""

    return pd.DataFrame(
        {
            "Open": [1.0, 2.0, 3.0],
            "Close": [1.2, 2.1, None],
            "Volume": [100, 200, 300],
            "Label": ["A", "B", "C"],
        }
    )


def test_dataset_summary_returns_dataframe(
    sample_dataframe: pd.DataFrame,
) -> None:
    result = dataset_summary(sample_dataframe)

    assert isinstance(result, pd.DataFrame)


def test_dataset_summary_expected_metrics(
    sample_dataframe: pd.DataFrame,
) -> None:
    result = dataset_summary(sample_dataframe)

    expected = {
        "Rows",
        "Columns",
        "Missing Values",
        "Duplicate Rows",
        "Memory (MB)",
    }

    assert expected.issubset(set(result["Metric"]))


def test_dataset_summary_empty_dataframe() -> None:
    with pytest.raises(ValueError):
        dataset_summary(pd.DataFrame())


def test_missing_values_returns_dataframe(
    sample_dataframe: pd.DataFrame,
) -> None:
    result = missing_values(sample_dataframe)

    assert isinstance(result, pd.DataFrame)


def test_missing_values_columns(
    sample_dataframe: pd.DataFrame,
) -> None:
    result = missing_values(sample_dataframe)

    assert list(result.columns) == [
        "missing",
        "percentage",
    ]


def test_missing_values_detects_missing(
    sample_dataframe: pd.DataFrame,
) -> None:
    result = missing_values(sample_dataframe)

    assert result.loc["Close", "missing"] == 1


def test_data_types_returns_dataframe(
    sample_dataframe: pd.DataFrame,
) -> None:
    result = data_types(sample_dataframe)

    assert isinstance(result, pd.DataFrame)


def test_data_types_contains_dtype_column(
    sample_dataframe: pd.DataFrame,
) -> None:
    result = data_types(sample_dataframe)

    assert "dtype" in result.columns


def test_memory_usage_returns_dataframe(
    sample_dataframe: pd.DataFrame,
) -> None:
    result = memory_usage(sample_dataframe)

    assert isinstance(result, pd.DataFrame)


def test_memory_usage_contains_column(
    sample_dataframe: pd.DataFrame,
) -> None:
    result = memory_usage(sample_dataframe)

    assert "memory_kb" in result.columns


def test_generate_summary_returns_dictionary(
    sample_dataframe: pd.DataFrame,
) -> None:
    result = generate_summary(sample_dataframe)

    assert isinstance(result, dict)


def test_generate_summary_expected_keys(
    sample_dataframe: pd.DataFrame,
) -> None:
    result = generate_summary(sample_dataframe)

    expected = {
        "dataset",
        "missing",
        "dtypes",
        "memory",
    }

    assert expected == set(result.keys())


def test_original_dataframe_not_modified(
    sample_dataframe: pd.DataFrame,
) -> None:
    original = sample_dataframe.copy(deep=True)

    generate_summary(sample_dataframe)

    pdt.assert_frame_equal(
        sample_dataframe,
        original,
    )
