"""
Unit tests for time-series dataset splitting.
"""

from __future__ import annotations

import pandas as pd
import pytest

from fx_forecast.data.split import (
    TimeSeriesSplit,
    split_time_series,
)


@pytest.fixture
def sample_data() -> tuple[pd.DataFrame, pd.Series]:
    """Return sample feature and target data."""

    index = pd.date_range(
        "2025-01-01",
        periods=20,
        freq="D",
    )

    X = pd.DataFrame(
        {
            "feature_1": range(20),
            "feature_2": range(20, 40),
        },
        index=index,
    )

    y = pd.Series(
        range(100, 120),
        index=index,
        name="target",
    )

    return X, y


def test_returns_timeseries_split(
    sample_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Split should return the expected result type."""

    X, y = sample_data

    result = split_time_series(X, y)

    assert isinstance(result, TimeSeriesSplit)


def test_split_sizes(
    sample_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Split should produce the expected dataset sizes."""

    X, y = sample_data

    result = split_time_series(X, y)

    assert len(result.X_train) == 14
    assert len(result.X_validation) == 3
    assert len(result.X_test) == 3

    assert len(result.y_train) == 14
    assert len(result.y_validation) == 3
    assert len(result.y_test) == 3


def test_split_preserves_chronological_order(
    sample_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Observations must remain in chronological order."""

    X, y = sample_data

    result = split_time_series(X, y)

    assert result.X_train.index[-1] < result.X_validation.index[0]
    assert result.X_validation.index[-1] < result.X_test.index[0]


def test_split_preserves_indexes(
    sample_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """X and y indexes should remain aligned."""

    X, y = sample_data

    result = split_time_series(X, y)

    assert result.X_train.index.equals(result.y_train.index)
    assert result.X_validation.index.equals(result.y_validation.index)
    assert result.X_test.index.equals(result.y_test.index)


def test_custom_split_sizes(
    sample_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Custom train and validation proportions should work."""

    X, y = sample_data

    result = split_time_series(
        X,
        y,
        train_size=0.60,
        validation_size=0.20,
    )

    assert len(result.X_train) == 12
    assert len(result.X_validation) == 4
    assert len(result.X_test) == 4


def test_rejects_empty_data() -> None:
    """Empty datasets should be rejected."""

    X = pd.DataFrame()
    y = pd.Series(dtype=float)

    with pytest.raises(ValueError, match="must not be empty"):
        split_time_series(X, y)


def test_rejects_mismatched_lengths(
    sample_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """X and y must contain the same number of rows."""

    X, y = sample_data

    with pytest.raises(ValueError, match="same number of rows"):
        split_time_series(X, y.iloc[:-1])


def test_rejects_mismatched_indexes(
    sample_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """X and y must have identical indexes."""

    X, y = sample_data

    y = y.copy()
    y.index = pd.date_range(
        "2026-01-01",
        periods=len(y),
        freq="D",
    )

    with pytest.raises(ValueError, match="identical indexes"):
        split_time_series(X, y)


@pytest.mark.parametrize(
    ("train_size", "validation_size"),
    [
        (0, 0.15),
        (1, 0.15),
        (0.70, 0),
        (0.70, 1),
        (0.80, 0.20),
        (0.90, 0.20),
    ],
)
def test_rejects_invalid_split_sizes(
    sample_data: tuple[pd.DataFrame, pd.Series],
    train_size: float,
    validation_size: float,
) -> None:
    """Invalid split proportions should be rejected."""

    X, y = sample_data

    with pytest.raises(ValueError):
        split_time_series(
            X,
            y,
            train_size=train_size,
            validation_size=validation_size,
        )
