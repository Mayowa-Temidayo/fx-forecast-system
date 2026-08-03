"""
Unit tests for feature selection.
"""

from __future__ import annotations

import pandas as pd
import pandas.testing as pdt
import pytest

from fx_forecast.features.selection import select_features


@pytest.fixture
def sample_features() -> pd.DataFrame:
    """Sample engineered feature dataset."""

    return pd.DataFrame(
        {
            "Open": [1.0, 2.0, 3.0],
            "High": [2.0, 3.0, 4.0],
            "Low": [0.5, 1.5, 2.5],
            "Close": [1.5, 2.5, 3.5],
            "Volume": [100, 200, 300],
            "return": [0.01, 0.02, 0.03],
            "ema_5": [1.2, 2.2, 3.2],
            "target": [1.6, 2.6, 3.6],
        }
    )


def test_returns_tuple(
    sample_features: pd.DataFrame,
) -> None:
    """Should return (X, y)."""

    X, y = select_features(sample_features)

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)


def test_target_removed_from_features(
    sample_features: pd.DataFrame,
) -> None:
    """Target column should not appear in X."""

    X, _ = select_features(sample_features)

    assert "target" not in X.columns


def test_target_series_correct(
    sample_features: pd.DataFrame,
) -> None:
    """Returned y should equal the target column."""

    _, y = select_features(sample_features)

    pdt.assert_series_equal(
        y,
        sample_features["target"],
        check_names=True,
    )


def test_feature_count(
    sample_features: pd.DataFrame,
) -> None:
    """Feature count should equal total columns minus target."""

    X, _ = select_features(sample_features)

    assert X.shape[1] == sample_features.shape[1] - 1


def test_preserves_column_order(
    sample_features: pd.DataFrame,
) -> None:
    """Feature column order should be preserved."""

    X, _ = select_features(sample_features)

    expected = [column for column in sample_features.columns if column != "target"]

    assert list(X.columns) == expected


def test_original_dataframe_not_modified(
    sample_features: pd.DataFrame,
) -> None:
    """Input DataFrame should remain unchanged."""

    original = sample_features.copy(deep=True)

    select_features(sample_features)

    pdt.assert_frame_equal(sample_features, original)


def test_empty_dataframe() -> None:
    """Empty DataFrame should raise."""

    with pytest.raises(
        ValueError,
        match="Input DataFrame is empty",
    ):
        select_features(pd.DataFrame())


def test_missing_target_column() -> None:
    """Missing target column should raise."""

    df = pd.DataFrame(
        {
            "Open": [1, 2],
            "Close": [1, 2],
        }
    )

    with pytest.raises(
        ValueError,
        match="Missing target column",
    ):
        select_features(df)


def test_no_feature_columns() -> None:
    """If only target exists, raise."""

    df = pd.DataFrame(
        {
            "target": [1, 2, 3],
        }
    )

    with pytest.raises(
        ValueError,
        match="No feature columns available",
    ):
        select_features(df)
