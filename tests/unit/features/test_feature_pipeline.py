"""
Unit tests for the feature engineering pipeline.
"""

from __future__ import annotations

import pandas as pd
import pandas.testing as pdt

from fx_forecast.features.pipeline import run_feature_pipeline


def make_dataframe() -> pd.DataFrame:
    """Create sample market data."""

    index = pd.date_range("2025-01-01", periods=40, freq="D")

    return pd.DataFrame(
        {
            "Open": [1.10 + i * 0.01 for i in range(40)],
            "High": [1.11 + i * 0.01 for i in range(40)],
            "Low": [1.09 + i * 0.01 for i in range(40)],
            "Close": [1.10 + i * 0.01 for i in range(40)],
            "Volume": [1000 + i * 10 for i in range(40)],
        },
        index=index,
    )


def test_returns_tuple() -> None:
    X, y = run_feature_pipeline(make_dataframe())

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)


def test_same_number_of_rows() -> None:
    X, y = run_feature_pipeline(make_dataframe())

    assert len(X) == len(y)


def test_target_removed_from_features() -> None:
    X, _ = run_feature_pipeline(make_dataframe())

    assert "target" not in X.columns


def test_pipeline_creates_calendar_features() -> None:
    X, _ = run_feature_pipeline(make_dataframe())

    assert "day_of_week" in X.columns
    assert "month" in X.columns


def test_pipeline_creates_statistical_features() -> None:
    X, _ = run_feature_pipeline(make_dataframe())

    assert "rolling_mean_5" in X.columns
    assert "rolling_std_20" in X.columns


def test_pipeline_creates_technical_features() -> None:
    X, _ = run_feature_pipeline(make_dataframe())

    assert "sma_5" in X.columns
    assert "ema_20" in X.columns


def test_pipeline_drops_missing_rows() -> None:
    X, y = run_feature_pipeline(make_dataframe())

    assert not X.isna().to_numpy().any()
    assert not y.isna().to_numpy().any()


def test_output_indices_match() -> None:
    X, y = run_feature_pipeline(make_dataframe())

    pdt.assert_index_equal(X.index, y.index)


def test_original_dataframe_not_modified() -> None:
    df = make_dataframe()
    original = df.copy(deep=True)

    run_feature_pipeline(df)

    pdt.assert_frame_equal(df, original)


def test_pipeline_generates_features() -> None:
    X, _ = run_feature_pipeline(make_dataframe())

    assert X.shape[1] > 5


def test_pipeline_not_empty() -> None:
    X, y = run_feature_pipeline(make_dataframe())

    assert not X.empty
    assert not y.empty
