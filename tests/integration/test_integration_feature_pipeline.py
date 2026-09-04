"""Integration tests for the feature engineering pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fx_forecast.features.pipeline import run_feature_pipeline


def test_feature_pipeline_produces_model_ready_data() -> None:
    """Test the complete feature engineering pipeline."""
    index = pd.date_range("2025-01-01", periods=40, freq="D")

    df = pd.DataFrame(
        {
            "Open": np.linspace(100.0, 139.0, 40),
            "High": np.linspace(101.0, 140.0, 40),
            "Low": np.linspace(99.0, 138.0, 40),
            "Close": np.linspace(100.5, 139.5, 40),
            "Volume": np.linspace(1000.0, 1039.0, 40),
        },
        index=index,
    )

    X, y = run_feature_pipeline(df)

    assert not X.empty
    assert not y.empty
    assert len(X) == len(y)

    assert X.index.equals(y.index)

    assert "target" not in X.columns
    assert "future_return_1d" not in X.columns

    assert X.isna().sum().sum() == 0
    assert y.isna().sum() == 0
