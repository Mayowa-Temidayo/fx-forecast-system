"""
Unit tests for the naive forecasting model.
"""

from __future__ import annotations

import pandas as pd
import pytest

from fx_forecast.models.naive import NaiveForecastModel


@pytest.fixture
def training_data() -> tuple[pd.DataFrame, pd.Series]:
    """Return sample training data."""

    index = pd.date_range(
        "2025-01-01",
        periods=5,
        freq="D",
    )

    X = pd.DataFrame(
        {
            "feature": [1.0, 2.0, 3.0, 4.0, 5.0],
        },
        index=index,
    )

    y = pd.Series(
        [100.0, 101.0, 102.0, 103.0, 104.0],
        index=index,
        name="target",
    )

    return X, y


def test_model_name() -> None:
    """Model should expose the expected name."""

    model = NaiveForecastModel()

    assert model.name == "naive"


def test_predict_requires_fit() -> None:
    """Prediction before fitting should fail."""

    model = NaiveForecastModel()

    X = pd.DataFrame({"feature": [1.0]})

    with pytest.raises(RuntimeError, match="fitted"):
        model.predict(X)


def test_fit_uses_last_target_value(
    training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Fit should use the final observed target."""

    X, y = training_data

    model = NaiveForecastModel()
    model.fit(X, y)

    X_future = pd.DataFrame(
        {"feature": [6.0, 7.0]},
        index=pd.date_range("2025-01-06", periods=2, freq="D"),
    )

    predictions = model.predict(X_future)

    assert predictions.tolist() == [104.0, 104.0]


def test_prediction_preserves_index(
    training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Predictions should preserve the input index."""

    X, y = training_data

    model = NaiveForecastModel().fit(X, y)

    X_future = pd.DataFrame(
        {"feature": [6.0, 7.0]},
        index=pd.date_range("2025-01-06", periods=2, freq="D"),
    )

    predictions = model.predict(X_future)

    assert predictions.index.equals(X_future.index)


def test_rejects_empty_training_data() -> None:
    """Empty training data should be rejected."""

    model = NaiveForecastModel()

    X = pd.DataFrame()
    y = pd.Series(dtype=float)

    with pytest.raises(ValueError, match="must not be empty"):
        model.fit(X, y)


def test_rejects_mismatched_training_lengths(
    training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Training features and targets must have equal lengths."""

    X, y = training_data

    model = NaiveForecastModel()

    with pytest.raises(ValueError, match="same number of rows"):
        model.fit(X, y.iloc[:-1])


def test_rejects_empty_prediction_data(
    training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Empty prediction data should be rejected."""

    X, y = training_data

    model = NaiveForecastModel().fit(X, y)

    with pytest.raises(ValueError, match="must not be empty"):
        model.predict(pd.DataFrame())
