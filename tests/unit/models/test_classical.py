"""
Unit tests for classical forecasting models.
"""

from __future__ import annotations

import pandas as pd
import pytest

from fx_forecast.models.classical import ARIMAForecastModel


@pytest.fixture
def training_data() -> tuple[pd.DataFrame, pd.Series]:
    """Return sample time-series training data."""

    index = pd.date_range(
        "2025-01-01",
        periods=20,
        freq="D",
    )

    X = pd.DataFrame(
        {
            "feature": range(20),
        },
        index=index,
    )

    y = pd.Series(
        [100 + i * 0.5 for i in range(20)],
        index=index,
        name="target",
    )

    return X, y


def test_model_name() -> None:
    """ARIMA should expose the expected name."""

    model = ARIMAForecastModel()

    assert model.name == "arima"


def test_default_order() -> None:
    """ARIMA should use the expected default order."""

    model = ARIMAForecastModel()

    assert model.order == (1, 1, 1)


def test_custom_order() -> None:
    """ARIMA should accept a custom order."""

    model = ARIMAForecastModel(order=(2, 1, 2))

    assert model.order == (2, 1, 2)


def test_predict_requires_fit() -> None:
    """Prediction before fitting should fail."""

    model = ARIMAForecastModel()

    X = pd.DataFrame({"feature": [1.0]})

    with pytest.raises(RuntimeError, match="fitted"):
        model.predict(X)


def test_fit_returns_model(
    training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Fit should return the fitted model."""

    X, y = training_data

    model = ARIMAForecastModel()

    result = model.fit(X, y)

    assert result is model


def test_predict_returns_series(
    training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Prediction should return a pandas Series."""

    X, y = training_data

    model = ARIMAForecastModel().fit(X, y)

    X_future = pd.DataFrame(
        {"feature": [20, 21, 22]},
        index=pd.date_range(
            "2025-01-21",
            periods=3,
            freq="D",
        ),
    )

    predictions = model.predict(X_future)

    assert isinstance(predictions, pd.Series)
    assert len(predictions) == len(X_future)
    assert predictions.index.equals(X_future.index)
    assert predictions.name == "prediction"


def test_rejects_empty_training_data() -> None:
    """Empty training data should be rejected."""

    model = ARIMAForecastModel()

    X = pd.DataFrame()
    y = pd.Series(dtype=float)

    with pytest.raises(ValueError, match="must not be empty"):
        model.fit(X, y)


def test_rejects_mismatched_training_lengths(
    training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Training features and targets must have equal lengths."""

    X, y = training_data

    model = ARIMAForecastModel()

    with pytest.raises(ValueError, match="same number of rows"):
        model.fit(X, y.iloc[:-1])


def test_rejects_empty_prediction_data(
    training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Empty prediction data should be rejected."""

    X, y = training_data

    model = ARIMAForecastModel().fit(X, y)

    with pytest.raises(ValueError, match="must not be empty"):
        model.predict(pd.DataFrame())
