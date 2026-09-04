"""Unit tests for boosting models."""

from __future__ import annotations

import pandas as pd
import pytest

from fx_forecast.models.boosting import LightGBMModel, XGBoostModel


@pytest.fixture
def training_data() -> tuple[pd.DataFrame, pd.Series]:
    """Return sample training features and target."""
    X = pd.DataFrame(
        {
            "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "feature_2": [2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        }
    )

    y = pd.Series(
        [1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
        name="target",
    )

    return X, y


@pytest.mark.parametrize("model_class", [XGBoostModel, LightGBMModel])
def test_model_initializes(model_class: type) -> None:
    """Boosting models should initialize successfully."""
    model = model_class()

    assert model is not None


@pytest.mark.parametrize(
    ("model_class", "expected_name"),
    [
        (XGBoostModel, "xgboost"),
        (LightGBMModel, "lightgbm"),
    ],
)
def test_model_name(model_class: type, expected_name: str) -> None:
    """Boosting models should expose their expected names."""
    model = model_class()

    assert model.name == expected_name


@pytest.mark.parametrize("model_class", [XGBoostModel, LightGBMModel])
def test_fit_returns_fitted_model(
    model_class: type,
    training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Fitting should return the model instance."""
    X, y = training_data

    model = model_class()
    result = model.fit(X, y)

    assert result is model


@pytest.mark.parametrize("model_class", [XGBoostModel, LightGBMModel])
def test_predict_returns_series(
    model_class: type,
    training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Prediction should return a pandas Series."""
    X, y = training_data

    model = model_class()
    model.fit(X, y)

    predictions = model.predict(X)

    assert isinstance(predictions, pd.Series)
    assert len(predictions) == len(X)
    assert predictions.index.equals(X.index)


@pytest.mark.parametrize("model_class", [XGBoostModel, LightGBMModel])
def test_predict_before_fit_raises(
    model_class: type,
    training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Prediction before fitting should raise an error."""
    X, _ = training_data

    model = model_class()

    with pytest.raises(RuntimeError, match="fitted"):
        model.predict(X)
