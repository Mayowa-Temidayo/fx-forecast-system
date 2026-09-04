"""Unit tests for ensemble models."""

from __future__ import annotations

import pandas as pd
import pytest

from fx_forecast.models.base import BaseForecastModel
from fx_forecast.models.ensemble import WeightedEnsembleModel


class DummyModel(BaseForecastModel):
    """Simple model for ensemble testing."""

    def __init__(self, value: float, model_name: str) -> None:
        """Initialize the dummy model."""
        self.value = value
        self._name = model_name

    @property
    def name(self) -> str:
        """Return the model name."""
        return self._name

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> DummyModel:
        """Return the fitted model."""
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Return constant predictions."""
        return pd.Series(self.value, index=X.index, name="prediction")


@pytest.fixture
def training_data() -> tuple[pd.DataFrame, pd.Series]:
    """Return sample training data."""
    X = pd.DataFrame(
        {"feature": [1.0, 2.0, 3.0]},
        index=pd.date_range("2025-01-01", periods=3),
    )

    y = pd.Series(
        [1.0, 2.0, 3.0],
        index=X.index,
        name="target",
    )

    return X, y


def test_model_name() -> None:
    """Ensemble should expose its expected name."""
    models = {"model_a": DummyModel(1.0, "model_a")}

    ensemble = WeightedEnsembleModel(models)

    assert ensemble.name == "weighted_ensemble"


def test_equal_weights_are_created() -> None:
    """Ensemble should create equal weights by default."""
    models = {
        "model_a": DummyModel(1.0, "model_a"),
        "model_b": DummyModel(2.0, "model_b"),
    }

    ensemble = WeightedEnsembleModel(models)

    assert ensemble.weights == {
        "model_a": 0.5,
        "model_b": 0.5,
    }


def test_custom_weights_are_used() -> None:
    """Ensemble should preserve valid custom weights."""
    models = {
        "model_a": DummyModel(1.0, "model_a"),
        "model_b": DummyModel(2.0, "model_b"),
    }

    weights = {
        "model_a": 0.7,
        "model_b": 0.3,
    }

    ensemble = WeightedEnsembleModel(models, weights)

    assert ensemble.weights == weights


def test_rejects_empty_models() -> None:
    """Ensemble should reject an empty model collection."""
    with pytest.raises(ValueError, match="At least one model"):
        WeightedEnsembleModel({})


def test_rejects_mismatched_weights() -> None:
    """Ensemble should reject weights for different models."""
    models = {
        "model_a": DummyModel(1.0, "model_a"),
    }

    with pytest.raises(ValueError, match="match"):
        WeightedEnsembleModel(
            models,
            {"model_b": 1.0},
        )


def test_rejects_negative_weights() -> None:
    """Ensemble should reject negative weights."""
    models = {
        "model_a": DummyModel(1.0, "model_a"),
        "model_b": DummyModel(2.0, "model_b"),
    }

    with pytest.raises(ValueError, match="non-negative"):
        WeightedEnsembleModel(
            models,
            {"model_a": -0.1, "model_b": 1.1},
        )


def test_rejects_weights_not_summing_to_one() -> None:
    """Ensemble should require weights to sum to one."""
    models = {
        "model_a": DummyModel(1.0, "model_a"),
        "model_b": DummyModel(2.0, "model_b"),
    }

    with pytest.raises(ValueError, match="sum to 1.0"):
        WeightedEnsembleModel(
            models,
            {"model_a": 0.4, "model_b": 0.4},
        )


def test_fit_returns_model(
    training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Fitting should return the ensemble instance."""
    X, y = training_data

    models = {
        "model_a": DummyModel(1.0, "model_a"),
        "model_b": DummyModel(2.0, "model_b"),
    }

    ensemble = WeightedEnsembleModel(models)

    result = ensemble.fit(X, y)

    assert result is ensemble


def test_predict_returns_weighted_average(
    training_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Predictions should equal the weighted model predictions."""
    X, y = training_data

    models = {
        "model_a": DummyModel(1.0, "model_a"),
        "model_b": DummyModel(3.0, "model_b"),
    }

    ensemble = WeightedEnsembleModel(
        models,
        {"model_a": 0.25, "model_b": 0.75},
    )

    ensemble.fit(X, y)
    predictions = ensemble.predict(X)

    expected = pd.Series(
        2.5,
        index=X.index,
        name="prediction",
    )

    pd.testing.assert_series_equal(predictions, expected)
