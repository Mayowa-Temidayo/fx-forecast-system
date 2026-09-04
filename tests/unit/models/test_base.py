"""
Unit tests for the base forecasting model interface.
"""

from __future__ import annotations

import pandas as pd
import pytest

from fx_forecast.models.base import BaseForecastModel


class IncompleteModel(BaseForecastModel):
    """Intentionally incomplete model for abstract-class testing."""

    pass


class DummyModel(BaseForecastModel):
    """Minimal concrete model for interface testing."""

    @property
    def name(self) -> str:
        """Return the model name."""
        return "dummy"

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> DummyModel:
        """Return the fitted model."""
        return self

    def predict(
        self,
        X: pd.DataFrame,
    ) -> pd.Series:
        """Return a constant prediction."""
        return pd.Series(0.0, index=X.index)


def test_base_model_is_abstract() -> None:
    """Base model should not be directly instantiable."""

    with pytest.raises(TypeError):
        IncompleteModel()


def test_concrete_model_can_be_instantiated() -> None:
    """Concrete implementations should be instantiable."""

    model = DummyModel()

    assert model.name == "dummy"


def test_fit_returns_model() -> None:
    """Fit should return the fitted model."""

    model = DummyModel()

    X = pd.DataFrame({"feature": [1.0, 2.0]})
    y = pd.Series([1.1, 1.2], name="target")

    result = model.fit(X, y)

    assert result is model


def test_predict_returns_series() -> None:
    """Predict should return a pandas Series."""

    model = DummyModel()

    X = pd.DataFrame(
        {"feature": [1.0, 2.0]},
        index=pd.date_range("2025-01-01", periods=2),
    )

    result = model.predict(X)

    assert isinstance(result, pd.Series)
    assert result.index.equals(X.index)
    assert len(result) == len(X)
