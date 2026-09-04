"""Unit tests for model training orchestration."""

from __future__ import annotations

import pandas as pd
import pytest

from fx_forecast.models.base import BaseForecastModel
from fx_forecast.training.trainer import ModelTrainer


class DummyModel(BaseForecastModel):
    """Simple forecasting model for trainer tests."""

    @property
    def name(self) -> str:
        """Return the model name."""
        return "dummy"

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> DummyModel:
        """Fit the dummy model."""
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Return constant predictions."""
        return pd.Series(
            1.5,
            index=X.index,
            name="prediction",
        )


@pytest.fixture
def training_data() -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
]:
    """Return sample training and validation data."""
    X_train = pd.DataFrame(
        {"feature": [1.0, 2.0, 3.0]},
        index=pd.date_range("2025-01-01", periods=3),
    )

    y_train = pd.Series(
        [1.1, 1.2, 1.3],
        index=X_train.index,
        name="target",
    )

    X_validation = pd.DataFrame(
        {"feature": [4.0, 5.0]},
        index=pd.date_range("2025-01-04", periods=2),
    )

    return X_train, y_train, X_validation


def test_trainer_initializes() -> None:
    """Trainer should initialize with a forecasting model."""
    model = DummyModel()

    trainer = ModelTrainer(model)

    assert trainer.model is model


def test_train_returns_result(
    training_data: tuple[
        pd.DataFrame,
        pd.Series,
        pd.DataFrame,
    ],
) -> None:
    """Training should return a training result."""
    X_train, y_train, X_validation = training_data
    model = DummyModel()

    result = ModelTrainer(model).train(
        X_train,
        y_train,
        X_validation,
    )

    assert result.model is model
    assert isinstance(result.validation_predictions, pd.Series)


def test_validation_predictions_preserve_index(
    training_data: tuple[
        pd.DataFrame,
        pd.Series,
        pd.DataFrame,
    ],
) -> None:
    """Validation predictions should preserve the validation index."""
    X_train, y_train, X_validation = training_data

    result = ModelTrainer(DummyModel()).train(
        X_train,
        y_train,
        X_validation,
    )

    assert result.validation_predictions.index.equals(
        X_validation.index,
    )


def test_validation_predictions_have_expected_values(
    training_data: tuple[
        pd.DataFrame,
        pd.Series,
        pd.DataFrame,
    ],
) -> None:
    """Validation predictions should come from the trained model."""
    X_train, y_train, X_validation = training_data

    result = ModelTrainer(DummyModel()).train(
        X_train,
        y_train,
        X_validation,
    )

    expected = pd.Series(
        1.5,
        index=X_validation.index,
        name="prediction",
    )

    pd.testing.assert_series_equal(
        result.validation_predictions,
        expected,
    )


def test_rejects_empty_training_data(
    training_data: tuple[
        pd.DataFrame,
        pd.Series,
        pd.DataFrame,
    ],
) -> None:
    """Trainer should reject empty training data."""
    _, y_train, X_validation = training_data

    X_train = pd.DataFrame()

    with pytest.raises(ValueError, match="Training data"):
        ModelTrainer(DummyModel()).train(
            X_train,
            y_train,
            X_validation,
        )


def test_rejects_mismatched_training_lengths(
    training_data: tuple[
        pd.DataFrame,
        pd.Series,
        pd.DataFrame,
    ],
) -> None:
    """Trainer should reject mismatched training lengths."""
    X_train, y_train, X_validation = training_data

    y_train = y_train.iloc[:-1]

    with pytest.raises(ValueError, match="same number of rows"):
        ModelTrainer(DummyModel()).train(
            X_train,
            y_train,
            X_validation,
        )


def test_rejects_empty_validation_data(
    training_data: tuple[
        pd.DataFrame,
        pd.Series,
        pd.DataFrame,
    ],
) -> None:
    """Trainer should reject empty validation data."""
    X_train, y_train, _ = training_data

    X_validation = pd.DataFrame()

    with pytest.raises(ValueError, match="Validation data"):
        ModelTrainer(DummyModel()).train(
            X_train,
            y_train,
            X_validation,
        )
