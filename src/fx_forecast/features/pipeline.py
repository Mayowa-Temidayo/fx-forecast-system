"""
Feature engineering pipeline.
"""

from __future__ import annotations

import pandas as pd

from fx_forecast.features.calendar import create_calendar_features
from fx_forecast.features.selection import select_features
from fx_forecast.features.statistical import create_statistical_features
from fx_forecast.features.target import create_target_features
from fx_forecast.features.technical import create_technical_features
from fx_forecast.utils.logger import logger


def run_feature_pipeline(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Execute the complete feature engineering pipeline.

    Pipeline
    --------
    1. Calendar features
    2. Statistical features
    3. Technical indicators
    4. Target engineering
    5. Remove incomplete rows
    6. Split into feature matrix (X) and target vector (y)

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed market data.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        Feature matrix (X) and target vector (y).
    """

    logger.info("Starting feature engineering pipeline.")

    data = df.copy()

    # Calendar features
    data = create_calendar_features(data)

    # Statistical features
    data = create_statistical_features(data)

    # Technical indicators
    data = create_technical_features(data)

    # Forecast targets
    data = create_target_features(data)

    # Remove rows made incomplete by rolling windows
    # and forward target shifts.
    data = data.dropna().copy()

    logger.info(f"Feature engineering complete ({len(data)} rows remaining).")

    X, y = select_features(data)

    logger.success(f"Pipeline finished ({X.shape[1]} features, {len(y)} samples).")

    return X, y
