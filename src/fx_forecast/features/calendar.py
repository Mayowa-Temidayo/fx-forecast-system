"""
Calendar-based feature engineering.
"""

from __future__ import annotations

import pandas as pd

from fx_forecast.utils.logger import logger


def create_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create calendar-derived features.
    """

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex.")

    result = df.copy()

    dates = pd.Series(df.index, index=df.index)

    iso = dates.dt.isocalendar()

    result["day_of_week"] = dates.dt.dayofweek
    result["day_of_month"] = dates.dt.day
    result["day_of_year"] = dates.dt.dayofyear

    result["week_of_year"] = iso.week.astype(int)

    result["month"] = dates.dt.month
    result["quarter"] = dates.dt.quarter
    result["year"] = dates.dt.year

    result["is_month_start"] = dates.dt.is_month_start.astype(int)
    result["is_month_end"] = dates.dt.is_month_end.astype(int)

    result["is_quarter_start"] = dates.dt.is_quarter_start.astype(int)
    result["is_quarter_end"] = dates.dt.is_quarter_end.astype(int)

    result["is_year_start"] = dates.dt.is_year_start.astype(int)
    result["is_year_end"] = dates.dt.is_year_end.astype(int)

    logger.success("Calendar features created.")

    return result
