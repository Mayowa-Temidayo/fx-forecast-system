"""
Seasonality analysis utilities.
"""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.figure
import matplotlib.pyplot as plt
import pandas as pd

from fx_forecast.utils.logger import logger


def monthly_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Average monthly returns.
    """

    if "Close" not in df.columns:
        raise ValueError("'Close' column not found.")

    data = df.copy()

    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex.")

    data["return"] = data["Close"].pct_change()
    data["month"] = data.index.strftime("%B")

    result = (
        data.groupby("month", sort=False)["return"].mean().to_frame("average_return")
    )

    logger.success("Monthly seasonality calculated.")

    return result


def weekday_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Average weekday returns.
    """

    if "Close" not in df.columns:
        raise ValueError("'Close' column not found.")

    data = df.copy()

    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex.")

    data["return"] = data["Close"].pct_change()
    data["weekday"] = data.index.strftime("%A")

    result = (
        data.groupby("weekday", sort=False)["return"].mean().to_frame("average_return")
    )

    logger.success("Weekday seasonality calculated.")

    return result


def plot_monthly_returns(
    df: pd.DataFrame,
) -> matplotlib.figure.Figure:
    """
    Plot monthly seasonality.
    """

    summary = monthly_returns(df)

    fig, ax = plt.subplots(figsize=(10, 5))

    summary.plot.bar(
        legend=False,
        ax=ax,
    )

    ax.set_title("Average Monthly Returns")
    ax.set_xlabel("")
    ax.set_ylabel("Average Return")
    ax.grid(alpha=0.3)

    fig.tight_layout()

    plt.close(fig)

    logger.success("Monthly seasonality plot generated.")

    return fig


def plot_weekday_returns(
    df: pd.DataFrame,
) -> matplotlib.figure.Figure:
    """
    Plot weekday seasonality.
    """

    summary = weekday_returns(df)

    fig, ax = plt.subplots(figsize=(10, 5))

    summary.plot.bar(
        legend=False,
        ax=ax,
    )

    ax.set_title("Average Weekday Returns")
    ax.set_xlabel("")
    ax.set_ylabel("Average Return")
    ax.grid(alpha=0.3)

    fig.tight_layout()

    plt.close(fig)

    logger.success("Weekday seasonality plot generated.")

    return fig


def generate_seasonality_analysis(
    df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Run complete seasonality analysis.
    """

    logger.info("Generating seasonality analysis.")

    return {
        "monthly_returns": monthly_returns(df),
        "weekday_returns": weekday_returns(df),
        "monthly_plot": plot_monthly_returns(df),
        "weekday_plot": plot_weekday_returns(df),
    }
