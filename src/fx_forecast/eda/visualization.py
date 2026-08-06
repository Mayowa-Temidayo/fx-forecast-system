"""
Visualization utilities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.figure
import matplotlib.pyplot as plt
import pandas as pd

from fx_forecast.utils.logger import logger


def _save(
    fig: matplotlib.figure.Figure,
    output_dir: Path | None,
    filename: str,
) -> None:
    """Save figure if output directory is provided."""

    if output_dir is None:
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        output_dir / filename,
        dpi=300,
        bbox_inches="tight",
    )


def plot_close_price(
    df: pd.DataFrame,
    output_dir: Path | None = None,
) -> matplotlib.figure.Figure:
    """
    Plot Close price.
    """

    if "Close" not in df.columns:
        raise ValueError("'Close' column not found.")

    fig, ax = plt.subplots(figsize=(12, 5))

    df["Close"].plot(ax=ax)

    ax.set_title("Close Price")
    ax.set_xlabel("")
    ax.set_ylabel("Price")
    ax.grid(alpha=0.3)

    fig.tight_layout()

    _save(fig, output_dir, "close_price.png")

    plt.close(fig)

    logger.success("Close price plot generated.")

    return fig


def plot_returns(
    df: pd.DataFrame,
    output_dir: Path | None = None,
    column: str = "return",
) -> matplotlib.figure.Figure:
    """
    Plot returns.
    """

    if column not in df.columns:
        raise ValueError(f"'{column}' column not found.")

    fig, ax = plt.subplots(figsize=(12, 4))

    df[column].plot(ax=ax)

    ax.set_title("Returns")
    ax.set_xlabel("")
    ax.set_ylabel(column)
    ax.grid(alpha=0.3)

    fig.tight_layout()

    _save(fig, output_dir, "returns.png")

    plt.close(fig)

    logger.success("Returns plot generated.")

    return fig


def plot_histogram(
    df: pd.DataFrame,
    column: str,
    output_dir: Path | None = None,
    bins: int = 30,
) -> matplotlib.figure.Figure:
    """
    Plot histogram.
    """

    if column not in df.columns:
        raise ValueError(f"'{column}' column not found.")

    fig, ax = plt.subplots(figsize=(8, 5))

    df[column].plot.hist(
        bins=bins,
        ax=ax,
    )

    ax.set_title(f"{column} Distribution")

    fig.tight_layout()

    _save(fig, output_dir, f"{column}_histogram.png")

    plt.close(fig)

    logger.success("Histogram generated.")

    return fig


def plot_correlation_heatmap(
    df: pd.DataFrame,
    output_dir: Path | None = None,
) -> matplotlib.figure.Figure:
    """
    Plot correlation heatmap.
    """

    corr = df.select_dtypes(include="number").corr()

    if corr.empty:
        raise ValueError("No numeric columns found.")

    fig, ax = plt.subplots(figsize=(10, 8))

    image = ax.imshow(
        corr,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
    )

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))

    ax.set_xticklabels(
        corr.columns,
        rotation=90,
    )

    ax.set_yticklabels(corr.columns)

    fig.colorbar(image)

    fig.tight_layout()

    _save(fig, output_dir, "correlation_heatmap.png")

    plt.close(fig)

    logger.success("Correlation heatmap generated.")

    return fig


def generate_visualizations(
    df: pd.DataFrame,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Generate all visualizations.
    """

    logger.info("Generating visualizations.")

    return {
        "close_price": plot_close_price(df, output_dir),
        "returns": plot_returns(df, output_dir),
        "histogram": plot_histogram(df, "Close", output_dir),
        "heatmap": plot_correlation_heatmap(df, output_dir),
    }
