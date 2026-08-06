"""
Distribution analysis utilities.
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


def plot_feature_distributions(
    df: pd.DataFrame,
    output_dir: Path | None = None,
    bins: int = 30,
) -> dict[str, matplotlib.figure.Figure]:
    """Plot histograms for all numeric columns."""

    numeric = df.select_dtypes(include="number")

    if numeric.empty:
        raise ValueError("No numeric columns found.")

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    figures: dict[str, matplotlib.figure.Figure] = {}

    for column in numeric.columns:
        fig, ax = plt.subplots(figsize=(8, 5))

        numeric[column].dropna().plot.hist(
            bins=bins,
            ax=ax,
        )

        ax.set_title(f"{column} Distribution")
        ax.set_xlabel(column)
        ax.set_ylabel("Frequency")

        fig.tight_layout()

        if output_dir is not None:
            fig.savefig(
                output_dir / f"{column}_distribution.png",
                dpi=300,
            )

        plt.close(fig)
        figures[column] = fig

    logger.success("Distribution plots generated.")

    return figures


def plot_boxplots(
    df: pd.DataFrame,
    output_dir: Path | None = None,
) -> dict[str, matplotlib.figure.Figure]:
    """Plot boxplots for all numeric columns."""

    numeric = df.select_dtypes(include="number")

    if numeric.empty:
        raise ValueError("No numeric columns found.")

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    figures: dict[str, matplotlib.figure.Figure] = {}

    for column in numeric.columns:
        fig, ax = plt.subplots(figsize=(8, 2))

        ax.boxplot(
            numeric[column].dropna(),
            orientation="horizontal",
        )

        ax.set_title(f"{column} Boxplot")

        fig.tight_layout()

        if output_dir is not None:
            fig.savefig(
                output_dir / f"{column}_boxplot.png",
                dpi=300,
            )

        plt.close(fig)
        figures[column] = fig

    logger.success("Boxplots generated.")

    return figures


def generate_distributions(
    df: pd.DataFrame,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Generate all distribution visualizations."""

    logger.info("Generating distribution analysis.")

    return {
        "histograms": plot_feature_distributions(
            df,
            output_dir=output_dir,
        ),
        "boxplots": plot_boxplots(
            df,
            output_dir=output_dir,
        ),
    }
