"""
Correlation analysis utilities.
"""

from __future__ import annotations

from typing import Any, Literal

import matplotlib

matplotlib.use("Agg")

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fx_forecast.utils.logger import logger

DEFAULT_THRESHOLD = 0.90

EXCLUDED_COLUMNS = [
    "target",
    "target_3d",
    "target_5d",
    "future_return_1d",
    "future_return_3d",
    "future_return_5d",
]


def correlation_matrix(
    df: pd.DataFrame,
    *,
    method: Literal["pearson", "spearman", "kendall"] = "pearson",
    drop_targets: bool = True,
) -> pd.DataFrame:
    """Compute correlation matrix."""

    numeric = df.select_dtypes(include="number").copy()

    if drop_targets:
        numeric = numeric.drop(
            columns=[c for c in EXCLUDED_COLUMNS if c in numeric.columns],
            errors="ignore",
        )

    if numeric.empty:
        raise ValueError("No numeric columns found.")

    corr = numeric.corr(method=method)

    logger.success("Correlation matrix generated.")

    return corr


def highly_correlated_features(
    corr: pd.DataFrame,
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> pd.DataFrame:
    """
    Return highly correlated feature pairs.
    """

    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    stacked = upper.stack()

    result = stacked.to_frame()
    result.columns = ["correlation"]
    result = result.reset_index()

    result.columns = [
        "feature_1",
        "feature_2",
        "correlation",
    ]

    result["correlation"] = result["correlation"].astype(float)

    result["abs_corr"] = result["correlation"].abs()

    result = result.loc[result["abs_corr"] >= threshold]

    result = result.sort_values(
        by=["abs_corr"],
        ascending=[False],
    )

    result = result.drop(columns=["abs_corr"])

    result = result.reset_index(drop=True)

    logger.success(f"Found {len(result)} highly correlated pairs.")

    return result


def plot_correlation_heatmap(
    corr: pd.DataFrame,
) -> matplotlib.figure.Figure:
    """Plot correlation heatmap."""

    fig, ax = plt.subplots(figsize=(10, 8))

    image = ax.imshow(
        corr,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
    )

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.columns)

    ax.set_title("Correlation Matrix")

    fig.colorbar(image, ax=ax)

    fig.tight_layout()

    logger.success("Correlation heatmap generated.")

    plt.close(fig)

    return fig


def generate_correlation_analysis(
    df: pd.DataFrame,
    *,
    method: Literal["pearson", "spearman", "kendall"] = "pearson",
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, Any]:
    """Generate complete correlation analysis."""

    logger.info("Generating correlation analysis.")

    corr = correlation_matrix(
        df,
        method=method,
    )

    return {
        "matrix": corr,
        "high_correlations": highly_correlated_features(
            corr,
            threshold=threshold,
        ),
        "heatmap": plot_correlation_heatmap(corr),
    }
