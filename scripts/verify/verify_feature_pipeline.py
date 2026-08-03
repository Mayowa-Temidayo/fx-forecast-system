"""
Verification script for the feature engineering pipeline.
"""

from __future__ import annotations

import pandas as pd

from fx_forecast.features.pipeline import run_feature_pipeline


def main() -> None:
    """Run a quick end-to-end verification of the feature pipeline."""

    df = pd.DataFrame(
        {
            "Open": [
                1.10,
                1.11,
                1.12,
                1.13,
                1.14,
                1.15,
                1.16,
                1.17,
                1.18,
                1.19,
                1.20,
                1.21,
                1.22,
                1.23,
                1.24,
                1.25,
                1.26,
                1.27,
                1.28,
                1.29,
                1.30,
                1.31,
                1.32,
                1.33,
                1.34,
                1.35,
                1.36,
                1.37,
                1.38,
                1.39,
            ],
            "High": [
                1.11,
                1.12,
                1.13,
                1.14,
                1.15,
                1.16,
                1.17,
                1.18,
                1.19,
                1.20,
                1.21,
                1.22,
                1.23,
                1.24,
                1.25,
                1.26,
                1.27,
                1.28,
                1.29,
                1.30,
                1.31,
                1.32,
                1.33,
                1.34,
                1.35,
                1.36,
                1.37,
                1.38,
                1.39,
                1.40,
            ],
            "Low": [
                1.09,
                1.10,
                1.11,
                1.12,
                1.13,
                1.14,
                1.15,
                1.16,
                1.17,
                1.18,
                1.19,
                1.20,
                1.21,
                1.22,
                1.23,
                1.24,
                1.25,
                1.26,
                1.27,
                1.28,
                1.29,
                1.30,
                1.31,
                1.32,
                1.33,
                1.34,
                1.35,
                1.36,
                1.37,
                1.38,
            ],
            "Close": [
                1.10,
                1.11,
                1.12,
                1.13,
                1.14,
                1.15,
                1.16,
                1.17,
                1.18,
                1.19,
                1.20,
                1.21,
                1.22,
                1.23,
                1.24,
                1.25,
                1.26,
                1.27,
                1.28,
                1.29,
                1.30,
                1.31,
                1.32,
                1.33,
                1.34,
                1.35,
                1.36,
                1.37,
                1.38,
                1.39,
            ],
            "Volume": [
                1000,
            ]
            * 30,
        },
        index=pd.date_range(
            start="2025-01-01",
            periods=30,
            freq="D",
        ),
    )

    X, y = run_feature_pipeline(df)

    print("\nFeature matrix (X)")
    print("------------------")
    print(X.head())

    print("\nTarget (y)")
    print("----------")
    print(y.head())

    print("\nShape Summary")
    print("-------------")
    print(f"X shape : {X.shape}")
    print(f"y shape : {y.shape}")

    assert len(X) == len(y)
    assert X.shape[0] > 0
    assert X.shape[1] > 0

    print("\n✓ Feature pipeline verification passed")


if __name__ == "__main__":
    main()
