"""
Verification script for feature selection.
"""

from __future__ import annotations

import pandas as pd

from fx_forecast.features.selection import select_features


def main() -> None:
    df = pd.DataFrame(
        {
            "Open": [1, 2, 3],
            "High": [2, 3, 4],
            "Low": [0, 1, 2],
            "Close": [1.5, 2.5, 3.5],
            "Volume": [100, 120, 140],
            "return": [0.01, 0.02, 0.03],
            "sma_5": [1.4, 1.5, 1.6],
            "target": [1.55, 2.55, 3.55],
        }
    )

    features = select_features(df)

    print(features.head())
    print()
    print(f"Selected columns: {list(features.columns)}")
    print()

    assert "target" not in features.columns

    print("✓ Feature selection verification passed")


if __name__ == "__main__":
    main()
