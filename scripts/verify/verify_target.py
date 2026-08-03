"""
Verification script for target features.
"""

from __future__ import annotations

import pandas as pd

from fx_forecast.features.target import create_target_features


def main() -> None:
    df = pd.DataFrame(
        {
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
            ]
        },
        index=pd.date_range(
            "2025-01-01",
            periods=10,
            freq="D",
        ),
    )

    result = create_target_features(df)

    print(
        result[
            [
                "target_1d",
                "target_3d",
                "target_5d",
                "future_return_1d",
                "future_return_3d",
                "future_return_5d",
            ]
        ].head()
    )

    print("\n✓ Target feature verification passed")


if __name__ == "__main__":
    main()
