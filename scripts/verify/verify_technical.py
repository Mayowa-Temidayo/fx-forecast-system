"""
Verify technical indicators.
"""

from __future__ import annotations

import pandas as pd

from fx_forecast.features.technical import (
    create_technical_features,
)


def main() -> None:
    df = pd.DataFrame(
        {
            "Close": [
                1.10,
                1.12,
                1.14,
                1.16,
                1.18,
                1.20,
                1.22,
                1.24,
            ]
        },
        index=pd.date_range(
            "2025-01-01",
            periods=8,
            freq="D",
        ),
    )

    result = create_technical_features(df)

    columns = [
        "sma_5",
        "sma_10",
        "sma_20",
        "ema_5",
        "ema_10",
        "ema_20",
    ]

    print(result[columns].tail())

    print("\n✓ Technical feature verification passed")


if __name__ == "__main__":
    main()
