"""
Verify statistical feature engineering.
"""

from __future__ import annotations

import pandas as pd

from fx_forecast.features.statistical import create_statistical_features


def main() -> None:
    df = pd.DataFrame(
        {
            "Open": [1.10, 1.12, 1.15, 1.18, 1.20, 1.22],
            "High": [1.12, 1.14, 1.17, 1.20, 1.22, 1.24],
            "Low": [1.08, 1.10, 1.13, 1.16, 1.18, 1.20],
            "Close": [1.11, 1.13, 1.16, 1.19, 1.21, 1.23],
            "Volume": [1000, 1100, 1200, 1300, 1400, 1500],
        },
        index=pd.date_range("2025-01-01", periods=6, freq="D"),
    )

    result = create_statistical_features(
        df,
        windows=(3,),
    )

    expected_columns = [
        "return",
        "log_return",
        "rolling_mean_3",
        "rolling_std_3",
        "rolling_min_3",
        "rolling_max_3",
    ]

    missing = [column for column in expected_columns if column not in result.columns]

    if missing:
        raise AssertionError(f"Missing statistical features: {missing}")

    print(result[expected_columns].tail())

    print("\n✓ Statistical feature verification passed")


if __name__ == "__main__":
    main()
