"""
Verify calendar feature engineering.
"""

from __future__ import annotations

import pandas as pd

from fx_forecast.features.calendar import create_calendar_features


def main() -> None:
    df = pd.DataFrame(
        {
            "Open": [1.10, 1.20, 1.30],
            "High": [1.15, 1.25, 1.35],
            "Low": [1.05, 1.15, 1.25],
            "Close": [1.12, 1.22, 1.32],
            "Volume": [1000, 1100, 1200],
        },
        index=pd.to_datetime(
            [
                "2025-01-01",
                "2025-01-02",
                "2025-01-03",
            ]
        ),
    )

    result = create_calendar_features(df)

    expected_columns = [
        "day_of_week",
        "day_of_month",
        "day_of_year",
        "week_of_year",
        "month",
        "quarter",
        "year",
        "is_month_start",
        "is_month_end",
        "is_quarter_start",
        "is_quarter_end",
        "is_year_start",
        "is_year_end",
    ]

    missing = [column for column in expected_columns if column not in result.columns]

    if missing:
        raise AssertionError(f"Missing calendar features: {missing}")

    print(result[expected_columns])

    print("\n✓ Calendar feature verification passed")


if __name__ == "__main__":
    main()
