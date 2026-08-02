"""
Verification script for validate.py.

Run:
    uv run python -m scripts.verify.verify_validate
"""

from __future__ import annotations

import pandas as pd

from fx_forecast.data.validate import validate_dataframe


def main() -> None:
    df = pd.DataFrame(
        {
            "Open": [1.10, 1.20, 1.30],
            "High": [1.15, 1.25, 1.35],
            "Low": [1.05, 1.15, 1.25],
            "Close": [1.12, 1.22, 1.32],
            "Volume": [1000, 1100, 1200],
        },
        index=pd.date_range("2025-01-01", periods=3, freq="D"),
    )

    validate_dataframe(df)

    print("✓ Validation verification passed")


if __name__ == "__main__":
    main()
