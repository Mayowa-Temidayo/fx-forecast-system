"""
Verify preprocess.py.

Run:
    uv run python -m scripts.verify.verify_preprocess
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fx_forecast.data.preprocess import preprocess_dataframe


def main() -> None:
    df = pd.DataFrame(
        {
            " Open ": [1.10, np.nan, 1.30],
            "High": [1.15, 1.25, 1.35],
            "Low": [1.05, 1.15, 1.25],
            "Close": [1.12, 1.22, 1.32],
            "Volume": [1000, 1100, 1200],
        },
        index=pd.to_datetime(
            [
                "2025-01-03",
                "2025-01-01",
                "2025-01-02",
            ]
        ),
    )

    cleaned = preprocess_dataframe(df)

    print(cleaned)
    print()
    print("✓ Preprocess verification passed")


if __name__ == "__main__":
    main()
