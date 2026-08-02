"""
Verify the end-to-end data pipeline.

Run:
    uv run python -m scripts.verify.verify_pipeline
"""

from __future__ import annotations

from fx_forecast.data.pipeline import run_pipeline


def main() -> None:
    df = run_pipeline(
        symbol="EURUSD=X",
        start="2024-01-01",
    )

    print(df.tail())
    print()
    print(f"Rows: {len(df)}")
    print()
    print("✓ Pipeline verification passed")


if __name__ == "__main__":
    main()
