from pathlib import Path

import pandas as pd

path = Path("data/processed/test.parquet")

print("1. Exists:", path.exists())

print("2. Reading...")

df = pd.read_parquet(path, engine="pyarrow")

print("3. Success")

print(df)
