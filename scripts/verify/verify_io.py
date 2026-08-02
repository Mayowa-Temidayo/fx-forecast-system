from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from fx_forecast.data.io import (
    load_dataframe,
    save_dataframe,
)

df = pd.DataFrame(
    {
        "Open": [1.10, 1.20, 1.30],
        "Close": [1.15, 1.18, 1.32],
    }
)

path = Path("data/processed/test")

save_dataframe(df, path)

loaded = load_dataframe(path)

assert_frame_equal(df, loaded)

print("✓ IO verification passed")
