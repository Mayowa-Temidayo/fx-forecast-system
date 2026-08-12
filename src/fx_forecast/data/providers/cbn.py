"""CBN FX data provider."""

from __future__ import annotations

from io import StringIO
from typing import cast

import pandas as pd
import requests

from fx_forecast.data.providers.base import FXProvider
from fx_forecast.data.validate import DataSchema


class CBNProvider(FXProvider):
    """Fetch official FX rates from CBN."""

    URL = "https://www.cbn.gov.ng/data-page.html"

    @property
    def schema(self) -> DataSchema:
        """Return the CBN dataframe schema."""

        return DataSchema(
            required_columns=(
                "Date",
                "Pair",
                "Close",
            ),
            numeric_columns=("Close",),
        )

    def fetch(
        self,
        pair: str,
        start: str,
        end: str | None = None,
    ) -> pd.DataFrame:
        """Fetch CBN exchange-rate observations."""

        response = requests.get(self.URL, timeout=30)
        response.raise_for_status()

        tables = pd.read_html(StringIO(response.text))
        df = pd.DataFrame(tables[0])

        return self._normalize(df, pair, start, end)

    @staticmethod
    def _normalize(
        df: pd.DataFrame,
        pair: str,
        start: str,
        end: str | None,
    ) -> pd.DataFrame:
        """Normalize CBN table into project format."""

        currency = "US DOLLAR" if pair == "USD/NGN" else "EURO"

        df = cast(
            pd.DataFrame,
            df.loc[df["Currency"].eq(currency)].copy(),
        )
        df["Date"] = pd.to_datetime(df["Rate Date"])

        df = df.rename(columns={"Central Rate": "Close"})
        df["Pair"] = pair

        mask = df["Date"] >= pd.Timestamp(start)

        if end:
            mask &= df["Date"] <= pd.Timestamp(end)

        result = cast(
            pd.DataFrame,
            df.loc[mask, ["Date", "Pair", "Close"]].copy(),
        )
        result = result.sort_values("Date")
        return result
