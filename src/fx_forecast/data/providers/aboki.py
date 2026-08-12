"""AbokiFX parallel-market data provider."""

from __future__ import annotations

from datetime import datetime
from typing import cast

import pandas as pd
import requests

from fx_forecast.data.providers.base import FXProvider
from fx_forecast.data.validate import DataSchema


class AbokiProvider(FXProvider):
    """Fetch parallel-market FX rates from AbokiFX."""

    def __init__(
        self,
        api_url: str,
        api_key: str,
    ) -> None:
        self.api_url = api_url
        self.api_key = api_key

    @property
    def schema(self) -> DataSchema:
        """Return the AbokiFX dataframe schema."""

        return DataSchema(
            required_columns=(
                "Date",
                "Pair",
                "Buy",
                "Sell",
                "Close",
            ),
            numeric_columns=(
                "Buy",
                "Sell",
                "Close",
            ),
        )

    def fetch(
        self,
        pair: str,
        start: str,
        end: str | None = None,
    ) -> pd.DataFrame:
        """Fetch AbokiFX parallel-market observations."""

        if pair not in {"USD/NGN", "EUR/NGN"}:
            raise ValueError(f"Unsupported AbokiFX pair: {pair}")

        response = requests.get(
            self.api_url,
            params={
                "pair": pair,
                "start": start,
                "end": end or datetime.today().strftime("%Y-%m-%d"),
            },
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
            },
            timeout=30,
        )

        response.raise_for_status()

        payload = response.json()

        return self._normalize(payload, pair)

    @staticmethod
    def _normalize(
        payload: object,
        pair: str,
    ) -> pd.DataFrame:
        """Normalize AbokiFX API observations."""

        if not isinstance(payload, dict):
            raise TypeError("AbokiFX response must be a JSON object.")

        records = payload.get("data")

        if not isinstance(records, list):
            raise ValueError("AbokiFX response does not contain a valid data list.")

        frame = pd.DataFrame(records)

        required = {"date", "buy", "sell"}

        missing = required - set(frame.columns)

        if missing:
            raise ValueError(
                f"AbokiFX response missing required fields: {sorted(missing)}"
            )

        frame["Date"] = pd.to_datetime(frame["date"], errors="raise")
        frame["Buy"] = pd.to_numeric(frame["buy"], errors="raise")
        frame["Sell"] = pd.to_numeric(frame["sell"], errors="raise")

        frame["Close"] = (frame["Buy"] + frame["Sell"]) / 2
        frame["Pair"] = pair

        result = cast(
            pd.DataFrame,
            frame.loc[:, ["Date", "Pair", "Buy", "Sell", "Close"]].copy(),
        )

        result = result.sort_values(by="Date").reset_index(drop=True)

        return result
