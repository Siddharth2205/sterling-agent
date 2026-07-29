"""Liquidity / tradeability filter — keep only stocks you could actually trade.

The static `scalemarketcap` tag is last-known, so a company that was mid-cap and then
crashed 98% stays in the universe as an untradeable penny stock — and the model loved
exactly those (bounce artifacts). This filters each (ticker, date) row by its ACTUAL
state that day: a real share price and real dollar volume. Point-in-time, no look-ahead.
"""

from __future__ import annotations

import pandas as pd

from sterling.research import store


def liquidity_frame(prices: dict) -> pd.DataFrame:
    """Long frame [ticker, date, price, dvol] where dvol = 21-day avg dollar volume."""
    frames = []
    for tk, df in prices.items():
        dvol = (df["Close"] * df["Volume"]).rolling(21).mean()
        frames.append(pd.DataFrame({
            "ticker": tk,
            "date": [d for d in df.index.date],
            "price": df["Close"].to_numpy(),
            "dvol": dvol.to_numpy(),
        }))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def filter_tradeable(feat_df: pd.DataFrame, prices: dict | None = None,
                     min_price: float = 5.0, min_dvol: float = 1_000_000.0) -> pd.DataFrame:
    """Keep rows priced >= min_price with >= min_dvol average daily dollar volume."""
    prices = prices if prices is not None else store.load_prices(set(feat_df["ticker"].unique()))
    liq = liquidity_frame(prices)
    m = feat_df.merge(liq, on=["ticker", "date"], how="left")
    keep = m[(m["price"] >= min_price) & (m["dvol"] >= min_dvol)]
    return keep.reset_index(drop=True)
