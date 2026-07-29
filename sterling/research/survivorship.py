"""Survivorship-free dataset from Sharadar bulk files.

The query API only surfaces currently-listed names. The BULK files
(data/sharadar_bulk/tickers_full.csv, stocks_10Y.csv) contain the ~38k delisted
companies too — so this module builds the universe the way it should be built:
every mid-cap-and-up common stock that traded in the window, INCLUDING the ones
that later died, with their price series ending at delisting.

Pipeline glue:
  load_tickers_meta()  -> metadata incl. isdelisted / lastpricedate / scalemarketcap
  select_universe()    -> mid-cap-and-up common stocks (delisted included)
  load_prices_bulk()   -> {ticker: OHLCV df}          (drop-in for features.build_features)
  delisting_map()      -> {ticker: (last_date, terminal_return)}
  add_labels()         -> forward EXCESS return vs the equal-weight universe, with the
                          delisting-aware rule so a company that dies books its real loss.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from sterling.research import config, store
from sterling.research.labels import forward_return_with_delisting

logger = logging.getLogger(__name__)

_MID_UP = config.MID_UP


def load_tickers_meta() -> pd.DataFrame:
    """Full tickers metadata (incl. delisted), via the DuckDB/Parquet store."""
    return store.load_tickers()


def select_universe(meta: pd.DataFrame, min_scale=_MID_UP) -> set:
    """Mid-cap-and-up common stocks (delisted included) — the survivorship-free set."""
    keep = meta[meta["scalemarketcap"].isin(min_scale)]
    return set(keep["ticker"].dropna())


def delisting_map(meta: pd.DataFrame, terminal_return: float = 0.0) -> dict:
    """{ticker: (last_trade_date, terminal_return)} for names flagged delisted."""
    out = {}
    for r in meta.itertuples(index=False):
        if str(getattr(r, "isdelisted", "N")) == "Y" and pd.notna(getattr(r, "lastpricedate", None)):
            out[r.ticker] = (r.lastpricedate.date(), terminal_return)
    return out


def load_prices_bulk(universe: set) -> dict[str, pd.DataFrame]:
    """{ticker: OHLCV df} for the universe, via the DuckDB/Parquet store (fast, low-mem).
    Delisted names' series end naturally at their last trading day."""
    return store.load_prices(universe)


def add_labels(feat_df: pd.DataFrame, prices: dict[str, pd.DataFrame],
               delisting: dict, horizon: int = 63) -> pd.DataFrame:
    """Forward return (delisting-aware) and its cross-sectional excess vs the equal-weight
    universe. `label` = fwd_return − (mean fwd_return across the universe that date), so
    the model learns to pick names that beat the average survivor-or-not name.
    """
    if feat_df.empty:
        return feat_df.assign(fwd_return=[], label=[])

    # Precompute per-ticker close arrays + date lists + integer position lookups.
    closes, dates_list, pos_of = {}, {}, {}
    for tk, df in prices.items():
        closes[tk] = df["Close"].to_numpy()
        dl = list(df.index.date)
        dates_list[tk] = dl
        pos_of[tk] = {d: i for i, d in enumerate(dl)}

    fwd = []
    for r in feat_df.itertuples(index=False):
        tk, d = r.ticker, r.date
        if tk not in pos_of or d not in pos_of[tk]:
            fwd.append(np.nan)
            continue
        dl = delisting.get(tk)
        val = forward_return_with_delisting(
            closes[tk], dates_list[tk], pos_of[tk][d], horizon,
            delist_date=dl[0] if dl else None,
            delist_return=dl[1] if dl else None,
        )
        fwd.append(val if val is not None else np.nan)

    out = feat_df.copy()
    out["fwd_return"] = fwd
    out = out[out["fwd_return"].notna()].reset_index(drop=True)
    # Cross-sectional excess vs equal-weight universe on each date.
    date_mean = out.groupby("date")["fwd_return"].transform("mean")
    out["bench_fwd"] = date_mean
    out["label"] = out["fwd_return"] - date_mean
    logger.info(f"survivorship labels: {len(out)} rows, "
                f"{out['ticker'].nunique()} tickers, "
                f"{sum(t in delisting for t in out['ticker'].unique())} delisted names present")
    return out
