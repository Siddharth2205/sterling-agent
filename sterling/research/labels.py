"""Forward-return labels for the research model.

Target = benchmark-relative forward return over `horizon` trading days:

    label = fwd_return(ticker, t, H) - fwd_return(benchmark, t, H)

Subtracting the market benchmark (SPY for US names, XIC.TO for Canadian) removes market
beta, so the model is asked to predict *stock-specific* outperformance — exactly what a
buy/hold/sell picker needs. Rows without a full horizon of future data are dropped
(label = NaN), which also prevents the most recent rows from leaking.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def forward_return_with_delisting(
    closes: "np.ndarray",
    dates: list,
    pos: int,
    horizon: int,
    delist_date=None,
    delist_return: Optional[float] = None,
) -> Optional[float]:
    """Forward return over `horizon` bars starting at index `pos`, correctly handling
    a stock that delists inside the window.

    This is the make-or-break rule for survivorship-free data. If a name delists during
    the forward window, we do NOT drop it (that would sneak survivorship bias back in) —
    we book the return through its last traded price, then apply the terminal delisting
    return: -1.0 (total loss) for a bankruptcy, or the deal return for a buyout. If no
    explicit delisting return is given for a name that simply stops trading, we assume a
    total loss (-100%), the conservative and usually-correct default for hard delistings.

    Args:
        closes: price array; dates: matching list of dates; pos: start index.
        delist_date: date the name delisted (or None if still listed).
        delist_return: terminal return on the delisting event (e.g. -1.0), or None.
    """
    p0 = closes[pos]
    if not (np.isfinite(p0) and p0 > 0):
        return None

    horizon_end_idx = pos + horizon
    delists_in_window = (
        delist_date is not None
        and (horizon_end_idx >= len(closes)                      # no full window of prices
             or dates[min(horizon_end_idx, len(dates) - 1)] >= delist_date)
    )

    if delists_in_window:
        # Return through the last valid price at/just before delisting, then the terminal event.
        last_idx = len(closes) - 1
        while last_idx > pos and not (np.isfinite(closes[last_idx]) and closes[last_idx] > 0):
            last_idx -= 1
        p_last = closes[last_idx]
        through = (p_last / p0) if (np.isfinite(p_last) and p_last > 0) else 1.0
        terminal = delist_return if delist_return is not None else -1.0  # bankruptcy default
        return float(through * (1.0 + terminal) - 1.0)

    if horizon_end_idx >= len(closes):
        return None
    p1 = closes[horizon_end_idx]
    if not (np.isfinite(p1) and p1 > 0):
        return None
    return float(p1 / p0 - 1.0)


def _fwd_map(df: pd.DataFrame, horizon: int) -> dict:
    """date -> forward `horizon`-bar return, for one price series."""
    closes = df["Close"].to_numpy()
    dates = list(df.index.date)
    out = {}
    for i in range(len(closes) - horizon):
        p0, p1 = closes[i], closes[i + horizon]
        if np.isfinite(p0) and np.isfinite(p1) and p0 > 0:
            out[dates[i]] = p1 / p0 - 1.0
    return out


def add_labels(
    feat_df: pd.DataFrame,
    prices: dict[str, pd.DataFrame],
    benchmarks: dict[str, pd.DataFrame],
    horizon: int = 63,
) -> pd.DataFrame:
    """Attach `fwd_return`, `bench_fwd`, and `label` (excess) columns; drop unlabelable rows."""
    if feat_df.empty:
        return feat_df.assign(fwd_return=[], bench_fwd=[], label=[])

    ticker_fwd = {tk: _fwd_map(df, horizon) for tk, df in prices.items()}
    bench_fwd = {mkt: _fwd_map(bdf, horizon) for mkt, bdf in benchmarks.items() if bdf is not None}

    fwd, bfwd = [], []
    for r in feat_df.itertuples(index=False):
        tf = ticker_fwd.get(r.ticker, {}).get(r.date, np.nan)
        bf = bench_fwd.get(r.market, {}).get(r.date, np.nan)
        fwd.append(tf)
        bfwd.append(bf)

    out = feat_df.copy()
    out["fwd_return"] = fwd
    out["bench_fwd"] = bfwd
    out["label"] = out["fwd_return"] - out["bench_fwd"]
    before = len(out)
    out = out[out["label"].notna()].reset_index(drop=True)
    logger.info(f"labelled {len(out)} rows (dropped {before - len(out)} without a full {horizon}-day forward window)")
    return out
