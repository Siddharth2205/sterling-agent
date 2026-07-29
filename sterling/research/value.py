"""Value signal from point-in-time daily valuation ratios (P/E, P/B, P/S).

Value (cheap vs expensive) is the classic diversifier to a momentum/price model: the two
are famously negatively correlated yet both positive over the long run, so combining them
is the textbook way to lift risk-adjusted return.

Ratios come from Sharadar's `daily` table (price-based, recomputed each day from price +
last-filed fundamentals) so they are already point-in-time. We attach the *latest ratio
known on or before each scan date* with a DuckDB ASOF join — no look-ahead. The signal is
"cheapness": a blend of earnings-yield (1/PE), book-to-price (1/PB) and sales-to-price
(1/PS), higher = cheaper = more attractive.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import duckdb
import numpy as np
import pandas as pd

from sterling.research import config, store, signals


def build_value_parquet(csv: Optional[Path] = None, parquet: Optional[Path] = None,
                        force: bool = False) -> Path:
    csv = csv or config.BULK / "daily_value.csv"
    parquet = parquet or config.BULK / "daily_value.parquet"
    return store.csv_to_parquet(csv, parquet, force)


def attach_value(panel: pd.DataFrame, parquet: Optional[Path] = None) -> pd.DataFrame:
    """Attach as-of P/E, P/B, P/S to each (date, ticker) via a DuckDB ASOF join."""
    parquet = parquet or config.BULK / "daily_value.parquet"
    keys = panel[["date", "ticker"]].drop_duplicates()
    con = duckdb.connect()
    con.register("p", keys)
    res = con.execute(
        f"SELECT p.date, p.ticker, v.pe, v.pb, v.ps "
        f"FROM p ASOF LEFT JOIN read_parquet('{parquet.as_posix()}') v "
        f"ON p.ticker = v.ticker AND p.date >= v.date"
    ).df()
    con.close()
    # DuckDB returns date as datetime64; match the panel's plain-date dtype for the merge.
    res["date"] = pd.to_datetime(res["date"]).dt.date
    return panel.merge(res, on=["date", "ticker"], how="left")


def value_score(df: pd.DataFrame) -> pd.Series:
    """Cheapness score: per-date z-score blend of 1/PE, 1/PB, 1/PS (higher = cheaper).
    Non-positive ratios (unprofitable / negative book) are treated as missing."""
    tmp = df.assign(
        ey=np.where(df["pe"] > 0, 1.0 / df["pe"], np.nan),
        bp=np.where(df["pb"] > 0, 1.0 / df["pb"], np.nan),
        sp=np.where(df["ps"] > 0, 1.0 / df["ps"], np.nan),
    )
    return signals.combine(tmp, ["ey", "bp", "sp"])
