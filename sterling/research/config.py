"""Central configuration for the research pipeline — paths and default params in one
place instead of scattered constants. Import from here; don't hard-code paths."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
BULK = DATA / "sharadar_bulk"          # raw Sharadar bulk files (gitignored)

# Raw CSV (as downloaded) and the Parquet we convert them to (queried via DuckDB).
STOCKS_CSV = BULK / "stocks_10Y.csv"
STOCKS_PARQUET = BULK / "stocks.parquet"
STOCKS_FULL_PARQUET = BULK / "stocks_full.parquet"
TICKERS_CSV = BULK / "tickers_full.csv"
TICKERS_PARQUET = BULK / "tickers.parquet"


def stocks_parquet() -> Path:
    """The price parquet to read: the standard 10Y one if present, else the
    full-history one (CI's bootstrap builds only that)."""
    if STOCKS_PARQUET.exists() or not STOCKS_FULL_PARQUET.exists():
        return STOCKS_PARQUET
    return STOCKS_FULL_PARQUET

# Derived artifacts.
FEATURES_PKL = DATA / "survivorship_features.pkl"
RESULT_JSON = DATA / "survivorship_result.json"

# Universe / modelling defaults.
MID_UP = ("4 - Mid", "5 - Large", "6 - Mega")   # market-cap tiers to include
HORIZON = 63          # forward-return horizon in trading days (~1 quarter)
SAMPLE_STEP = 5       # sample every N trading days per ticker
WARMUP = 252          # bars needed before a ticker is eligible (1y of history)
MIN_PRICE = 5.0             # tradeability: drop penny stocks (point-in-time)
MIN_DOLLAR_VOL = 1_000_000  # tradeability: min 21d avg daily $ volume
MIN_BARS = 250        # drop tickers with fewer usable bars
N_FOLDS = 5           # walk-forward folds
