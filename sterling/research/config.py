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
TICKERS_CSV = BULK / "tickers_full.csv"
TICKERS_PARQUET = BULK / "tickers.parquet"

# Derived artifacts.
FEATURES_PKL = DATA / "survivorship_features.pkl"
RESULT_JSON = DATA / "survivorship_result.json"

# Universe / modelling defaults.
MID_UP = ("4 - Mid", "5 - Large", "6 - Mega")   # market-cap tiers to include
HORIZON = 63          # forward-return horizon in trading days (~1 quarter)
SAMPLE_STEP = 5       # sample every N trading days per ticker
WARMUP = 252          # bars needed before a ticker is eligible (1y of history)
MIN_BARS = 250        # drop tickers with fewer usable bars
N_FOLDS = 5           # walk-forward folds
