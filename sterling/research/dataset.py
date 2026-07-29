"""Download and cache ~10 years of daily prices for the research universe.

Per-ticker CSV cache so re-runs are instant and offline-friendly. Prices are
auto-adjusted (splits/dividends) and NaN-close bars are dropped (yfinance emits a
NaN row for the current unsettled session — the same bug that poisoned the old
backtest stats).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "research_cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_file(ticker: str) -> Path:
    safe = ticker.replace("/", "_").replace("\\", "_")
    return _CACHE_DIR / f"{safe}.csv"


def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    """Force a tz-naive, midnight-normalized DatetimeIndex regardless of source.

    yfinance returns a tz-aware index; a cached CSV reloads as strings. Both must end
    up identical so `.date`, position lookups and date joins behave consistently.
    """
    idx = pd.to_datetime(df.index, utc=True)
    df.index = idx.tz_convert(None).normalize()
    return df


def download_one(ticker: str, years: int = 10, force: bool = False) -> Optional[pd.DataFrame]:
    """Return a cached (or freshly downloaded) OHLCV frame for `ticker`, or None."""
    fp = _cache_file(ticker)
    if fp.exists() and not force:
        try:
            df = pd.read_csv(fp, index_col=0)
            if len(df) >= 250:
                return _normalize_index(df)
        except Exception:
            pass  # fall through to re-download

    import yfinance as yf
    end = datetime.utcnow()
    start = end - timedelta(days=years * 365 + 10)
    try:
        df = yf.Ticker(ticker).history(
            start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), auto_adjust=True
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(f"download failed {ticker}: {e}")
        return None

    if df is None or df.empty:
        return None
    df = df[df["Close"].notna()]
    if len(df) < 250:
        return None
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.to_csv(fp)
    return _normalize_index(df)


def load_prices(universe: list[str], years: int = 10, force: bool = False) -> dict[str, pd.DataFrame]:
    """Load prices for every ticker in `universe`. Skips names without enough history."""
    out: dict[str, pd.DataFrame] = {}
    for i, tk in enumerate(universe, 1):
        df = download_one(tk, years=years, force=force)
        if df is not None:
            out[tk] = df
        if i % 25 == 0:
            logger.info(f"  loaded {i}/{len(universe)} ({len(out)} usable)")
    return out


def coverage_report(prices: dict[str, pd.DataFrame]) -> dict:
    """Summarise how much usable data we actually have."""
    if not prices:
        return {"tickers": 0}
    starts = [df.index.min() for df in prices.values()]
    ends = [df.index.max() for df in prices.values()]
    bar_counts = [len(df) for df in prices.values()]
    s = pd.Series(bar_counts)
    return {
        "tickers": len(prices),
        "earliest_start": str(min(starts).date()),
        "latest_end": str(max(ends).date()),
        "median_bars": int(s.median()),
        "min_bars": int(s.min()),
        "max_bars": int(s.max()),
        "names_with_10y": int((s >= 2500).sum()),
    }
