"""Sharadar connector — survivorship-free US equities via the sharadar.com REST API.

IMPORTANT: this uses sharadar.com's own API (https://api.sharadar.com/v1.0/data/),
which is what a direct sharadar.com subscription provides — NOT the separate
Nasdaq Data Link / quandl service. Auth is a simple `?api_key=` query parameter.

Why this exists: yfinance only has today's survivors. Sharadar's `stocks` table includes
*delisted* tickers with full history (back to the 1990s), and `tickers` flags what died
and when — exactly what we need to test whether the model avoids blow-ups.

Outputs drop straight into the existing pipeline:
  - load_prices_sharadar()    -> {ticker: OHLCV DataFrame}   (like research.dataset)
  - ticker_metadata()         -> delisting flags/dates, sector, market-cap tier
  - delisting_map()           -> {ticker: (last_trade_date, terminal_return)}
  - historical_fundamentals() -> point-in-time records (only if the plan includes them)

Endpoints (columns confirmed against the API):
  tickers       ticker,name,exchange,isdelisted,category,sector,scalemarketcap,
                firstpricedate,lastpricedate
  stocks        ticker,date,open,high,low,close,closeadj,closeunadj,volume,lastupdated
  fundamentals  ticker,dimension,calendardate,date(=filing/point-in-time),pe,de,roe,
                revenue,fcf,marketcap,…                         (not in the $9 Prices plan)

Key is read from env NASDAQ_API_KEY or SHARADAR_API_KEY (put it in .env).
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_API = "https://api.sharadar.com/v1.0/data/"
_CACHE = Path(__file__).resolve().parents[2] / "data" / "sharadar_cache"
_CACHE.mkdir(parents=True, exist_ok=True)

_PAGE_LIMIT = 10000          # API max rows per request
_PACE_SECONDS = 0.5          # polite gap between requests
_COMMON_CATEGORIES = ("Domestic Common Stock", "Domestic Common Stock Primary Class")


def _api_key() -> Optional[str]:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass
    return os.getenv("NASDAQ_API_KEY") or os.getenv("SHARADAR_API_KEY")


def _get(endpoint: str, params: dict, paginate: bool = True) -> list[dict]:
    """GET an endpoint, following skip-pagination, with back-off on rate limits."""
    key = _api_key()
    if not key:
        raise RuntimeError("No API key — add NASDAQ_API_KEY=... to .env")
    rows: list[dict] = []
    skip = 0
    backoffs = [15, 30, 60, 120]
    while True:
        q = dict(params, api_key=key, format="json", limit=_PAGE_LIMIT, skip=skip)
        attempt = 0
        while True:
            try:
                r = requests.get(_API + endpoint, params=q, timeout=60)
                status = r.status_code
            except requests.RequestException:
                status = None  # network hiccup — treat as transient
            if status in (429, 500, 502, 503, 504, None) and attempt < len(backoffs):
                wait = backoffs[attempt]
                logger.warning(f"Sharadar transient error ({status}) on {endpoint}; retry in {wait}s")
                time.sleep(wait)
                attempt += 1
                continue
            if status != 200:
                # NEVER surface the URL/exception — it embeds the API key.
                raise RuntimeError(f"Sharadar {endpoint} failed (HTTP {status})")
            break
        batch = r.json().get("data", [])
        rows.extend(batch)
        time.sleep(_PACE_SECONDS)
        if not paginate or len(batch) < _PAGE_LIMIT:
            break
        skip += _PAGE_LIMIT
    return rows


def ticker_metadata(force: bool = False) -> pd.DataFrame:
    """All US common-stock tickers incl. delisted — the survivorship-free backbone."""
    fp = _CACHE / "tickers.csv"
    if fp.exists() and not force:
        return pd.read_csv(fp, parse_dates=["firstpricedate", "lastpricedate"])
    rows = _get("tickers", {"table": "stocks"})
    df = pd.DataFrame(rows)
    if not df.empty and "category" in df.columns:
        df = df[df["category"].isin(_COMMON_CATEGORIES)].copy()
    for c in ("firstpricedate", "lastpricedate"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    df.to_csv(fp, index=False)
    return df


def select_universe(meta: Optional[pd.DataFrame] = None,
                    min_scale: tuple = ("4 - Mid", "5 - Large", "6 - Mega")) -> list[str]:
    """Mid-cap-and-up common stocks (incl. delisted names of that size). Keeps the pull
    tractable while retaining the large/mid-cap blow-ups the test cares about."""
    meta = meta if meta is not None else ticker_metadata()
    if "scalemarketcap" not in meta.columns:
        return meta["ticker"].dropna().unique().tolist()
    keep = meta[meta["scalemarketcap"].isin(min_scale)]
    return keep["ticker"].dropna().unique().tolist()


def delisting_map(meta: Optional[pd.DataFrame] = None, terminal_return: float = 0.0) -> dict:
    """{ticker: (last_trade_date, terminal_return)} for delisted names. terminal_return
    is applied AFTER the last traded price (0.0 = exit at last print; -1.0 = stub to zero)."""
    meta = meta if meta is not None else ticker_metadata()
    out = {}
    for r in meta.itertuples(index=False):
        if str(getattr(r, "isdelisted", "N")) == "Y" and pd.notna(getattr(r, "lastpricedate", None)):
            out[r.ticker] = (r.lastpricedate.date(), terminal_return)
    return out


def _rows_to_ohlcv(rows: list[dict]) -> pd.DataFrame:
    g = pd.DataFrame(rows).sort_values("date")
    idx = pd.DatetimeIndex(pd.to_datetime(g["date"]).dt.tz_localize(None).dt.normalize())
    out = pd.DataFrame({
        "Open": pd.to_numeric(g["open"], errors="coerce").to_numpy(),
        "High": pd.to_numeric(g["high"], errors="coerce").to_numpy(),
        "Low": pd.to_numeric(g["low"], errors="coerce").to_numpy(),
        "Close": pd.to_numeric(g["closeadj"], errors="coerce").to_numpy(),  # adjusted
        "Volume": pd.to_numeric(g["volume"], errors="coerce").to_numpy(),
    }, index=idx)
    return out[out["Close"].notna()]


def load_prices_sharadar(tickers: list[str], years: int = 25,
                         cache: bool = True) -> dict[str, pd.DataFrame]:
    """{ticker: OHLCV df} from the `stocks` endpoint (one call per ticker; full history
    fits under the 10k-row page limit for daily data over ~40y). Per-ticker CSV cache."""
    start = (datetime.utcnow() - timedelta(days=years * 365 + 10)).strftime("%Y-%m-%d")
    out: dict[str, pd.DataFrame] = {}
    for i, tk in enumerate(tickers, 1):
        fp = _CACHE / f"px_{tk.replace('/', '_')}.csv"
        if cache and fp.exists():
            df = pd.read_csv(fp, index_col=0, parse_dates=True)
            if len(df) >= 250:
                out[tk] = df
                continue
        try:
            rows = _get("stocks", {"ticker": tk, "date.gte": start})
        except Exception as e:  # noqa: BLE001
            logger.warning(f"price fetch failed {tk}: {e}")
            continue
        if not rows:
            continue
        df = _rows_to_ohlcv(rows)
        if len(df) >= 250:
            if cache:
                df.to_csv(fp)
            out[tk] = df
        if i % 100 == 0:
            logger.info(f"  prices {i}/{len(tickers)} ({len(out)} usable)")
    return out


def date_snapshot(date: str) -> pd.DataFrame:
    """All stocks trading on `date` (YYYY-MM-DD) — the full cross-section, including
    names that later delisted. This is how we build a survivorship-free universe: from
    the price panel itself, not the (recent-only) metadata list."""
    rows = _get("stocks", {"date": date}, paginate=True)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["close"] = pd.to_numeric(df["closeadj"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df["dollar_vol"] = df["close"] * df["volume"]
    return df[["ticker", "close", "volume", "dollar_vol"]]


def build_universe_from_prices(start_year: int = 2010, end_year: int = 2026,
                               min_price: float = 5.0, top_n_per_date: int = 1500) -> dict:
    """Survivorship-free universe from monthly price snapshots. At each month-start we
    take the most liquid `top_n_per_date` names (price >= min_price) — including ones
    that later delisted. Returns {'tickers': [...], 'membership': {date: set(tickers)}}."""
    months = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-01", freq="MS")
    all_tickers: set = set()
    membership: dict = {}
    for i, m in enumerate(months, 1):
        snap = date_snapshot(m.strftime("%Y-%m-%d"))
        if snap.empty:
            continue
        snap = snap[snap["close"] >= min_price].nlargest(top_n_per_date, "dollar_vol")
        tks = set(snap["ticker"])
        membership[m.date().isoformat()] = tks
        all_tickers |= tks
        if i % 24 == 0:
            logger.info(f"  snapshots {i}/{len(months)} — universe so far {len(all_tickers)}")
    return {"tickers": sorted(all_tickers), "membership": membership}


def historical_fundamentals(ticker: str, _cache: dict = {}) -> list[dict]:
    """Point-in-time fundamentals from the `fundamentals` endpoint (ARQ). Uses the API's
    `date` field (filing date) as available_from → no look-ahead. Returns [] if the plan
    doesn't include fundamentals (e.g. the $9 Prices plan)."""
    if ticker in _cache:
        return _cache[ticker]
    try:
        rows = _get("fundamentals", {"ticker": ticker, "dimension": "ARQ"})
    except Exception as e:  # noqa: BLE001 — plan may not include fundamentals
        logger.info(f"fundamentals unavailable for {ticker}: {e}")
        _cache[ticker] = []
        return []
    recs = []
    for r in sorted(rows, key=lambda x: x.get("date", "")):
        def num(k):
            v = r.get(k)
            try:
                return float(v)
            except (TypeError, ValueError):
                return None
        recs.append({
            "available_from": str(r.get("date", ""))[:10],
            "pe": num("pe"), "revenue_growth": None,
            "debt_to_equity": num("de"), "roe": num("roe"),
            "fcf_yield_pct": (num("fcf") / num("marketcap") * 100)
                             if num("fcf") and num("marketcap") else None,
            "_revenue": num("revenue"),
        })
    for i in range(len(recs)):
        cur, prior = recs[i]["_revenue"], recs[i - 4]["_revenue"] if i >= 4 else None
        if cur and prior:
            recs[i]["revenue_growth"] = (cur - prior) / abs(prior)
    _cache[ticker] = recs
    return recs


def verify() -> dict:
    """Cheap smoke test with the user's key: auth + schema + delisted coverage."""
    if not _api_key():
        return {"ok": False, "reason": "NASDAQ_API_KEY not set in .env"}
    try:
        meta = ticker_metadata()
        n = len(meta)
        n_delisted = int((meta["isdelisted"] == "Y").sum()) if "isdelisted" in meta else 0
        uni = select_universe(meta)
        px = load_prices_sharadar(["AAPL"], years=3, cache=False)
        fun = historical_fundamentals("AAPL")
        return {
            "ok": True,
            "common_stocks_total": n,
            "delisted_included": n_delisted,
            "mid_cap_and_up_universe": len(uni),
            "sample_AAPL_price_bars": len(px.get("AAPL", [])),
            "fundamentals_in_plan": len(fun) > 0,
        }
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "reason": repr(e)[:200]}
