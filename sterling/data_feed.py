"""Market data: yfinance (prices/fundamentals), Finnhub (news/insiders), pytrends (sentiment)."""

import json
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import requests
import yfinance as yf
import finnhub
import pandas as pd

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).parent.parent / "data" / "cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

_FUNDAMENTAL_TTL = 86400   # 24h seconds
_SENTIMENT_TTL = 14400     # 4h seconds
_FX_TTL = 3600             # 1h seconds

_fx_cache: dict = {}
_finnhub_client: Optional[object] = None

_default_retries: int = 3
_default_delay: float = 2.0


def set_scan_mode(enabled: bool = True) -> None:
    """Reduce retries/delay for the scan path where speed > robustness."""
    global _default_retries, _default_delay
    if enabled:
        _default_retries = 1
        _default_delay = 0.5
    else:
        _default_retries = 3
        _default_delay = 2.0


def _get_finnhub(api_key: str):
    global _finnhub_client
    if _finnhub_client is None:
        _finnhub_client = finnhub.Client(api_key=api_key)
    return _finnhub_client


def _cache_path(key: str) -> Path:
    safe = key.replace("/", "_").replace(".", "_")
    return _CACHE_DIR / f"{safe}.json"


def _cache_read(key: str, ttl: int) -> Optional[dict]:
    p = _cache_path(key)
    if not p.exists():
        return None
    age = time.time() - p.stat().st_mtime
    if age > ttl:
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def _cache_write(key: str, data: dict) -> None:
    p = _cache_path(key)
    with open(p, "w") as f:
        json.dump(data, f)


def _retry(fn, retries: int | None = None, delay: float | None = None):
    """Exponential backoff retry wrapper.  Defaults honour set_scan_mode()."""
    if retries is None:
        retries = _default_retries
    if delay is None:
        delay = _default_delay
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            if attempt == retries - 1:
                raise
            wait = delay * (2 ** attempt)
            logger.warning(f"Retry {attempt+1}/{retries} after {wait:.1f}s — {e}")
            time.sleep(wait)


# ── FX ──────────────────────────────────────────────────────────────────────

def get_usd_cad() -> float:
    """Live USD/CAD exchange rate via yfinance."""
    cached = _cache_read("usdcad_fx", _FX_TTL)
    if cached:
        return cached["rate"]

    def _fetch():
        ticker = yf.Ticker("CAD=X")
        hist = ticker.history(period="1d")
        if hist.empty:
            raise ValueError("No FX data returned")
        return float(hist["Close"].iloc[-1])

    rate = _retry(_fetch)
    _cache_write("usdcad_fx", {"rate": rate})
    return rate


# ── QUOTES ──────────────────────────────────────────────────────────────────

def get_quote(ticker: str) -> dict:
    """Live quote for a ticker. Returns price in native currency."""
    def _fetch():
        t = yf.Ticker(ticker)
        info = t.fast_info
        price = getattr(info, "last_price", None)
        prev_close = getattr(info, "previous_close", None)
        volume = getattr(info, "three_month_average_volume", None)

        # Fallback: pull 1-day history
        if price is None:
            hist = t.history(period="2d")
            if not hist.empty:
                price = float(hist["Close"].iloc[-1])
                prev_close = float(hist["Close"].iloc[-2]) if len(hist) > 1 else price

        if price is None:
            raise ValueError(f"No price for {ticker}")

        change = price - prev_close if prev_close else 0.0
        change_pct = (change / prev_close * 100) if prev_close else 0.0

        return {
            "ticker": ticker,
            "price": round(float(price), 4),
            "prev_close": round(float(prev_close), 4) if prev_close else None,
            "change": round(change, 4),
            "change_pct": round(change_pct, 2),
            "volume": int(volume) if volume else None,
            "timestamp": datetime.utcnow().isoformat(),
        }

    return _retry(_fetch)


def get_history(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """OHLCV history. period examples: '3mo', '1y', '3y'."""
    def _fetch():
        t = yf.Ticker(ticker)
        df = t.history(period=period, interval=interval, auto_adjust=True)
        if df.empty:
            raise ValueError(f"No history for {ticker}")
        return df

    return _retry(_fetch)


# ── FUNDAMENTALS ─────────────────────────────────────────────────────────────

def get_fundamentals(ticker: str) -> dict:
    """Key fundamentals with 24h disk cache. Returns empty dict on failure.

    For CDR tickers (.NE), redirects the yfinance lookup to the underlying US symbol
    so P/E, FCF yield, etc. are populated (CDRs have no standalone fundamental data).
    """
    # CDR redirect: fetch fundamentals from the US underlying
    fetch_ticker = ticker
    if ticker.endswith(".NE"):
        try:
            from sterling.cdr_mapping import CDR_UNIVERSE
            fetch_ticker = CDR_UNIVERSE.get(ticker, ticker.replace(".NE", ""))
        except ImportError:
            fetch_ticker = ticker.replace(".NE", "")

    cache_key = f"fundamentals_{fetch_ticker}"
    cached = _cache_read(cache_key, _FUNDAMENTAL_TTL)
    if cached:
        return cached

    def _fetch():
        t = yf.Ticker(fetch_ticker)
        info = t.info or {}

        pe = info.get("trailingPE") or info.get("forwardPE")
        eps = info.get("trailingEps")
        debt_eq = info.get("debtToEquity")
        roe = info.get("returnOnEquity")
        rev_growth = info.get("revenueGrowth")
        fcf = info.get("freeCashflow")
        market_cap = info.get("marketCap")
        shares = info.get("sharesOutstanding", 1) or 1
        fcf_yield = (fcf / market_cap * 100) if (fcf and market_cap) else None

        sector = info.get("sector", "Unknown")
        analyst_target = info.get("targetMeanPrice")

        return {
            "ticker": ticker,
            "pe": pe,
            "eps": eps,
            "debt_to_equity": debt_eq,
            "roe": roe,
            "revenue_growth": rev_growth,
            "fcf": fcf,
            "fcf_yield_pct": round(fcf_yield, 2) if fcf_yield is not None else None,
            "market_cap": market_cap,
            "sector": sector,
            "analyst_target": analyst_target,
            "cached_at": datetime.utcnow().isoformat(),
        }

    try:
        data = _retry(_fetch)
        _cache_write(cache_key, data)
        return data
    except Exception as e:
        logger.warning(f"Fundamentals fetch failed for {ticker}: {e}")
        return {"ticker": ticker, "error": str(e)}


# ── NEWS / SENTIMENT ─────────────────────────────────────────────────────────

def get_news_sentiment(ticker: str, api_key: str) -> dict:
    """Finnhub news sentiment. Returns score in [-1, 1] range."""
    # Strip exchange suffixes for Finnhub (uses US symbols or base)
    # .NE CDRs redirect to their underlying US symbol
    base = ticker.replace(".TO", "").replace(".V", "").replace(".TSX", "").replace(".NE", "")

    cache_key = f"sentiment_{base}"
    cached = _cache_read(cache_key, _SENTIMENT_TTL)
    if cached:
        return cached

    try:
        client = _get_finnhub(api_key)
        now = datetime.utcnow()
        from_dt = (now - timedelta(days=7)).strftime("%Y-%m-%d")
        to_dt = now.strftime("%Y-%m-%d")

        news = _retry(lambda: client.company_news(base, _from=from_dt, to=to_dt))
        if not news:
            result = {"ticker": ticker, "sentiment_score": 0.0, "article_count": 0, "cached_at": now.isoformat()}
            _cache_write(cache_key, result)
            return result

        # Finnhub news items have no built-in sentiment score on free tier —
        # approximate by counting positive/negative keywords in headlines
        positive = {"beat", "surge", "gain", "record", "strong", "growth", "upgrade", "buy", "profit", "rise", "rally"}
        negative = {"miss", "drop", "loss", "decline", "cut", "downgrade", "sell", "warn", "fall", "crash", "weak"}

        pos_count = 0
        neg_count = 0
        for item in news:
            headline = (item.get("headline", "") + " " + item.get("summary", "")).lower()
            pos_count += sum(1 for w in positive if w in headline)
            neg_count += sum(1 for w in negative if w in headline)

        total = pos_count + neg_count
        score = (pos_count - neg_count) / total if total > 0 else 0.0
        score = max(-1.0, min(1.0, score))

        result = {
            "ticker": ticker,
            "sentiment_score": round(score, 3),
            "article_count": len(news),
            "pos_signals": pos_count,
            "neg_signals": neg_count,
            "cached_at": now.isoformat(),
        }
        _cache_write(cache_key, result)
        return result

    except Exception as e:
        logger.warning(f"News sentiment failed for {ticker}: {e}")
        return {"ticker": ticker, "sentiment_score": 0.0, "article_count": 0, "error": str(e)}


def get_insider_activity(ticker: str, api_key: str) -> dict:
    """Finnhub insider transactions. Returns buy/sell ratio and net shares."""
    base = ticker.replace(".TO", "").replace(".V", "").replace(".NE", "")

    cache_key = f"insider_{base}"
    cached = _cache_read(cache_key, _FUNDAMENTAL_TTL)
    if cached:
        return cached

    try:
        client = _get_finnhub(api_key)
        txns = _retry(lambda: client.stock_insider_transactions(base, "2024-01-01", datetime.utcnow().strftime("%Y-%m-%d")))
        data_list = txns.get("data", []) if isinstance(txns, dict) else []

        buys = [t for t in data_list if t.get("transactionCode") in ("P", "A")]
        sells = [t for t in data_list if t.get("transactionCode") in ("S", "D")]

        buy_shares = sum(abs(t.get("share", 0) or 0) for t in buys)
        sell_shares = sum(abs(t.get("share", 0) or 0) for t in sells)
        total_shares = buy_shares + sell_shares

        ratio = buy_shares / total_shares if total_shares > 0 else 0.5
        net = buy_shares - sell_shares

        result = {
            "ticker": ticker,
            "buy_shares": buy_shares,
            "sell_shares": sell_shares,
            "net_shares": net,
            "buy_ratio": round(ratio, 3),
            "transaction_count": len(data_list),
            "cached_at": datetime.utcnow().isoformat(),
        }
        _cache_write(cache_key, result)
        return result

    except Exception as e:
        logger.warning(f"Insider data failed for {ticker}: {e}")
        return {"ticker": ticker, "buy_ratio": 0.5, "net_shares": 0, "transaction_count": 0, "error": str(e)}


# ── GOOGLE TRENDS ────────────────────────────────────────────────────────────

def get_trends_slope(ticker: str) -> float:
    """Google Trends interest slope over 90 days. Positive = rising retail interest."""
    base = ticker.replace(".TO", "").replace(".V", "")
    cache_key = f"trends_{base}"
    cached = _cache_read(cache_key, _SENTIMENT_TTL)
    if cached:
        return cached.get("slope", 0.0)

    try:
        from pytrends.request import TrendReq
        pt = TrendReq(hl="en-CA", tz=-240)
        pt.build_payload([base], timeframe="today 3-m", geo="CA")
        df = pt.interest_over_time()
        if df.empty or base not in df.columns:
            return 0.0

        series = df[base].values.astype(float)
        if len(series) < 4:
            return 0.0

        import numpy as np
        x = list(range(len(series)))
        slope = float(np.polyfit(x, series, 1)[0])
        # Normalise slope to [-1, 1] using a reasonable max of 5 pts/week
        normalised = max(-1.0, min(1.0, slope / 5.0))

        _cache_write(cache_key, {"slope": normalised})
        return normalised

    except Exception as e:
        logger.warning(f"Google Trends failed for {ticker}: {e}")
        return 0.0


# ── MACRO INDICATORS ─────────────────────────────────────────────────────────

def get_macro_indicators() -> dict:
    """VIX, USD/CAD, oil price (WTI), TSX composite intraday change."""
    cache_key = "macro_indicators"
    cached = _cache_read(cache_key, 1800)  # 30-min cache
    if cached:
        return cached

    results: dict = {}

    def _safe_price(sym: str) -> Optional[float]:
        try:
            t = yf.Ticker(sym)
            hist = t.history(period="2d")
            if hist.empty:
                return None
            return float(hist["Close"].iloc[-1])
        except Exception:
            return None

    def _safe_change_pct(sym: str) -> Optional[float]:
        try:
            t = yf.Ticker(sym)
            hist = t.history(period="2d")
            if len(hist) < 2:
                return None
            prev = float(hist["Close"].iloc[-2])
            cur = float(hist["Close"].iloc[-1])
            return (cur - prev) / prev * 100
        except Exception:
            return None

    results["vix"] = _safe_price("^VIX")
    results["usd_cad"] = _safe_price("CAD=X")
    results["wti_oil"] = _safe_price("CL=F")
    results["tsx_change_pct"] = _safe_change_pct("^GSPTSE")
    results["timestamp"] = datetime.utcnow().isoformat()

    # Risk-off flags
    results["risk_off"] = any([
        results["vix"] and results["vix"] > 25,
        results["tsx_change_pct"] and results["tsx_change_pct"] < -2.0,
    ])
    results["oil_crash"] = (
        results["wti_oil"] is not None
        and results["tsx_change_pct"] is not None
        and results["tsx_change_pct"] < -3.0
    )

    _cache_write(cache_key, results)
    return results


# ── VOLUME FILTER ────────────────────────────────────────────────────────────

def get_avg_volume(ticker: str, days: int = 30) -> Optional[float]:
    """Average daily volume over last N trading days."""
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period=f"{days*2}d")  # fetch extra to ensure N trading days
        if hist.empty:
            return None
        return float(hist["Volume"].tail(days).mean())
    except Exception as e:
        logger.warning(f"Volume fetch failed for {ticker}: {e}")
        return None
