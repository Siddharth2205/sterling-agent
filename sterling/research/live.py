"""Live paper-trading harness for the risk-managed market-neutral book.

Not a money play — a way to check whether the ~0.4-Sharpe backtest edge holds up on data
the model has never seen, and to learn what running a real book feels like.

Flow:
  book      -> train the final model on all history, build TODAY's sector-neutral,
               inverse-vol top-quintile long book, and append it to a paper ledger.
  evaluate  -> for every book already in the ledger, compare its realized return (delisting
               -aware) against the equal-weight universe over the same window.

A genuine forward test = refresh the Sharadar bulk (`parquet`), rebuild `features`, then run
`book` again each quarter. Each logged book is timestamped and scored later by `evaluate`.
"""

from __future__ import annotations

import logging
from datetime import date

import numpy as np
import pandas as pd

from sterling.research import config, store
from sterling.research import model as M

logger = logging.getLogger(__name__)

MODEL_PATH = config.DATA / "live_model.joblib"
LEDGER = config.DATA / "paper_ledger.csv"


def _features_path():
    full = config.DATA / "survivorship_full_features.pkl"
    return full if full.exists() else config.FEATURES_PKL


def train_and_save(features_pkl=None) -> list:
    """Fit the final model on all labelled history and persist it."""
    import joblib
    df = pd.read_pickle(features_pkl or _features_path())
    mdl, used = M.train_final_model(df)
    joblib.dump({"model": mdl, "features": used}, MODEL_PATH)
    logger.info(f"trained final model on {len(df)} rows -> {MODEL_PATH.name}")
    return used


def _load_model():
    import joblib
    if not MODEL_PATH.exists():
        train_and_save()
    d = joblib.load(MODEL_PATH)
    return d["model"], d["features"]


def current_cross_section(recent_days: int = 10) -> pd.DataFrame:
    """Build TODAY's feature cross-section from the latest prices — one row per currently
    active mid-cap-and-up name, features evaluated at its most recent bar. Unlike the
    labelled matrix (whose newest rows are just-delisted names ~3 months stale), this is
    the live snapshot to rank. Fundamental/macro features are left NaN — they shift all
    names equally on a date, so they don't change the cross-sectional ranking."""
    from sterling.research import features as F
    from sterling.research.survivorship import select_universe

    meta = store.load_tickers()
    active = set(meta[meta["isdelisted"] != "Y"]["ticker"].dropna())
    uni = active & select_universe(meta)
    prices = store.load_prices(uni)
    if not prices:
        return pd.DataFrame()
    gmax = max(df.index[-1].date() for df in prices.values())

    rows = []
    for tk, df in prices.items():
        if len(df) < 252 or (gmax - df.index[-1].date()).days > recent_days:
            continue                                    # too short, or not currently trading
        # Tradeability screen — only names you could actually buy (no penny/illiquid junk).
        price = float(df["Close"].iloc[-1])
        dvol = float((df["Close"] * df["Volume"]).tail(21).mean())
        if price < config.MIN_PRICE or dvol < config.MIN_DOLLAR_VOL:
            continue
        fr = F._price_feature_frame(df).iloc[-1]
        row = {"date": df.index[-1].date(), "ticker": tk}
        for c in F.PRICE_FEATURES:
            row[c] = float(fr[c]) if pd.notna(fr[c]) else np.nan
        for c in F.FUND_FEATURES + F.MACRO_FEATURES:
            row[c] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def generate_book(features_pkl=None, long_frac: float = 0.2,
                  today: pd.DataFrame | None = None) -> tuple[date, pd.DataFrame]:
    """Build the current market-neutral long book: sector-neutralized top `long_frac`,
    inverse-volatility weighted. Returns (as-of date, book)."""
    today = today if today is not None else current_cross_section()
    if today.empty:
        return date.today(), today
    asof = today["date"].max()
    today = today.copy()
    mdl, used = _load_model()
    today["pred"] = mdl.predict(today[used])

    tk = store.load_tickers()
    sect = dict(zip(tk["ticker"], tk["sector"]))
    today["sector"] = today["ticker"].map(sect).fillna("Unknown")
    # Sector-neutralize the score so we bet on stocks, not sectors.
    today["sig"] = today["pred"] - today.groupby("sector")["pred"].transform("mean")

    n = max(1, int(len(today) * long_frac))
    book = today.nlargest(n, "sig").copy()
    inv = 1.0 / book["vol_63"].clip(lower=1e-3)      # calmer names get more weight
    book["weight"] = (inv / inv.sum()).round(5)
    return asof, book[["ticker", "sector", "pred", "weight"]].reset_index(drop=True)


def log_book(asof: date, book: pd.DataFrame) -> int:
    """Append a dated book to the paper ledger (idempotent per rebalance date)."""
    rows = book.assign(rebalance_date=str(asof))[
        ["rebalance_date", "ticker", "sector", "pred", "weight"]]
    if LEDGER.exists():
        prev = pd.read_csv(LEDGER)
        prev = prev[prev["rebalance_date"] != str(asof)]        # replace same-date book
        rows = pd.concat([prev, rows], ignore_index=True)
    rows.to_csv(LEDGER, index=False)
    return len(book)


def evaluate() -> dict:
    """Score every logged book against the equal-weight universe, delisting-aware."""
    if not LEDGER.exists():
        return {"error": "no ledger yet — run `book` first"}
    led = pd.read_csv(LEDGER)
    tickers = set(led["ticker"])
    prices = store.load_prices(tickers)
    meta = store.load_tickers()
    delist = {r.ticker: r.lastpricedate for r in meta.itertuples(index=False)
              if str(getattr(r, "isdelisted", "N")) == "Y" and pd.notna(getattr(r, "lastpricedate", None))}

    def ret_since(tk, d0) -> float | None:
        df = prices.get(tk)
        if df is None:
            return None
        sub = df[df.index.date >= d0]
        if sub.empty:
            return None
        p0 = float(sub["Close"].iloc[0])
        p1 = float(df["Close"].iloc[-1])              # latest price (or last before delist)
        return p1 / p0 - 1 if p0 > 0 else None

    out = []
    for d0s, g in led.groupby("rebalance_date"):
        d0 = pd.to_datetime(d0s).date()
        rets, ws = [], []
        for r in g.itertuples(index=False):
            rr = ret_since(r.ticker, d0)
            if rr is not None:
                rets.append(rr)
                ws.append(r.weight)
        if not rets:
            continue
        ws = np.array(ws) / np.sum(ws)
        book_ret = float(np.dot(ws, rets))
        out.append({"rebalance_date": d0s, "names": len(rets),
                    "book_return_pct": round(book_ret * 100, 2)})
    return {"books": out, "note": "returns are since each book's log date, to the latest price"}
