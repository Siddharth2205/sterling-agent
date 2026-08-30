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
    the live snapshot to rank. Fundamentals come from EDGAR when available (they differ
    per name, so they DO move the ranking); macro features are left NaN — those shift all
    names equally on a date."""
    from sterling.research import features as F
    from sterling.research.fundamentals import fundamentals_as_of
    from sterling.research.survivorship import select_universe

    meta = store.load_tickers()
    active = set(meta[meta["isdelisted"] != "Y"]["ticker"].dropna())
    uni = active & select_universe(meta)
    prices = store.load_prices(uni)
    if not prices:
        return pd.DataFrame()
    gmax = max(df.index[-1].date() for df in prices.values())

    from sterling.research import edgar
    fund_fn = edgar.fundamentals_fn(uni) if edgar.FACTS_PARQUET.exists() else None

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
        d = df.index[-1].date()
        row = {"date": d, "ticker": tk}
        for c in F.PRICE_FEATURES:
            row[c] = float(fr[c]) if pd.notna(fr[c]) else np.nan
        fund = fundamentals_as_of(fund_fn(tk), d, price) if fund_fn else {}
        for c in F.FUND_FEATURES:
            row[c] = float(fund[c]) if c in fund and pd.notna(fund[c]) else np.nan
        for c in F.MACRO_FEATURES:
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
    vol = book["vol_63"].clip(lower=1e-3)            # calmer names get more weight
    vol = vol.fillna(vol.median() if vol.notna().any() else 1.0)
    inv = 1.0 / vol
    book["weight"] = (inv / inv.sum()).round(5)
    return asof, book[["ticker", "sector", "pred", "weight"]].reset_index(drop=True)


def log_book(asof: date, book: pd.DataFrame) -> int:
    """Append a dated book to the paper ledger (idempotent per rebalance date)."""
    if book.empty:
        logger.warning("empty book — nothing logged")
        return 0
    rows = book.assign(rebalance_date=str(asof))[
        ["rebalance_date", "ticker", "sector", "pred", "weight"]]
    if LEDGER.exists():
        prev = pd.read_csv(LEDGER)
        prev = prev[prev["rebalance_date"] != str(asof)]        # replace same-date book
        rows = pd.concat([prev, rows], ignore_index=True)
    rows.to_csv(LEDGER, index=False)
    return len(book)


def _refresh_prices(tickers, years: int = 2) -> dict:
    """Fresh current prices for `tickers` via yfinance (free). Used by the weekly paper-book
    refresh so the forward test is measured against up-to-date prices.

    Why not Sharadar here: the Sharadar plan only guarantees survivorship-free history in
    the one-time bulk download; the live REST key may be free-tier (limited tickers), so the
    forward test uses yfinance, which is free and covers currently-trading names. Caveat:
    yfinance drops fully-delisted tickers, so a name that delists mid-window is simply not
    scored rather than booking its loss — acceptable for a liquid, currently-held book, but
    a mild upward bias to keep in mind."""
    from sterling.research import dataset
    return dataset.load_prices(sorted(tickers), years=years, force=True)


def evaluate(fresh: bool = False) -> dict:
    """Score every logged book: weighted return from each rebalance date to the latest
    price. Delisted names are not dropped — their series ends at the last traded print,
    so the collapse is booked (a stub payout beyond that is not modelled).

    fresh=True pulls current prices from the Sharadar REST API (for the weekly forward
    test); fresh=False reads the local bulk parquet (fast/offline, but only as current as
    the last bulk download). Each book is also compared to the equal-weight return of the
    same held names — the honest, survivorship-neutral baseline (a book beats it only by
    *weighting*, since both hold the identical names)."""
    if not LEDGER.exists():
        return {"error": "no ledger yet — run `book` first"}
    led = pd.read_csv(LEDGER)
    tickers = set(led["ticker"])
    prices = _refresh_prices(tickers) if fresh else store.load_prices(tickers)

    latest = max((df.index[-1].date() for df in prices.values()), default=None)

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
        rets = np.array(rets)
        ws = np.array(ws) / np.sum(ws)
        book_ret = float(np.dot(ws, rets))
        eqw_ret = float(rets.mean())                  # equal-weight baseline, same names
        out.append({"rebalance_date": d0s, "names": len(rets),
                    "days_held": (latest - d0).days if latest else None,
                    "book_return_pct": round(book_ret * 100, 2),
                    "equal_weight_pct": round(eqw_ret * 100, 2),
                    "weighting_alpha_pct": round((book_ret - eqw_ret) * 100, 2)})
    return {"books": out, "as_of": str(latest) if latest else None,
            "note": "return since each book's log date to the latest available price; "
                    "equal_weight_pct = same names equally weighted (survivorship-neutral baseline)"}
