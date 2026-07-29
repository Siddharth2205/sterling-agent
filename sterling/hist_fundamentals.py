"""Point-in-time fundamentals for honest backtesting.

The live model reads a *current* fundamentals snapshot (yfinance ``.info``), which
is correct at a live decision point. Feeding that same snapshot into a historical
backtest is look-ahead bias: a 2024 scan would "know" a company's 2026 P/E, revenue
growth and ROE. Since fundamentals are 40% of the score, that quietly inflates every
backtest number.

This module reconstructs the same five fundamental metrics
(``pe``, ``fcf_yield_pct``, ``debt_to_equity``, ``revenue_growth``, ``roe``)
*as they would have been known* on a given date, from dated financial statements:

  - Annual statements (``income_stmt`` / ``balance_sheet`` / ``cashflow``) form the
    backbone — they reach ~4 fiscal years back and cover the whole backtest window.
  - Quarterly statements refine recent dates with a trailing-twelve-month (TTM) view.

Every statement period only becomes usable ``FILING_LAG`` days after its period end,
approximating when the numbers were actually filed and public. When no statement is
available as of a date, the caller gets an empty dict → the fundamental axis scores a
neutral 50 (never the current snapshot). Coverage is measurable, not assumed.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd

from sterling.data_feed import _cache_read, _cache_write, _retry

logger = logging.getLogger(__name__)

# Filing lag: a fiscal period's figures are only "known" this many days after the
# period end. Canadian issuers file annual financials within ~90 days and interim
# within ~45; we use conservative (larger) lags so we never peek at unfiled numbers.
ANNUAL_LAG_DAYS = 90
QUARTERLY_LAG_DAYS = 50

# Cache TTL for the reconstructed period series (statements change quarterly at most).
_HIST_FUND_TTL = 7 * 86400  # 7 days

# Row-label aliases across yfinance statement schemas.
_REVENUE = ("Total Revenue", "Operating Revenue")
_NET_INCOME = ("Net Income", "Net Income Common Stockholders", "Net Income Continuous Operations")
_DILUTED_EPS = ("Diluted EPS", "Basic EPS")
_EQUITY = ("Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest")
_TOTAL_DEBT = ("Total Debt",)
_SHARES = ("Ordinary Shares Number", "Share Issued", "Diluted Average Shares")
_FCF = ("Free Cash Flow",)
_OCF = ("Operating Cash Flow", "Cash Flow From Continuing Operating Activities")
_CAPEX = ("Capital Expenditure", "Capital Expenditure Reported")


def _row(df: Optional[pd.DataFrame], col, names) -> Optional[float]:
    """First present, non-NaN value among *names* for statement column *col*."""
    if df is None or df.empty or col not in df.columns:
        return None
    for n in names:
        if n in df.index:
            v = df.loc[n, col]
            try:
                f = float(v)
            except (TypeError, ValueError):
                continue
            if pd.notna(f):
                return f
    return None


def _nearest_col(df: Optional[pd.DataFrame], target, tol_days: int = 15):
    """Statement column whose period end is closest to *target* within *tol_days*."""
    if df is None or df.empty:
        return None
    best, best_gap = None, None
    for c in df.columns:
        try:
            gap = abs((c.date() - target).days)
        except AttributeError:
            continue
        if gap <= tol_days and (best_gap is None or gap < best_gap):
            best, best_gap = c, gap
    return best


def _resolve_ticker(ticker: str) -> str:
    """CDR (.NE) fundamentals come from the US underlying, mirroring data_feed."""
    if ticker.endswith(".NE"):
        try:
            from sterling.cdr_mapping import CDR_UNIVERSE
            return CDR_UNIVERSE.get(ticker, ticker.replace(".NE", ""))
        except ImportError:
            return ticker.replace(".NE", "")
    return ticker


def _build_records(income, qincome, balance, qbalance, cashflow, qcashflow) -> list[dict]:
    """Assemble as-of fundamental records (flows are TTM; stocks are point-in-time)."""
    records: list[dict] = []

    # ── Annual backbone ────────────────────────────────────────────────────────
    if income is not None and not income.empty:
        cols = list(income.columns)  # newest → oldest
        for i, col in enumerate(cols):
            rev = _row(income, col, _REVENUE)
            ni = _row(income, col, _NET_INCOME)
            if rev is None and ni is None:
                continue
            prior = cols[i + 1] if i + 1 < len(cols) else None
            rev_prior = _row(income, prior, _REVENUE) if prior is not None else None
            bcol = _nearest_col(balance, col.date())
            ccol = _nearest_col(cashflow, col.date())
            fcf = _row(cashflow, ccol, _FCF)
            if fcf is None:
                ocf, capex = _row(cashflow, ccol, _OCF), _row(cashflow, ccol, _CAPEX)
                fcf = (ocf + capex) if (ocf is not None and capex is not None) else None
            records.append({
                "period_end": col.date().isoformat(),
                "source": "annual",
                "available_from": (col.date() + timedelta(days=ANNUAL_LAG_DAYS)).isoformat(),
                "revenue_ttm": rev,
                "revenue_ttm_prior": rev_prior,
                "net_income_ttm": ni,
                "eps_ttm": _row(income, col, _DILUTED_EPS),
                "total_equity": _row(balance, bcol, _EQUITY),
                "total_debt": _row(balance, bcol, _TOTAL_DEBT),
                "shares": _row(balance, bcol, _SHARES),
                "fcf_ttm": fcf,
            })

    # ── Quarterly refinement (needs 4 trailing quarters for TTM flows) ──────────
    if qincome is not None and not qincome.empty:
        qcols = list(qincome.columns)  # newest → oldest
        for i in range(len(qcols) - 3):
            window = qcols[i:i + 4]
            revs = [_row(qincome, c, _REVENUE) for c in window]
            nis = [_row(qincome, c, _NET_INCOME) for c in window]
            if any(v is None for v in revs) or any(v is None for v in nis):
                continue
            end = qcols[i]  # most recent quarter end in the window
            # Prior-year TTM for YoY growth, if 8 quarters are available.
            prior_window = qcols[i + 4:i + 8]
            rev_prior = None
            if len(prior_window) == 4:
                pr = [_row(qincome, c, _REVENUE) for c in prior_window]
                rev_prior = sum(pr) if all(v is not None for v in pr) else None
            eps_parts = [_row(qincome, c, _DILUTED_EPS) for c in window]
            eps_ttm = sum(eps_parts) if all(v is not None for v in eps_parts) else None
            fcf_parts = []
            for c in window:
                f = _row(qcashflow, c, _FCF)
                if f is None:
                    o, cx = _row(qcashflow, c, _OCF), _row(qcashflow, c, _CAPEX)
                    f = (o + cx) if (o is not None and cx is not None) else None
                fcf_parts.append(f)
            fcf_ttm = sum(fcf_parts) if all(v is not None for v in fcf_parts) else None
            bcol = _nearest_col(qbalance, end.date())
            records.append({
                "period_end": end.date().isoformat(),
                "source": "quarterly",
                "available_from": (end.date() + timedelta(days=QUARTERLY_LAG_DAYS)).isoformat(),
                "revenue_ttm": sum(revs),
                "revenue_ttm_prior": rev_prior,
                "net_income_ttm": sum(nis),
                "eps_ttm": eps_ttm,
                "total_equity": _row(qbalance, bcol, _EQUITY),
                "total_debt": _row(qbalance, bcol, _TOTAL_DEBT),
                "shares": _row(qbalance, bcol, _SHARES),
                "fcf_ttm": fcf_ttm,
            })

    records.sort(key=lambda r: r["available_from"])
    return records


def get_historical_fundamentals(ticker: str) -> list[dict]:
    """Reconstructed point-in-time fundamental records for *ticker*, cached to disk.

    Returns a list sorted by ``available_from`` (ISO date). Empty on failure.
    """
    fetch = _resolve_ticker(ticker)
    cache_key = f"histfund_{fetch}"
    cached = _cache_read(cache_key, _HIST_FUND_TTL)
    if cached is not None:
        return cached.get("records", [])

    def _fetch():
        import yfinance as yf
        t = yf.Ticker(fetch)
        return _build_records(
            t.income_stmt, t.quarterly_income_stmt,
            t.balance_sheet, t.quarterly_balance_sheet,
            t.cashflow, t.quarterly_cashflow,
        )

    try:
        records = _retry(_fetch)
    except Exception as e:  # noqa: BLE001 — network/parse failures degrade to neutral
        logger.warning(f"Historical fundamentals failed for {ticker}: {e}")
        records = []

    _cache_write(cache_key, {"records": records, "cached_at": datetime.utcnow().isoformat()})
    return records


def _as_date(d) -> date:
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    return datetime.fromisoformat(str(d)[:10]).date()


def fundamentals_as_of(records: list[dict], on_date, price: Optional[float]) -> dict:
    """Build a ``score_fundamental``-compatible dict from the latest record known by
    *on_date*. Returns ``{}`` when nothing is known yet (→ neutral 50, never the
    current snapshot). *price* is the historical close on that date (no look-ahead).
    """
    if not records:
        return {}
    target = _as_date(on_date)
    chosen = None
    for r in records:  # sorted ascending by available_from
        if _as_date(r["available_from"]) <= target:
            chosen = r
        else:
            break
    if chosen is None:
        return {}

    out: dict = {}
    eps = chosen.get("eps_ttm")
    if eps is None and chosen.get("net_income_ttm") and chosen.get("shares"):
        eps = chosen["net_income_ttm"] / chosen["shares"]
    if price and eps is not None:
        out["pe"] = (price / eps) if eps > 0 else -1.0  # <0 → score_fundamental penalises

    rev, rev_prior = chosen.get("revenue_ttm"), chosen.get("revenue_ttm_prior")
    if rev is not None and rev_prior:
        out["revenue_growth"] = (rev - rev_prior) / abs(rev_prior)

    debt, equity = chosen.get("total_debt"), chosen.get("total_equity")
    if debt is not None and equity and equity > 0:
        out["debt_to_equity"] = debt / equity

    ni = chosen.get("net_income_ttm")
    if ni is not None and equity and equity > 0:
        out["roe"] = ni / equity

    fcf, shares = chosen.get("fcf_ttm"), chosen.get("shares")
    if fcf is not None and price and shares:
        mkt_cap = price * shares
        if mkt_cap > 0:
            out["fcf_yield_pct"] = fcf / mkt_cap * 100

    return out
