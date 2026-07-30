"""Point-in-time fundamental selection (pure, data-source agnostic).

Salvaged from the deprecated yfinance pipeline. `fundamentals_as_of` picks the latest
fundamental record *known* by a given date (using each record's `available_from` filing
date, so there is no look-ahead) and derives the model's fundamental metrics. Works for
any source that yields records with an `available_from` plus the raw TTM components
(Sharadar's connector produces compatible records).
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional


def _as_date(d) -> date:
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    return datetime.fromisoformat(str(d)[:10]).date()


def fundamentals_as_of(records: list[dict], on_date, price: Optional[float]) -> dict:
    """Fundamental metrics known as of `on_date`, or {} if nothing is filed yet (→ the
    model treats it as missing, never a current-snapshot leak). `price` is the historical
    close on that date, so P/E and FCF-yield use no look-ahead."""
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
    # Records may carry the final metrics directly (the Sharadar connector does);
    # take those as-is, then derive anything still missing from raw components.
    for k in ("pe", "revenue_growth", "debt_to_equity", "roe", "fcf_yield_pct",
              "gross_profitability", "asset_growth", "margin_trend", "net_issuance"):
        if chosen.get(k) is not None:
            out[k] = chosen[k]

    eps = chosen.get("eps_ttm")
    if eps is None and chosen.get("net_income_ttm") and chosen.get("shares"):
        eps = chosen["net_income_ttm"] / chosen["shares"]
    if price and eps is not None:
        out["pe"] = (price / eps) if eps > 0 else -1.0

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

    # ── classic factor metrics (Novy-Marx quality, asset growth, issuance) ──
    gp, assets = chosen.get("gross_profit_ttm"), chosen.get("total_assets")
    if gp is not None and assets and assets > 0:
        out["gross_profitability"] = gp / assets

    assets_prior = chosen.get("total_assets_prior")
    if assets is not None and assets_prior and assets_prior > 0:
        out["asset_growth"] = (assets - assets_prior) / assets_prior

    ni_prior = chosen.get("net_income_ttm_prior")
    if (ni is not None and rev and rev > 0
            and ni_prior is not None and rev_prior and rev_prior > 0):
        out["margin_trend"] = ni / rev - ni_prior / rev_prior

    shares_prior = chosen.get("shares_prior")
    if shares is not None and shares_prior and shares_prior > 0:
        out["net_issuance"] = (shares - shares_prior) / shares_prior

    return out
