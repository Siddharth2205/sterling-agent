"""Risk-managed portfolio construction — the professional layer.

Equal-weighting the top names ignores risk, which is why our best long-only book earned
more than the market but with gut-wrenching drawdowns and no risk-adjusted edge. A real
desk instead:

  - **sizes by risk** (inverse-volatility): calmer stocks get bigger weights, jumpy ones
    smaller, so the portfolio's overall wobble is controlled;
  - **neutralizes sector bets** (score each stock vs its *own sector*), so the return comes
    from stock selection, not an accidental "overweight tech" bet;
  - **can go market-neutral** (long the best, short the worst) done right — dollar-neutral
    and sector-balanced — which cancels market crashes instead of amplifying them.

`simulate_neutral` reports risk-adjusted stats (Sharpe, max drawdown), because that — not
raw return — is what a professional judges a strategy on.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from sterling.research.portfolio_sim import _rebalance_dates, _series_stats, PERIODS_PER_YEAR

logger = logging.getLogger(__name__)


def prepare_panel(panel: pd.DataFrame, features_df: pd.DataFrame, sector_map: dict) -> pd.DataFrame:
    """Attach each row's recent volatility (from the features) and sector (from metadata)."""
    vol = features_df[["date", "ticker", "vol_63"]]
    p = panel.merge(vol, on=["date", "ticker"], how="left")
    p["sector"] = p["ticker"].map(sector_map).fillna("Unknown")
    return p


def _weights(sub: pd.DataFrame, inverse_vol: bool) -> pd.Series:
    """Position weights summing to 1: inverse-volatility (risk-sized) or equal."""
    if inverse_vol and "vol_63" in sub.columns and sub["vol_63"].notna().any():
        v = sub["vol_63"].clip(lower=0.005)
        v = v.fillna(v.median())
        w = 1.0 / v
    else:
        w = pd.Series(1.0, index=sub.index)
    return w / w.sum()


def _wturnover(cur: dict, prev: dict) -> float:
    """Sum of absolute weight changes across rebalances (proper turnover)."""
    keys = set(cur) | set(prev)
    return sum(abs(cur.get(k, 0.0) - prev.get(k, 0.0)) for k in keys)


def simulate_neutral(panel: pd.DataFrame, long_frac: float = 0.2, short_frac: float = 0.0,
                     sector_neutral: bool = True, inverse_vol: bool = True,
                     horizon: int = 63, cost_bps: float = 10.0,
                     spacing_cal_days: Optional[int] = None) -> dict:
    """Risk-managed book. Long the top `long_frac` by (optionally sector-neutralized)
    score, risk-weighted; optionally short the bottom `short_frac` (market-neutral)."""
    if panel.empty:
        return {"error": "empty panel"}
    spacing = spacing_cal_days or int(horizon * 1.4)
    rebals = _rebalance_dates(sorted(panel["date"].unique()), spacing)

    prev_l: dict = {}
    prev_s: dict = {}
    net, univ = [], []
    for d in rebals:
        day = panel[panel["date"] == d].copy()
        if len(day) < 30:
            continue
        if sector_neutral:
            day["sig"] = day["pred"] - day.groupby("sector")["pred"].transform("mean")
        else:
            day["sig"] = day["pred"]

        lo = day.nlargest(max(1, int(len(day) * long_frac)), "sig")
        wl = _weights(lo, inverse_vol)
        long_ret = float((wl.values * lo["fwd_return"].values).sum())
        gross = long_ret
        cur_l = dict(zip(lo["ticker"], wl.values))
        cost = _wturnover(cur_l, prev_l) * (cost_bps / 1e4)
        prev_l = cur_l

        if short_frac > 0:
            sh = day.nsmallest(max(1, int(len(day) * short_frac)), "sig")
            ws = _weights(sh, inverse_vol)
            short_ret = float((ws.values * sh["fwd_return"].values).sum())
            gross = long_ret - short_ret          # dollar-neutral spread
            cur_s = dict(zip(sh["ticker"], ws.values))
            cost += _wturnover(cur_s, prev_s) * (cost_bps / 1e4)
            prev_s = cur_s

        net.append(gross - cost)
        univ.append(float(day["fwd_return"].mean()))

    if len(net) < 2:
        return {"error": "not enough periods"}
    excess = np.array(net) - np.array(univ)
    t = (excess.mean() / (excess.std(ddof=1) / np.sqrt(len(excess)))
         if excess.std(ddof=1) > 0 else 0.0)
    mode = ("mkt-neutral " if short_frac > 0 else "long-only ") + \
           ("sector-neutral " if sector_neutral else "") + \
           ("inv-vol" if inverse_vol else "equal-wt")
    return {
        "mode": mode.strip(), "periods": len(net), "cost_bps": cost_bps,
        "long_frac": long_frac, "short_frac": short_frac,
        "strategy_net": _series_stats(net),
        "equal_weight_universe": _series_stats(univ),
        # Index-hedged: long the book, short a broad index (the universe). Isolates the
        # stock-selection alpha and hedges out market crashes — the professional hedge,
        # avoiding the short-squeeze risk of shorting individual bad names.
        "index_hedged_net": _series_stats(list(excess)),
        "net_alpha_vs_universe_annual_pct": round(float(excess.mean()) * PERIODS_PER_YEAR * 100, 2),
        "net_alpha_t_stat": round(float(t), 2),
    }
