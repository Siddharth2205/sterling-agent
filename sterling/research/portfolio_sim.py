"""Net-of-cost portfolio simulation on out-of-sample predictions.

Turns the signal-level result into a strategy-level one. At each rebalance it holds the
top-N names by predicted excess return, equal-weighted, held one full horizon (so periods
don't overlap), charged turnover costs on entry/exit.

Two baselines are reported, and the second is the honest one:
  - vs the market benchmark (SPY/XIC blend): the headline, but survivorship-inflated
    because the universe is today's survivors.
  - vs the EQUAL-WEIGHT UNIVERSE of the same names: survivorship-*neutral*. Both the
    top-N basket and this baseline are drawn from the identical survivor set, so their
    spread is not a survivorship artifact. If top-N beats equal-weight-universe net of
    costs, the edge is real beyond "we picked a basket of survivors."
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PERIODS_PER_YEAR = 252 / 63  # quarterly rebalance ≈ 4/yr


def _rebalance_dates(all_dates: list, spacing_cal_days: int) -> list:
    """Greedily pick rebalance dates ~spacing apart so holding periods don't overlap."""
    if not all_dates:
        return []
    picks = [all_dates[0]]
    for d in all_dates[1:]:
        if (d - picks[-1]).days >= spacing_cal_days:
            picks.append(d)
    return picks


def simulate(
    panel: pd.DataFrame,
    top_n: int = 15,
    horizon: int = 63,
    cost_bps: float = 10.0,
    spacing_cal_days: Optional[int] = None,
) -> dict:
    """Simulate the top-N strategy on the OOS prediction `panel`.

    panel columns: date, ticker, pred, fwd_return (raw H-day return), bench_fwd, label.
    cost_bps: one-way transaction cost in basis points (10 = 0.10%).
    """
    if panel.empty:
        return {"error": "empty panel"}
    spacing_cal_days = spacing_cal_days or int(horizon * 1.4)
    all_dates = sorted(panel["date"].unique())
    rebals = _rebalance_dates(all_dates, spacing_cal_days)

    prev_holds: set = set()
    strat_gross, strat_net, univ_ret, bench_ret, turnovers = [], [], [], [], []
    n_periods = 0

    for d in rebals:
        day = panel[panel["date"] == d]
        if len(day) < top_n:
            continue
        top = day.nlargest(top_n, "pred")
        holds = set(top["ticker"])

        gross = float(top["fwd_return"].mean())          # equal-weight top-N raw return
        universe = float(day["fwd_return"].mean())        # equal-weight ALL survivors
        bench = float(day["bench_fwd"].mean())            # blended market benchmark

        # Turnover cost: names entered + exited, one-way cost each side.
        entered = len(holds - prev_holds)
        exited = len(prev_holds - holds)
        turnover = (entered + exited) / top_n
        cost = turnover * (cost_bps / 10_000.0)
        net = gross - cost

        strat_gross.append(gross)
        strat_net.append(net)
        univ_ret.append(universe)
        bench_ret.append(bench)
        turnovers.append(turnover)
        prev_holds = holds
        n_periods += 1

    if n_periods < 2:
        return {"error": "not enough rebalance periods"}

    def _stats(rets):
        r = np.array(rets)
        total = float(np.prod(1 + r) - 1)
        years = n_periods / PERIODS_PER_YEAR
        cagr = float((1 + total) ** (1 / years) - 1) if total > -1 and years > 0 else float("nan")
        sharpe = float(r.mean() / r.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR)) if r.std(ddof=1) > 0 else 0.0
        return {"total_return_pct": round(total * 100, 2),
                "cagr_pct": round(cagr * 100, 2),
                "sharpe": round(sharpe, 2),
                "mean_period_pct": round(float(r.mean()) * 100, 3)}

    net_arr = np.array(strat_net)
    univ_arr = np.array(univ_ret)
    excess_vs_univ = net_arr - univ_arr  # survivorship-neutral alpha, net of cost
    t_univ = (excess_vs_univ.mean() / (excess_vs_univ.std(ddof=1) / np.sqrt(len(excess_vs_univ)))
              if excess_vs_univ.std(ddof=1) > 0 else 0.0)

    return {
        "periods": n_periods,
        "top_n": top_n,
        "cost_bps": cost_bps,
        "avg_turnover_pct": round(float(np.mean(turnovers)) * 100, 1),
        "strategy_gross": _stats(strat_gross),
        "strategy_net": _stats(strat_net),
        "equal_weight_universe": _stats(univ_ret),
        "market_benchmark": _stats(bench_ret),
        "net_alpha_vs_universe_pct_per_period": round(float(excess_vs_univ.mean()) * 100, 3),
        "net_alpha_vs_universe_annual_pct": round(float(excess_vs_univ.mean()) * PERIODS_PER_YEAR * 100, 2),
        "net_alpha_vs_universe_t_stat": round(float(t_univ), 2),
        "net_beats_universe": bool(excess_vs_univ.mean() > 0),
        "net_beats_market": bool(_stats(strat_net)["cagr_pct"] > _stats(bench_ret)["cagr_pct"]),
    }
