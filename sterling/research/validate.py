"""Edge validation: does the composite score actually predict forward returns?

This is deliberately upstream of the trading strategy. Strategy P&L is confounded by
position sizing, stop-losses, slippage and the entry threshold (which we've seen is
overfit). The cleaner question is: **on each scan date, do higher-scored names go on to
outperform lower-scored names?** If yes, there is signal to trade. If not, no amount of
threshold tuning creates an edge — you're polishing noise.

We measure it three ways, all on the de-biased (point-in-time) engine:

  1. Information Coefficient (IC): Spearman rank correlation between score and the
     subsequent `horizon`-day return, pooled across all (ticker, date) observations.
     Rule of thumb: |IC| > ~0.03 is a weak-but-real signal; ~0 is noise; negative means
     the score is backwards.
  2. Score-bucket forward returns: split observations into quartiles by score; a real
     signal shows mean forward return rising monotonically from Q1 → Q4, and a positive
     top-minus-bottom (Q4−Q1) spread.
  3. Out-of-sample stability: recompute the IC in each of several non-overlapping time
     periods. A real edge keeps the same sign across periods; an artifact flips around.

We also compare the top-quartile basket's forward return against XIC.TO over the same
horizon — the only comparison that matters for "should I bother vs. buying the index."
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _fwd_return(closes: np.ndarray, pos: int, horizon: int) -> Optional[float]:
    """Return over `horizon` bars starting at integer position `pos`, or None."""
    if pos < 0 or pos + horizon >= len(closes):
        return None
    p0, p1 = closes[pos], closes[pos + horizon]
    if not (np.isfinite(p0) and np.isfinite(p1)) or p0 <= 0:
        return None
    return float(p1 / p0 - 1.0)


def build_edge_frame(bundle: dict, records: list, horizon: int = 20) -> pd.DataFrame:
    """Join score records with realized forward returns.

    `records` is the output of backtester.scan_score_distribution (each has
    date/ticker/post_score). Returns a DataFrame [date, ticker, score, fwd_return].
    """
    price_data = bundle["price_data"]
    # Precompute per-ticker date→position maps and close arrays.
    date_pos: dict = {}
    closes: dict = {}
    for tk, df in price_data.items():
        d = [x for x in df.index.date]
        closes[tk] = df["Close"].to_numpy()
        date_pos[tk] = {dd: i for i, dd in enumerate(d)}

    rows = []
    for r in records:
        tk = r["ticker"]
        if tk not in date_pos:
            continue
        d = date.fromisoformat(r["date"]) if isinstance(r["date"], str) else r["date"]
        pos = date_pos[tk].get(d)
        if pos is None:
            continue
        fwd = _fwd_return(closes[tk], pos, horizon)
        if fwd is None:
            continue
        rows.append({"date": d, "ticker": tk, "score": float(r["post_score"]), "fwd_return": fwd})

    return pd.DataFrame(rows)


def information_coefficient(df: pd.DataFrame) -> float:
    """Spearman rank correlation between score and forward return.

    Computed as Pearson correlation of the ranks (Spearman's definition) so it needs
    no scipy — pandas' built-in "spearman" method imports scipy, which isn't a dep here.
    """
    if len(df) < 10 or df["score"].nunique() < 2:
        return float("nan")
    r = df["score"].rank().corr(df["fwd_return"].rank())  # Pearson on ranks == Spearman
    return float(r)


def bucket_returns(df: pd.DataFrame, n: int = 4) -> pd.DataFrame:
    """Mean forward return per score quantile bucket (Q1 lowest … Qn highest)."""
    if len(df) < n * 5 or df["score"].nunique() < n:
        return pd.DataFrame()
    try:
        buckets = pd.qcut(df["score"].rank(method="first"), n, labels=[f"Q{i+1}" for i in range(n)])
    except ValueError:
        return pd.DataFrame()
    g = df.groupby(buckets, observed=True)["fwd_return"]
    return pd.DataFrame({
        "mean_fwd_return_pct": (g.mean() * 100).round(3),
        "count": g.count(),
        "avg_score": df.groupby(buckets, observed=True)["score"].mean().round(1),
    })


def walk_forward_ic(df: pd.DataFrame, n_periods: int = 4) -> pd.DataFrame:
    """IC computed within each of `n_periods` contiguous, non-overlapping date windows."""
    if df.empty:
        return pd.DataFrame()
    dfs = df.sort_values("date")
    dates = sorted(dfs["date"].unique())
    if len(dates) < n_periods * 2:
        n_periods = max(1, len(dates) // 2)
    edges = np.array_split(dates, n_periods)
    rows = []
    for i, chunk in enumerate(edges):
        if len(chunk) == 0:
            continue
        sub = dfs[dfs["date"].isin(set(chunk))]
        rows.append({
            "period": i + 1,
            "start": str(chunk[0]),
            "end": str(chunk[-1]),
            "obs": len(sub),
            "ic": round(information_coefficient(sub), 4),
        })
    return pd.DataFrame(rows)


def top_bucket_vs_benchmark(bundle: dict, df: pd.DataFrame, horizon: int = 20, top_q: float = 0.75) -> dict:
    """Compare the top-quartile basket's mean forward return to XIC.TO over the same
    horizon, averaged across scan dates. Positive spread ⇒ the score adds value over
    just holding the index."""
    xic = bundle.get("xic")
    if xic is None or xic.empty:
        return {}
    xic_dates = list(xic.index.date)
    xic_closes = xic["Close"].to_numpy()
    xic_pos = {d: i for i, d in enumerate(xic_dates)}

    spreads, top_rets, xic_rets = [], [], []
    for d, day in df.groupby("date"):
        if len(day) < 4:
            continue
        thresh = day["score"].quantile(top_q)
        top = day[day["score"] >= thresh]
        if top.empty:
            continue
        xp = xic_pos.get(d)
        xf = _fwd_return(xic_closes, xp, horizon) if xp is not None else None
        if xf is None:
            continue
        tr = float(top["fwd_return"].mean())
        top_rets.append(tr)
        xic_rets.append(xf)
        spreads.append(tr - xf)

    if not spreads:
        return {}
    spreads_arr = np.array(spreads)
    # Simple t-stat on the mean spread (H0: mean spread = 0).
    t_stat = (spreads_arr.mean() / (spreads_arr.std(ddof=1) / np.sqrt(len(spreads_arr)))
              if len(spreads_arr) > 1 and spreads_arr.std(ddof=1) > 0 else 0.0)
    return {
        "horizon_days": horizon,
        "scan_dates": len(spreads),
        "top_bucket_mean_fwd_pct": round(np.mean(top_rets) * 100, 3),
        "xic_mean_fwd_pct": round(np.mean(xic_rets) * 100, 3),
        "mean_spread_pct": round(spreads_arr.mean() * 100, 3),
        "spread_hit_rate_pct": round(float((spreads_arr > 0).mean()) * 100, 1),
        "t_stat": round(float(t_stat), 2),
    }


def validate_edge(bundle: dict, records: list, horizons=(10, 20, 60), n_periods: int = 4) -> dict:
    """Full edge report across multiple horizons. Returns a JSON-friendly dict."""
    report: dict = {"horizons": {}}
    for h in horizons:
        df = build_edge_frame(bundle, records, horizon=h)
        if df.empty:
            report["horizons"][h] = {"error": "no observations"}
            continue
        buckets = bucket_returns(df)
        wf = walk_forward_ic(df, n_periods)
        report["horizons"][h] = {
            "observations": len(df),
            "ic": round(information_coefficient(df), 4),
            "buckets": buckets.reset_index().rename(columns={"index": "bucket"}).to_dict("records")
            if not buckets.empty else [],
            "q4_minus_q1_pct": (round(buckets["mean_fwd_return_pct"].iloc[-1]
                                      - buckets["mean_fwd_return_pct"].iloc[0], 3)
                                if not buckets.empty else None),
            "walk_forward_ic": wf.to_dict("records") if not wf.empty else [],
            "ic_sign_stable": (bool((wf["ic"] > 0).all() or (wf["ic"] < 0).all())
                               if not wf.empty else None),
            "top_bucket_vs_xic": top_bucket_vs_benchmark(bundle, df, horizon=h),
        }
    return report
