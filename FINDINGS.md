# Findings — Survivorship-Free ML Rebuild

The short version: **a fake edge was killed, a real-but-thin edge was found and honestly
stress-tested, and the honest conclusion is "index for real money."** This document is the
evidence trail. (For why the *original heuristic agent* was retired, see [DIAGNOSIS.md](DIAGNOSIS.md).)

---

## 1. The question

Can a model actually pick US stocks — measured the way it would be measured at a real quant
desk: **survivorship-free**, **out-of-sample**, **net of costs**, and judged on **risk-adjusted**
return, not raw return?

## 2. The data (the part most retail projects get wrong)

- **Source:** Sharadar bulk download — the entire `stocks` + `tickers` tables, including the
  **~38,000 delisted companies** the query API hides. Stored as Parquet, queried with **DuckDB**
  so a 1 GB+ price table never sits in RAM.
- **Universe:** mid-cap-and-up US common stocks, 1998–2026. **~50% of the names are dead
  companies** — bankruptcies and buyouts that a survivor-only dataset would silently erase.
- **No look-ahead:** features use only information available on each date; labels use a
  forward window and are **delisting-aware** — a company that dies books its real loss.
- Scale: **3.4M observations, 5,162 stocks, 2,557 of them delisted.**

## 3. The signal — real, and it survived 25 years

Gradient-boosted model (`HistGradientBoostingRegressor`) predicting cross-sectional excess
return, validated with **time-ordered walk-forward** (train on the past, test on unseen future,
embargo gap so labels can't leak).

| Quintile of model score | Q1 | Q2 | Q3 | Q4 | Q5 |
|---|---|---|---|---|---|
| Mean forward excess return | −1.05% | −0.69% | −0.36% | −0.005% | **+2.10%** |

- **Perfectly monotonic** Q1→Q5, and out-of-sample **IC positive in all 5 folds** — across the
  dot-com crash *and* 2008. The signal did not wash out; it got cleaner with more history.
- This is a genuine, survivorship-free ranking edge. It is also **small** (IC ≈ 0.01) — which is
  normal: single signals at real desks are individually weak too.

## 4. Turning picks into a portfolio — where it got hard

| Construction | Net CAGR | Sharpe | Worst drawdown | vs index |
|---|---|---|---|---|
| Concentrated (top 5%) | 13.6% | 0.45 | −59% | worse |
| Broad long (top 20%) | 20.4% | 0.65 | −53% | beats on **return** (t=2.06), not risk-adjusted |
| Naive long–short | 2–8% | 0.28–0.42 | up to −77% | **backfired** |
| **Risk-managed market-neutral** | — | **~0.4** | **~−24%** (halved) | **~+5%/yr steady alpha** |
| *Equal-weight universe (baseline)* | 15.7% | 0.72 | — | — |

Lessons, in order:
1. **Concentration is a trap** — a tiny top-N book gets destroyed by crashes.
2. **Broadening helps** — the top-20% beats the index on return with real significance (t=2.06),
   but at *higher* risk, so no risk-adjusted edge yet.
3. **Naive shorting is a trap** — the worst-ranked names can rocket up (buyouts, squeezes).
4. **Risk management is the unlock** — sector-neutral, market-neutral, volatility-weighted
   construction converts the signal into a genuine market-neutral ~5%/yr at ~0.4 Sharpe with the
   drawdown roughly halved. Thin, but real and honest.

## 5. Breadth — why "just add more signals" failed

The professional way to raise Sharpe is combining *uncorrelated* signals. We tried three add-ons —
short-term reversal, low-volatility, and value — and **all three lost money in 2010–2026.** That
is not a bug: those factors genuinely had a terrible decade. The uncomfortable implication:
**our edge is momentum-flavored, and it worked partly because this specific era favored momentum.**
Real breadth (value + *quality* from fundamentals) is blocked by data cost, and history suggests
it might not have helped much in this exact window anyway.

## 6. Honest conclusion

- **We have one real, small, survivorship-free, out-of-sample market-neutral alpha (~0.4 Sharpe).**
  Most attempts at this never produce a clean positive result; this one did.
- It is **not deployable retail alpha** — the edge is thin, regime-dependent, and would be eroded
  by costs and capacity at any serious scale, with none of a real fund's diversification across
  hundreds of alphas.
- **For real capital, a broad low-cost index is the honest default** — a conclusion *earned*
  through testing, not assumed.
- The lasting asset is the **method and the engine**: survivorship-free data, point-in-time
  features, walk-forward validation, risk-managed construction, and the discipline to refuse to
  deploy on a result that doesn't survive.

## Reproduce
```bash
python -m sterling.research verify      # data access + delisted coverage
python -m sterling.research parquet     # bulk CSV -> Parquet (one-time)
python -m sterling.research all         # features + walk-forward + portfolio sim
pytest -q                               # full test suite
```
