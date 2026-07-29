# Sterling — Edge Validation: Null Result

**Date:** 2026-07-27
**Conclusion:** Sterling's composite score has **no demonstrable, deployable edge** at the
horizon it trades. Scheduled paper trading has been paused and the recommendation for the
actual $1,000 is to hold the index (XIC.TO). Sterling is retained as a research/learning
project, not a live signal.

This document records how we reached that conclusion so it isn't re-litigated by vibes.

---

## What we fixed before judging (so the judgment is fair)

The original backtest (+5.08% CAGR vs XIC +26.79%, Sharpe 0.147) was not testing the real
strategy. Two structural problems were removed first so the model was evaluated honestly:

1. **Look-ahead bias in fundamentals.** The backtester fed *today's* P/E, revenue growth,
   ROE, etc. into every historical scan date (fundamentals are 40% of the score). Replaced
   with point-in-time reconstruction from dated financial statements + a filing lag
   (`sterling/hist_fundamentals.py`). Point-in-time coverage on the re-run was **100%**.
2. **NaN-bar bug.** yfinance's current-session NaN close was silently poisoning CAGR and the
   benchmark into `NaN`. Now dropped on load, with a regression test.

Survivorship bias remains (universe = *current* TSX 60 constituents; point-in-time
membership isn't available on free data). It biases returns **upward**, so the honest edge
is if anything *weaker* than the numbers below.

## De-biased backtest (for reference)

Best surviving config (thr 57.3 / hold 10, ~2.6y through 2026-07): **18.1% CAGR, Sharpe 1.33,
−6.5% max DD** — but **still below XIC.TO (24.8% CAGR)**, and fragile: nudging the entry
threshold from 57.3 → 65 collapses it to ~0% CAGR with a negative Sharpe. That fragility was
the first sign the "edge" was a tuned artifact, not signal.

## The decisive test: does the score predict forward returns?

Strategy P&L is confounded by stops, slippage, and the (overfit) threshold. So we tested the
**signal itself**: score every name every 5 trading days (3,451 observations, de-biased
engine) and measure whether a higher score predicts a higher forward return — via the
information coefficient (Spearman rank corr), score-quartile forward returns, and stability
across four non-overlapping time periods. Code: `sterling/validate.py`.

| Horizon | IC | Q1 → Q4 mean forward return | Q4−Q1 | Top-quartile vs XIC | Sign stable? |
|---|---|---|---|---|---|
| **10 days** | −0.021 | 0.98 · 1.04 · 0.88 · **0.83%** | −0.16% | −0.12% (t=−0.9) | No (−,−,+,+) |
| **20 days** | −0.028 | 1.92 · 2.09 · 1.88 · **1.50%** | −0.43% | −0.04% (t=−0.2) | No |
| **60 days** | +0.054 | 4.84 · 5.87 · 6.03 · **6.56%** | +1.73% | +1.16% (t=3.05) | No (3 of 4 +) |

**Reading:**
- **At 10–20 days — the horizon Sterling actually trades (min-hold 10, hold 20) — there is no
  edge, and it is slightly inverted.** The highest-scored quartile delivers the *lowest*
  forward return; IC is negative; the top basket underperforms XIC. The strategy harvests at
  exactly the horizon where its signal is worthless. This is why no threshold ever beat the
  index — it was never a threshold problem.
- **A weak signal exists only at ~60 days** (monotonic buckets, IC +0.05, top basket beats XIC
  by ~1.16%/60d, t=3.05). But IC of 0.05 is faint, one of four sub-periods went negative, and
  survivorship bias inflates even this. Regime-dependent, not robust.

## Decision

Calibrating scores or thresholds at the 10–20 day horizon would be **fitting noise**. The only
direction the data supported was lengthening the hold to ~60 days, but a faint, regime-dependent,
survivorship-inflated quarterly signal is not a basis for trading a $1,000 account against a
low-cost index that returned ~25% CAGR over the same window.

**Actions taken:**
- Scheduled paper trading **paused** (`.github/workflows/sterling.yml` — `schedule:` block
  commented out; `workflow_dispatch` retained for manual research runs).
- This null result documented here rather than buried.

**Recommendation for real capital:** buy and hold XIC.TO. Revisit Sterling only with (a) a
point-in-time-clean *and* survivorship-free universe, and (b) a signal that shows a stable,
positive IC across periods at the horizon it intends to trade — neither of which holds today.

## Reproduce

```bash
# de-biased backtest → data/debiased_stage1.json
# edge report        → data/edge_report.json  (validate.validate_edge)
pytest -v            # 253 tests, incl. test_hist_fundamentals.py, test_validate.py
```
