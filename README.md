# Sterling — Equity Research Pipeline

An evidence-based pipeline for testing whether a model can actually pick stocks — on
**survivorship-free** data, with **out-of-sample** validation and **net-of-cost** portfolio
simulation. Built to falsify its own edge before believing it, not to produce confident
signals.

> **Honest status.** The original heuristic BUY/HOLD/SELL agent was retired — it had no
> predictive edge (see [DIAGNOSIS.md](DIAGNOSIS.md)). The rebuilt gradient-boosted model,
> tested **survivorship-free over 25 years (1998–2026)**, *appeared* to have a ~5%/yr
> market-neutral edge — but a **tradeability stress test killed it**: that edge lived almost
> entirely in **untradeable penny stocks** the model expected to bounce. Restricted to stocks
> you could actually buy (≥$5, real volume), the edge falls to **≈0 (Sharpe −0.05, t=−0.14)**.
> **No deployable edge — for real capital, a broad low-cost index is the answer.** The value
> here is the *method*: a pipeline rigorous enough to prove its own edge was fake. Full story
> in [FINDINGS.md](FINDINGS.md).

## What the honest testing found

| Stage | Finding |
|---|---|
| Original heuristic agent | **No edge** — buy/hold/sell calls were ~a coin flip and lost to the index. |
| Backtest de-biased | Removed a look-ahead leak (point-in-time fundamentals) and a NaN-bar bug; paused the live loop. |
| Survivorship-free signal (25y) | **Real ranking skill**: monotonic score quintiles, out-of-sample IC positive in **all 5** time folds. |
| Naive portfolio (top-N) | Concentration got crushed by crashes; a broad top-20% beats the index on *return* (t=2.06) but **not** risk-adjusted. |
| Risk-managed market-neutral | *Appeared* to give ~+5%/yr at ~0.4 Sharpe (t=2.06). |
| **Tradeability stress test** | **The edge was fake** — it lived in untradeable penny stocks. Filter to real, liquid names → alpha falls to **≈0 (Sharpe −0.05, t=−0.14)**. |
| Adding signals (breadth) | Reversal / low-vol / value all **lost** in 2010–26 → no cheap way to add a real second edge. |

**The point of the project is the method, not the strategy:** find a fake edge and kill it,
find a real one and stress-test it until it's honest, and refuse to deploy on a result that
doesn't survive. Full story in [FINDINGS.md](FINDINGS.md) (and [DIAGNOSIS.md](DIAGNOSIS.md) for the original agent).

---

## Requirements
- Python 3.11+
- A **Sharadar** subscription (sharadar.com) with bulk-download access — for
  survivorship-free US equities (prices including delisted companies).
- `NASDAQ_API_KEY` in `.env` (never hard-code or share it).

## Install
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
pip install -e .
```

## Setup
```bash
echo "NASDAQ_API_KEY=your_sharadar_key" >> .env
python check_api_key.py         # confirm the key responds
```

## Usage
```bash
python -m sterling.research verify      # API key + universe/delisted coverage
python -m sterling.research parquet     # convert bulk CSVs -> Parquet (one-time)
python -m sterling.research features    # survivorship-free feature + label matrix
python -m sterling.research analyze     # walk-forward validation + portfolio sim
python -m sterling.research all         # features + analyze
pytest -v
```
(The Sharadar bulk `stocks`/`tickers` files go in `data/sharadar_bulk/`; download them from
your account, then `parquet` converts them for fast DuckDB queries.)

---

## How it works
1. **Data** — Sharadar bulk download (survivorship-free: ~38k delisted names included),
   stored as Parquet and queried with **DuckDB** so the 1 GB+ price table never has to sit
   in RAM.
2. **Universe** — mid-cap-and-up US common stocks that traded in the window, **including
   the ones that later died** (their series end at delisting).
3. **Features** — point-in-time price/technical + macro (and optional point-in-time
   fundamentals), sampled every 5 days. No look-ahead.
4. **Labels** — forward return over ~1 quarter, **delisting-aware**: a company that dies
   books its real loss instead of vanishing (the classic survivorship trap).
5. **Model** — `HistGradientBoostingRegressor` predicting cross-sectional excess return.
6. **Validation** — time-ordered **walk-forward** with an embargo gap (information
   coefficient, quantile buckets), then a **net-of-cost** top-N portfolio sim compared to
   the **equal-weight universe of the same names** (cancels survivorship bias).

## Design principles (learned the hard way — see [DIAGNOSIS.md](DIAGNOSIS.md))
- No look-ahead. Survivorship-free. Validate out-of-sample before believing anything.
- A green test suite proves the code is correct, **not** that the strategy makes money.
- If the edge doesn't survive honest testing, say so and don't deploy.

## Layout
```
sterling/research/   config, sharadar, store (DuckDB/Parquet), survivorship, features,
                     fundamentals, labels, model, portfolio_sim, validate, pipeline, __main__
docs / *.md          DIAGNOSIS.md · DECISIONS.md · SURVIVORSHIP_REBUILD_SCOPE.md
tests/               pipeline + logic tests
check_api_key.py     standalone key health check
```

---
*Research, not advice.*
