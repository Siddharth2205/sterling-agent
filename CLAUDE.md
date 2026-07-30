# Sterling — Claude Code Context

> **New here? Read [HANDOFF.md](HANDOFF.md) first** — current state, the honest verdict
> (no tradeable edge; the wall is data/market, not code), what's live (daily cloud search +
> dashboard + Telegram), and the non-negotiable discipline. Then [FINDINGS.md](FINDINGS.md).

## What This Is (current direction)
An **evidence-based equity research pipeline**. Given a broad, **survivorship-free** US
universe (Sharadar bulk data, including delisted companies), it engineers point-in-time
features, trains a gradient-boosted model to predict forward returns, and **validates
out-of-sample** (walk-forward + net-of-cost portfolio sim) before believing any edge.

The original heuristic "5-axis" BUY/HOLD/SELL agent was **retired** — diagnostics proved
it had no edge at its trading horizon (see `DIAGNOSIS.md`). It still exists on the `master`
branch for reference; branch `research/survivorship-ml` keeps only the research work.

## Architecture
```
sterling/research/
  config.py        — central paths + params (no hard-coded paths)
  sharadar.py      — Sharadar REST + BULK download connector (survivorship-free source)
  store.py         — DuckDB + Parquet data layer (query big data without loading it all)
  survivorship.py  — universe selection, delisting map, delisting-aware labels
  universe.py      — legacy free-data universe list (yfinance path)
  dataset.py       — yfinance loader for benchmarks/macro (SPY, ^VIX)
  features.py      — point-in-time features (price/technical + macro + fundamentals)
  fundamentals.py  — pure point-in-time fundamental selection (as-of, no look-ahead)
  labels.py        — forward returns incl. delisting-aware rule (dead names book real loss)
  model.py         — HistGradientBoosting + walk-forward gate (ship/no-ship criteria)
  portfolio_sim.py — net-of-cost top-N sim vs survivorship-neutral equal-weight baseline
  validate.py      — information-coefficient / bucket edge diagnostics
  pipeline.py      — orchestration (build_features -> analyze)
  __main__.py      — CLI
```

## Key Commands
```bash
.venv\Scripts\activate

python check_api_key.py                 # standalone Sharadar key health check

python -m sterling.research verify      # check API key responds
python -m sterling.research parquet     # convert bulk CSVs -> Parquet (one-time)
python -m sterling.research features    # build survivorship-free feature matrix
python -m sterling.research analyze     # walk-forward + portfolio sim
python -m sterling.research all         # features + analyze

pytest -v
```

## Data
- Requires `NASDAQ_API_KEY` in `.env` (Sharadar). Never hard-code or print keys.
- Bulk files in `data/sharadar_bulk/` (**gitignored**): `stocks_10Y.csv` -> `stocks.parquet`
  (~400 MB), `tickers_full.csv` -> `tickers.parquet`.
- Derived artifacts (`data/*.pkl`, `data/*.json`) are gitignored.

## Data-layer principle
Do NOT load multi-GB CSVs into pandas. Convert to Parquet once (`store.ensure_parquet`)
and query slices with DuckDB (`store.load_prices`, `store.load_tickers`).

## Method principles (hard-won — see DIAGNOSIS.md)
- No look-ahead: features/fundamentals are point-in-time; labels use a filing/return lag.
- Survivorship-free: the universe includes delisted names; a company that dies books its
  real loss, never silently dropped.
- Validate out-of-sample before believing anything: walk-forward IC, then net-of-cost
  top-N vs the **equal-weight universe of the same names** (cancels survivorship bias).
- "Passing tests" != "makes money": a green suite proves the code does what we said, not
  that the strategy has edge.

## Status
Survivorship-free 10-year run: concentrated top-15 beats the universe (~+17%/yr net) but
**not yet statistically significant (t≈1.5)** and it dilutes fast. Next: full-history run
for statistical power. Not deployable; for real capital an index/equal-weight is the
honest default.

## Constraints
- No auto-trading. Research/signals only.
- Keep secrets in `.env`; the connector redacts keys from any error/log.
