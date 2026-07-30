# HANDOFF — read this first

You're continuing **Sterling**. This note gets you productive in ~2 minutes. Deeper detail:
[FINDINGS.md](FINDINGS.md) (the full journey) · [DECISIONS.md](DECISIONS.md) · [README.md](README.md).

## What Sterling is now
An evidence-based US-equity research engine. It began as a heuristic BUY/HOLD/SELL agent
(**retired — no edge**), was rebuilt as a **survivorship-free ML pipeline**, and now runs an
**autonomous strategy search** daily in the cloud until **2026-08-28**, publishing to a live
dashboard and Telegram.

## The honest verdict — don't re-discover this the hard way
- Heuristic agent: no predictive edge at its horizon. Retired.
- ML model on survivorship-free data: a **real but tiny** ranking signal (information coefficient ≈ 0.01).
- The apparent **+5%/yr market-neutral alpha was an artifact of untradeable penny stocks.** With a
  point-in-time liquidity filter (≥ $5, ≥ $1M/day), it collapses to **≈ 0 (Sharpe −0.05, t = −0.14)**.
- **For real capital: a broad index.** A smarter model does **not** fix this — the wall is the
  *data and the market*, not the code. See [FINDINGS.md](FINDINGS.md).

## Non-negotiables (the discipline that *is* the project)
1. **No look-ahead** — point-in-time features; fundamentals gated by a filing lag.
2. **Survivorship-free** — include delisted names; a company that dies books its real loss.
3. **Tradeability filter always on** — no penny/illiquid stocks (this is what killed the fake edge).
4. **Judge on a locked hold-out** the search never tuned on. **Report the truth, even "no edge."**
5. A green test suite proves the code is correct, **not** that the strategy makes money.

## Data
- **Sharadar** (sharadar.com direct REST API), **US-only, survivorship-free**, accessed via **bulk
  download** (not per-ticker — that path hides delisted names). Key in `.env` as `NASDAQ_API_KEY`.
- Current plan = prices + tickers + metrics; **no fundamentals** — that's the **$19/$29 add-on**, the
  main unexplored lever (point-in-time value + quality across the full universe).
- Bulk CSVs live in `data/sharadar_bulk/` → converted to **Parquet** and queried with **DuckDB**
  (`store.py`). Never load the multi-GB CSV into pandas.

## Architecture — `sterling/research/`
`config` · `store` (DuckDB/Parquet) · `sharadar` (connector) · `survivorship` · `features` ·
`fundamentals` · `labels` · `tradeable` (liquidity filter) · `model` (walk-forward + ship gate) ·
`construction` (risk-managed market-neutral book) · `portfolio_sim` · `signals` · `validate` ·
`experiment` (autonomous search) · `live` (paper book + Telegram) · `dashboard` · `pipeline` ·
`__main__` (CLI).

## Run it
```bash
python -m sterling.research verify      # data access + delisted coverage
python -m sterling.research parquet     # one-time bulk CSV -> Parquet
python -m sterling.research all         # features + walk-forward + portfolio sim
python -m sterling.research experiment -n 15   # one autonomous search cycle
python -m sterling.research book        # generate + log today's paper book (+ Telegram)
python -m sterling.research dashboard   # rebuild docs/dashboard_data.json
pytest -q                               # 48 tests
```
First run builds the ~3.4M-row matrix from the Sharadar bulk (~40 min) then caches it
(`data/experiment_matrix.pkl`); later runs are fast.

## What's live / automated
- **GitHub Actions** `search.yml`: daily **07:00 UTC until Aug 28** — runs a search cycle, refreshes
  the paper book, rebuilds the dashboard, commits results back (robust rebase-and-retry push), and
  Telegrams progress; a final report fires on Aug 28.
- **Dashboard**: GitHub Pages from `master /docs` → **https://siddharth2205.github.io/sterling-agent/**
- **Secrets** (already set): `NASDAQ_API_KEY`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`.
- The leaderboard (`data/experiment_leaderboard.csv`) is committed each run and **accumulates** — it
  grows daily, it does not reset.

## Open levers (priority order) — with honest expectations
1. **Fundamentals add-on** ($19/$29) → point-in-time **value + quality** signals across the
   survivorship-free universe. The *right* way to add breadth (naive price-only add-ons — reversal,
   low-vol — all lost in 2010–26).
2. **Global markets** (survivorship-free global data) → more independent bets.
3. **Better portfolio construction / a proper risk model.**

None is guaranteed to beat the index. They may strengthen a thin signal; the constraint is data and
market efficiency, not the assistant. Build honestly; if it doesn't survive the hold-out, say so.
