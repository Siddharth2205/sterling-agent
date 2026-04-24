# Sterling — Build Progress

## Deliverables

- [x] Bootstrap: git init, venv, pip
- [x] Project skeleton (dirs + stub files)
- [x] `sterling/config.py` — env loading + validation
- [x] `sterling/portfolio.py` — holdings CRUD
- [x] `sterling/data_feed.py` — yfinance + finnhub + pytrends
- [x] `sterling/analyst.py` — 5-axis scoring engine
- [x] `sterling/notifier.py` — Telegram alerts + throttle
- [x] `setup_telegram.py` — interactive Telegram setup
- [x] `sterling/backtester.py` — walk-forward backtest
- [x] `sterling/scheduler.py` — APScheduler jobs
- [x] `sterling/cli.py` — click CLI + pyproject.toml
- [x] `tests/` — 84/84 tests green
- [x] `README.md` — with real backtest numbers

## Backtest Summary (2023-12-11 → 2026-04-23)
- Sterling CAGR: +5.08% | XIC.TO CAGR: +26.79%
- Sharpe: 0.147 | Sortino: 0.194 | Max DD: -20.73%
- Win rate: 51.1% | Profit factor: 1.21 | Trades: 135
- **Verdict: Underperforms benchmark. See README limitations section.**

## Last Updated
2026-04-24
