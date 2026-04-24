# Sterling — Claude Code Context

## What This Is
A Python financial agent for Canadian stock portfolio management. Generates BUY/HOLD/SELL/TRIM signals with calibrated confidence scores. Sends alerts via Telegram. Backtests on TSX data. CLI-driven, scheduled, no auto-trading.

## Architecture
```
sterling/
  config.py      — env loading, validation
  portfolio.py   — holdings CRUD (data/portfolio.json)
  data_feed.py   — yfinance + finnhub + pytrends, disk cache
  analyst.py     — 5-axis scoring: technical/fundamental/sentiment/macro/insider
  notifier.py    — Telegram bot, SQLite throttle (data/notifications.db)
  backtester.py  — walk-forward backtest, outputs to data/backtest_<ts>/
  scheduler.py   — APScheduler: 09:30, 12:30, 15:55 ET
  cli.py         — click CLI entry point
```

## Key Commands
```bash
# Activate venv (Windows)
.venv\Scripts\activate

# Install deps
pip install -r requirements.txt

# Install CLI in editable mode
pip install -e .

# Run full analysis
sterling analyze

# Add a holding
sterling add SHOP.TO 5 98.50

# Run backtest
sterling backtest --years 3 --capital 1000

# Start scheduler
sterling run

# One-shot run
sterling run --once

# Setup Telegram
python setup_telegram.py

# Run tests
pytest -v
```

## Environment Variables
See `.env.example`. Copy to `.env` and fill in:
- `FINNHUB_API_KEY` (required)
- `TELEGRAM_BOT_TOKEN` (required)
- `TELEGRAM_CHAT_ID` (required)
- `NEWSAPI_KEY` (optional)

## Data Files
- `data/portfolio.json` — holdings (gitignored)
- `data/notifications.db` — alert throttle state (gitignored)
- `data/cache/` — 24h fundamental cache
- `data/backtest_<timestamp>/` — backtest outputs

## Signal Thresholds
≥75 BUY | 60–74 ACCUMULATE | 40–59 HOLD | 25–39 TRIM | <25 SELL

## Constraints
- Wealthsimple has no public API — signals only, manual execution
- $1,000 CAD account, max 4 positions, max 2% risk per trade ($20)
- Skip tickers under $5 or <100k daily volume
- All alerts end with: "Signal, not advice. You decide."

## TSX Ticker Conventions
- TSX: `SHOP.TO`, `ENB.TO`, `BNS.TO`
- TSX-V: `ABX.V`
- USD cross-listings flagged with 1.5% FX drag warning

## Dependencies
See `requirements.txt`. Key: yfinance, finnhub-python, pytrends, ta, APScheduler, click, python-telegram-bot, python-dotenv, pandas, numpy, matplotlib, seaborn.
