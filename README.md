# Sterling — Canadian Portfolio Analysis Agent

A Python financial agent that scores your TSX/TSX-V holdings across five axes, generates BUY/HOLD/SELL/TRIM signals with calibrated confidence scores, and pushes alerts to your phone via Telegram. CLI-driven. No auto-trading. Built for a \$1,000 CAD account managed through Wealthsimple Trade.

> **⚠️ Status (2026-07-27): not a live trading signal.** After removing look-ahead bias from
> the backtester, edge validation found the composite score has **no predictive edge at the
> 10–20 day horizon the strategy trades** (information coefficient ≈ 0). Scheduled paper trading
> is paused; for real capital, hold the index (XIC.TO). Full write-up: [DIAGNOSIS.md](DIAGNOSIS.md).
> Sterling is retained as a research/learning project.

---

## Requirements

- Python 3.11+
- A [Finnhub](https://finnhub.io/register) API key (free tier)
- A Telegram bot (5-minute setup via @BotFather)

---

## Install

```bash
git clone <repo>
cd sterling-agent

python3 -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
pip install -e .
```

---

## Environment Setup

```bash
copy .env.example .env          # Windows
# cp .env.example .env          # macOS/Linux
```

Edit `.env`:

```
FINNHUB_API_KEY=your_key_here
TELEGRAM_BOT_TOKEN=from_botfather
TELEGRAM_CHAT_ID=your_chat_id
```

### Telegram Setup (60 seconds)

```bash
python setup_telegram.py
```

The wizard walks you through @BotFather, captures your token, detects your chat ID, and writes everything to `.env`.

---

## Usage

```bash
# Add holdings to your portfolio
sterling add SHOP.TO 5 98.50
sterling add ENB.TO 10 47.20
sterling add BNS.TO 8 62.00
sterling add CNR.TO 3 175.00

# Add a watchlist
sterling watch T.TO RY.TO ATD.TO

# Show portfolio with live prices and P&L
sterling portfolio

# Run full analysis (no alerts)
sterling analyze

# Run analysis and fire Telegram alerts for actionable signals
sterling analyze --notify

# Analyze specific tickers only
sterling analyze -t SHOP.TO -t ENB.TO

# Start the scheduled worker (09:30 / 12:30 / 15:55 ET, weekdays)
sterling run

# Single-pass run and exit
sterling run --once

# Run backtest
sterling backtest --years 3 --capital 1000

# Run test suite
sterling test
```

---

## How Signals Work

Each ticker is scored across five axes, each 0–100:

| Axis | Weight | What it measures |
|---|---|---|
| Technical | 25% | RSI(14), MACD, SMA 50/200 cross, Bollinger position, volume |
| Fundamental | 25% | P/E vs norm, FCF yield, debt/equity, revenue growth, ROE |
| Sentiment | 20% | Finnhub news keyword score, Google Trends slope |
| Macro | 15% | VIX, TSX breadth, USD/CAD |
| Insider | 15% | Insider buy/sell ratio, net shares transacted |

**Confidence = weighted sum of all five axes**

| Confidence | Action |
|---|---|
| ≥ 75 | BUY |
| 60–74 | ACCUMULATE |
| 40–59 | HOLD |
| 25–39 | TRIM |
| < 25 | SELL |

**Macro overlay:** If VIX > 25 or TSX drops > 2% intraday, all BUY signals are automatically downgraded by 15 points.

**All alerts end with:** *Signal, not advice. You decide.*

---

## Position Sizing Rules (\$1,000 CAD Account)

These are hard-coded constraints, not guidelines:

- Maximum 4 concurrent positions (~\$250 each)
- Skip any name under \$5/share or under 100k average daily volume
- Maximum 2% account risk per trade = \$20 stop-distance budget
- USD-listed cross-listings flagged with a 1.5% FX drag warning

---

## Backtest Results

**Run:** 2023-12-11 → 2026-04-23 (2.37 years, 29 TSX 60 constituents, seed=42)

| Metric | Sterling Strategy | XIC.TO (B&H + FX drag) |
|---|---|---|
| Total return | +12.45% | +75.32% |
| CAGR | +5.08% | +26.79% |
| Sharpe ratio | 0.147 | — |
| Sortino ratio | 0.194 | — |
| Max drawdown | -20.73% | — |
| Win rate | 51.1% | — |
| Profit factor | 1.21 | — |
| Avg R:R | 1.16x | — |
| Total trades | 135 | — |

**Verdict: The strategy underperforms XIC.TO buy-and-hold by a wide margin over this period.**

This is honest and expected. The backtest covers Dec 2023 – Apr 2026, a period where the TSX delivered exceptional returns. A simple technical signal screener (RSI + MACD + SMA cross) cannot reliably outperform a diversified index in a strong bull market after costs. The 51.1% win rate and 1.21 profit factor indicate the system is marginally profitable on individual trades but insufficient alpha to beat a rising tide.

**What this system is for:** Filtering a universe of TSX names to surface potential entries with a defined stop-loss, not to beat a benchmark. It helps you avoid buying everything and structures risk on each position. Whether it generates meaningful alpha at \$1k scale is unclear — the Sharpe of 0.147 suggests it does not on a risk-adjusted basis.

---

## Limitations

**Free-tier API rate limits**
- Finnhub free tier: 60 requests/minute. Analyzing >15 tickers back-to-back may trigger throttling. Fundamentals and news are cached (24h and 4h respectively) to minimize this.
- Google Trends: no rate limit key, but aggressive polling will trigger blocks. Trending data cached 4h.

**No auto-trading**
Wealthsimple Trade has no public retail API. All signals require manual execution. The system tells you what to consider; you place the order.

**Slippage and spread**
Backtest assumes 0.1% slippage per side on liquid TSX names. Real-world execution on a \$250 position may be worse for names with wide bid/ask spreads.

**FX cost**
Any USD-listed cross-listing carries a real 1.5% Wealthsimple FX spread (buy + sell = 3% round-trip). The system flags these.

**Small account math**
At \$250 per position, odd-lot friction is real. A \$50 stock means 5 shares — one bad tick and your stop is hit. A \$200 stock means 1 share. Position sizing at this scale is a problem of rounding, not optimization.

**Sentiment is approximate**
The free Finnhub tier doesn't return pre-computed sentiment scores. The keyword-counting approach in `data_feed.py` is a serviceable proxy but not a trained NLP model.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `FINNHUB_API_KEY` missing error | Copy `.env.example` → `.env`, add your key |
| Telegram not sending | Run `python setup_telegram.py` to re-verify token and chat ID |
| `yfinance` returns no data | Ticker may be delisted or suffix wrong (use `.TO` not `-CA`) |
| Rate limit from Finnhub | Wait 60s; analysis uses disk cache, so re-running usually hits cache |
| Scheduler not firing | Confirm your timezone is correct; check `data/sterling.log` |

---

## Architecture

```
sterling/
  config.py      — .env loading and validation
  portfolio.py   — holdings CRUD (data/portfolio.json)
  data_feed.py   — yfinance + finnhub + pytrends, 24h disk cache
  analyst.py     — 5-axis scoring, macro overlay, position sizing rules
  notifier.py    — Telegram bot, SQLite throttle (data/notifications.db)
  backtester.py  — walk-forward backtest, equity curve + heatmap output
  scheduler.py   — APScheduler: 09:30, 12:30, 15:55 ET
  cli.py         — click CLI (sterling add / analyze / backtest / run)
```

---

*Signal, not advice. You decide.*
