# Sterling — Architecture Decisions

## Signal Mapping (analyst.py)
- ≥75 → BUY
- 60–74 → ACCUMULATE
- 40–59 → HOLD
- 25–39 → TRIM
- <25 → SELL

Rationale: thresholds chosen to bias toward HOLD under ambiguity. A $1k account cannot afford churn.

## Confidence Score Formula
```
confidence = (technical * 0.25) + (fundamental * 0.25) + (sentiment * 0.20) + (macro * 0.15) + (insider * 0.15)
```
Each axis scores 0–100. Weights reflect typical quant factor alpha rankings for Canadian small-cap liquid names. Technical and fundamental are co-equal leads; sentiment is real but noisier.

## Data Stack
- **yfinance**: prices, OHLCV, fundamentals (P/E, EPS, debt/equity, FCF yield) — free, supports `.TO` and `.V`
- **finnhub-python**: news sentiment, insider transactions, company news — free tier (60 req/min)
- **pytrends**: Google Trends slope as a retail sentiment proxy — free, no key required
- **ta (Technical Analysis Library)**: RSI, MACD, Bollinger Bands, SMA — avoids reimplementing indicators

## Caching Strategy
- Fundamental data cached 24h to disk (`data/cache/`) to stay under Finnhub free tier limits.
- Live quotes: no cache (need real-time during market hours).
- News sentiment: 4h cache.

## Notification Throttle
- Minimum 4h between alerts per ticker.
- Override if confidence delta > 15 points in either direction.
- Stored in SQLite (`data/notifications.db`) for persistence across restarts.

## Position Sizing (Hard Rules)
- Max 4 concurrent positions at ~$250 CAD each.
- Skip tickers under $5/share or <100k avg daily volume.
- Max 2% account risk per trade = $20 stop-distance budget.
- Flag any USD-listed cross-listing with 1.5% FX drag warning.

## Macro Overlay
- VIX > 25 OR TSX intraday < -2% OR oil crash (for energy names): downgrade all BUY signals by 15 points.
- Implemented as `apply_macro_overlay()` in analyst.py — single authoritative function.

## Backtest Assumptions
- Walk-forward on TSX 60 constituents, 3 years of data.
- Slippage: 0.1% per side (realistic for liquid TSX names).
- FX drag on USD trades: 1.5% (Wealthsimple spread).
- Benchmark: XIC.TO buy-and-hold with dividends reinvested.
- Reporting: CAGR, Sharpe, Sortino, max drawdown, win rate, profit factor.

## Wealthsimple Trade
- No public retail trading API exists. All signals are manual-execution only.
- The system generates the signal; the human places the order.

## Telegram vs. Other Notifiers
- Chosen because: free, no infra required, 60s setup via @BotFather, rich text formatting, reliable delivery.
- Email and SMS are fallback options the user can wire in later.

## Scheduler Times (Eastern)
- 09:30 — TSX open pulse
- 12:30 — midday re-score
- 15:55 — near-close signal (avoids last-minute noise from 15:59 algos)

## CLI Framework
- `click` chosen over `typer`: battle-tested, zero implicit magic, better for scripting.
