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

## Backtester Signal Fidelity (Known Approximations)

The walk-forward engine uses a 5-axis composite signal with the following fidelity levels per axis:

| Axis        | Fidelity       | Notes |
|-------------|----------------|-------|
| technical   | fully historical | Price history slice used up to scan date — no look-ahead |
| fundamental | **snapshot proxy** | yfinance returns current-day values, not point-in-time historical. Introduces look-ahead bias in P/E, FCF yield, etc. Accept and document; fixing requires a paid point-in-time data vendor. |
| sentiment   | neutral 50.0   | No free API provides historical news sentiment compatible with daily walk-forward |
| macro       | fully historical | VIX (^VIX) and TSX Composite (^GSPTSE) fetched from yfinance per scan date |
| insider     | neutral 50.0   | Finnhub free tier does not provide dated historical insider transactions |

`apply_macro_overlay()` is now wired into the backtest loop and fires on historical VIX/TSX data.

Previous behaviour (pre-fix): the backtester used `_quick_signal()` — RSI + MACD + SMA only — and never called `apply_macro_overlay()`. This was the primary cause of the measured +5.08% CAGR vs +26.79% XIC.TO: the backtest was not testing Sterling, it was testing a 1990s technical model.

## Backtester Hold Parameters
- `hold_days` (default 20): maximum bars held before forced time-exit.
- `min_hold_days` (default 10): minimum bars before a TIME exit can fire. Stop-loss and target exits always override this floor. This prevents churn from weekly scan noise pushing the strategy into a 5-day hold → re-score → sell loop.

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
