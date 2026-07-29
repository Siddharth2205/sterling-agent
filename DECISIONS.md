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
| fundamental | **point-in-time** | Reconstructed from dated financial statements (annual backbone + quarterly TTM refinement) with a filing lag (annual 90d, quarterly 50d). A scan on date D only sees numbers public by D. Missing history → neutral 50, never today's snapshot. See `sterling/hist_fundamentals.py`. Coverage reported in stats as `fundamental_pit_coverage_pct`. |
| sentiment   | neutral 50.0   | No free API provides historical news sentiment compatible with daily walk-forward |
| macro       | fully historical | VIX (^VIX) and TSX Composite (^GSPTSE) fetched from yfinance per scan date |
| insider     | neutral 50.0   | Finnhub free tier does not provide dated historical insider transactions |

`apply_macro_overlay()` is now wired into the backtest loop and fires on historical VIX/TSX data.

### Fidelity history
- **Original**: the backtester used `_quick_signal()` — RSI + MACD + SMA only — and never called `apply_macro_overlay()`. This was the primary cause of the measured +5.08% CAGR vs +26.79% XIC.TO: the backtest was not testing Sterling, it was testing a 1990s technical model.
- **Stage 1 (de-bias)**: the fundamental axis previously used a **current-snapshot proxy** — today's P/E, revenue growth, ROE applied to every historical scan date. Since fundamentals are 40% of the score, this was look-ahead bias that flattered every number. Replaced with point-in-time reconstruction (above). Caveat: yfinance statements are *restated* figures, not true as-originally-reported vintage data, so a small residual bias remains; the filing lag is conservative to compensate.

### Survivorship bias (known, unfixed)
`TSX60_TICKERS` is the **current** index membership. Names dropped or delisted over the backtest window never appear, so the universe is pre-filtered to survivors — an upward bias on returns. Point-in-time index constituent history is not available on free data, so this remains a documented limitation, surfaced in stats as `survivorship_bias: "current-constituents (upward bias)"`. Any headline backtest number should be read as an **optimistic** estimate for this reason.

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

## Scan Universe Composition

The `sterling scan` command operates on a ~220-ticker universe composed of four segments:

| Segment               | Count | Source                    | Notes                                    |
|----------------------|-------|---------------------------|------------------------------------------|
| TSX 60 constituents  | ~60   | `scan_universe.py`        | Major Canadian large-caps across sectors  |
| TSX mid-caps         | ~55   | `scan_universe.py`        | Liquid mid-cap names not in TSX 60       |
| CIBC CDR catalog     | ~85   | `cdr_mapping.py` (reused) | US mega/large-caps traded on NEO (.NE)   |
| TSX-listed ETFs      | ~20   | `scan_universe.py`        | Broad market, sector, and thematic ETFs  |

Deduplication runs at load time (first occurrence wins). CDRs redirect fundamentals and news
sentiment to the US underlying via `cdr_mapping.get_underlying()` as the existing `analyst.analyze()`
already does.

Rate-limit safety: Finnhub free tier allows 60 API calls/min. Each ticker may trigger up to 2
Finnhub calls (news sentiment + insider). The scanner paces at 2 s between tickers (~30 tickers/min
= ~60 Finnhub calls/min worst-case). The existing 24 h fundamentals disk cache and 4 h sentiment
cache reduce actual API calls on subsequent runs. A full cold scan takes ~7-8 minutes.

The scan command is read-only: it never writes to `paper_trades.csv`, the notification throttle DB,
or any part of the `analyze` / `backtest` pipeline. Digest notifications use a separate
`send_scan_digest()` function that bypasses the throttle entirely.

## CLI Framework
- `click` chosen over `typer`: battle-tested, zero implicit magic, better for scripting.
