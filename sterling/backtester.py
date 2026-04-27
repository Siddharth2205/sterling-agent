"""Walk-forward backtest on TSX 60 constituents vs XIC.TO buy-and-hold.

Signal fidelity notes (see also DECISIONS.md):
  - Technical axis: fully historical (uses price history slice at each scan date).
  - Fundamental axis: current-snapshot proxy — yfinance returns present-day values,
    NOT point-in-time historical. This introduces look-ahead bias in the fundamental
    score. Accept the limitation; document it loudly.
  - Sentiment axis: fixed at 50.0 (neutral). No free historical sentiment API exists.
  - Macro axis: fully historical — VIX (^VIX) and TSX (^GSPTSE) fetched from yfinance.
    apply_macro_overlay() is now wired into every weekly scan.
  - Insider axis: fixed at 50.0 (neutral). Finnhub free tier does not provide dated
    historical insider transactions compatible with walk-forward simulation.
"""

import csv
import json
import logging
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf
import ta

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).parent.parent / "data"

# TSX 60 representative constituents (liquid, covers sectors)
TSX60_TICKERS = [
    "SHOP.TO", "RY.TO", "TD.TO", "BNS.TO", "BMO.TO", "CM.TO", "NA.TO",
    "ENB.TO", "CNQ.TO", "SU.TO", "TRP.TO", "PPL.TO",
    "CNR.TO", "CP.TO",
    "BCE.TO", "T.TO",
    "MFC.TO", "SLF.TO", "GWO.TO",
    "ATD.TO", "L.TO", "MRU.TO",
    "WCN.TO", "WSP.TO", "CAE.TO",
    "ABX.TO", "AEM.TO",
    "BAM.TO", "BIP-UN.TO",
]

# Tickers whose base symbol is in this set are treated as energy for the oil-crash overlay.
# Includes CDR underlyings (XOM, CVX) so NE-listed energy CDRs are also downgraded.
_ENERGY_BASES = {"ENB", "SU", "CVE", "CNQ", "TRP", "PPL", "WCP", "ARX", "PEY", "TVE", "XOM", "CVX"}


# ── Data fetchers ─────────────────────────────────────────────────────────────

def _get_history_safe(ticker: str, start: str, end: str) -> Optional[pd.DataFrame]:
    try:
        t = yf.Ticker(ticker)
        df = t.history(start=start, end=end, auto_adjust=True)
        if df.empty or len(df) < 50:
            return None
        return df
    except Exception as e:
        logger.warning(f"History fetch failed for {ticker}: {e}")
        return None


def _get_index_history(ticker: str, start: str, end: str) -> Optional[pd.DataFrame]:
    """Like _get_history_safe but without minimum-bar requirement (for macro indices)."""
    try:
        t = yf.Ticker(ticker)
        df = t.history(start=start, end=end, auto_adjust=True)
        return df if not df.empty else None
    except Exception as e:
        logger.warning(f"Index history fetch failed for {ticker}: {e}")
        return None


def _fetch_historical_macro(start_str: str, end_str: str) -> dict:
    """
    Build a {date: macro_dict} map using historical VIX and TSX Composite data.
    risk_off flag = VIX > 25 OR TSX daily change < -2%.
    oil_crash flag is always False here — callers that need it pass is_energy=True
    and can override with sector-level logic if needed.
    Falls back gracefully to an empty dict (all fields None) on fetch failure.
    """
    vix_df = _get_index_history("^VIX", start_str, end_str)
    tsx_df = _get_index_history("^GSPTSE", start_str, end_str)

    macro_by_date: dict = {}

    # Collect all dates that appear in either series
    all_dates = set()
    if vix_df is not None and not vix_df.empty:
        all_dates.update(vix_df.index.date)
    if tsx_df is not None and not tsx_df.empty:
        all_dates.update(tsx_df.index.date)

    prev_tsx_close = None
    for d in sorted(all_dates):
        vix_val = None
        tsx_chg = None

        if vix_df is not None:
            rows = vix_df[vix_df.index.date == d]
            if not rows.empty:
                vix_val = float(rows["Close"].iloc[-1])

        if tsx_df is not None:
            rows = tsx_df[tsx_df.index.date == d]
            if not rows.empty:
                tsx_close = float(rows["Close"].iloc[-1])
                if prev_tsx_close is not None and prev_tsx_close > 0:
                    tsx_chg = (tsx_close - prev_tsx_close) / prev_tsx_close * 100
                prev_tsx_close = tsx_close

        risk_off = bool(
            (vix_val is not None and vix_val > 25)
            or (tsx_chg is not None and tsx_chg < -2.0)
        )
        macro_by_date[d] = {
            "vix": vix_val,
            "tsx_change_pct": tsx_chg,
            "usd_cad": None,   # omitted — not fetched to limit API calls
            "risk_off": risk_off,
            "oil_crash": False,
        }

    return macro_by_date


_NEUTRAL_MACRO = {"vix": None, "tsx_change_pct": None, "risk_off": False, "oil_crash": False}


# ── Signal generators ─────────────────────────────────────────────────────────

def _quick_signal(hist: pd.DataFrame) -> float:
    """Fast technical-only score [0, 100] using RSI + MACD + 50/200 SMA.
    Kept for test isolation (use_full_signal=False) and speed comparison."""
    close = hist["Close"]
    signals = []

    try:
        rsi = ta.momentum.RSIIndicator(close, window=14).rsi().dropna()
        if not rsi.empty:
            r = float(rsi.iloc[-1])
            if r < 30:
                signals.append(90)
            elif r < 45:
                signals.append(70)
            elif r < 55:
                signals.append(50)
            elif r < 70:
                signals.append(40)
            else:
                signals.append(20)
    except Exception:
        pass

    try:
        macd_obj = ta.trend.MACD(close)
        macd_line = macd_obj.macd().dropna()
        sig_line = macd_obj.macd_signal().dropna()
        if len(macd_line) >= 2:
            above = float(macd_line.iloc[-1]) > float(sig_line.iloc[-1])
            signals.append(70 if above else 30)
    except Exception:
        pass

    try:
        sma50 = ta.trend.SMAIndicator(close, window=50).sma_indicator().dropna()
        sma200 = ta.trend.SMAIndicator(close, window=200).sma_indicator().dropna()
        if not sma50.empty and not sma200.empty:
            signals.append(75 if float(sma50.iloc[-1]) > float(sma200.iloc[-1]) else 25)
    except Exception:
        pass

    return float(np.mean(signals)) if signals else 50.0


def _full_signal_backtest(
    hist_slice: pd.DataFrame,
    fundamentals: dict,
    macro_for_day: dict,
    is_energy: bool = False,
) -> tuple[float, float]:
    """
    3-axis composite signal for backtesting. Returns (pre_overlay_score, post_overlay_score).

    Axes and data fidelity:
      technical   — fully historical (hist_slice up to scan date; no look-ahead)
      fundamental — CURRENT-SNAPSHOT PROXY (see DECISIONS.md)
      macro       — fully historical (pre-fetched VIX + TSX composite)

    Sentiment and insider are excluded — they cannot be accurately replicated from
    free historical data. They are informational-only in live mode. See DECISIONS.md.
    apply_macro_overlay() fires on the historical risk_off flag.
    """
    from sterling.analyst import (
        score_technical, score_fundamental, score_macro,
        apply_macro_overlay, WEIGHTS,
    )

    tech = score_technical(hist_slice)
    fund = score_fundamental(fundamentals)
    macro_score = score_macro(macro_for_day)

    raw = sum({
        "technical":   tech,
        "fundamental": fund,
        "macro":       macro_score,
    }[k] * w for k, w in WEIGHTS.items())

    adjusted = apply_macro_overlay(raw, macro_for_day, is_energy=is_energy)
    return float(max(0.0, min(100.0, raw))), float(max(0.0, min(100.0, adjusted)))


# ── Data bundle ───────────────────────────────────────────────────────────────

def load_backtest_data(
    years: int,
    tickers: Optional[list] = None,
    use_full_signal: bool = True,
    universe: str = "tsx",
) -> dict:
    """
    Fetch all data needed for one or more backtest runs and return a reusable bundle.
    Pass the returned dict as `_preloaded` to run_backtest() to skip re-fetching.

    universe: "tsx" (default), "cdr", or "tsx-cdr"
    """
    from sterling.cdr_mapping import CDR_UNIVERSE, CDR_TICKERS

    end_dt = datetime.utcnow()
    start_dt = end_dt - timedelta(days=years * 365 + 60)
    start_str = start_dt.strftime("%Y-%m-%d")
    end_str = end_dt.strftime("%Y-%m-%d")

    if tickers is not None:
        pass  # explicit override
    elif universe == "cdr":
        tickers = CDR_TICKERS
    elif universe == "tsx-cdr":
        seen: set = set()
        tickers = []
        for t in list(TSX60_TICKERS) + list(CDR_TICKERS):
            if t not in seen:
                seen.add(t)
                tickers.append(t)
    else:
        tickers = TSX60_TICKERS

    logger.info(f"[load_backtest_data] universe={universe}, fetching {len(tickers)} tickers {start_str} to {end_str}...")

    price_data: dict = {}
    for ticker in tickers:
        df = _get_history_safe(ticker, start_str, end_str)
        if df is not None:
            price_data[ticker] = df
            logger.info(f"  Loaded {ticker}: {len(df)} bars")
        else:
            logger.warning(f"  Skipped {ticker}: insufficient data")

    xic = _get_history_safe("XIC.TO", start_str, end_str)
    if xic is None:
        xic = list(price_data.values())[0] if price_data else pd.DataFrame()

    macro_by_date: dict = {}
    fundamentals: dict = {}

    if use_full_signal:
        logger.info("[load_backtest_data] Fetching historical macro (VIX + TSX)...")
        macro_by_date = _fetch_historical_macro(start_str, end_str)
        logger.info(f"  Macro data: {len(macro_by_date)} dates")

        # Fetch current-snapshot fundamentals once per ticker.
        # WARNING: these are NOT point-in-time — they represent today's values
        # used as a proxy across the entire backtest window.
        # For CDR tickers (.NE), get_fundamentals() redirects to the US underlying internally.
        logger.info("[load_backtest_data] Fetching current-snapshot fundamentals (not point-in-time)...")
        from sterling import data_feed
        for ticker in price_data:
            try:
                fundamentals[ticker] = data_feed.get_fundamentals(ticker)
            except Exception:
                fundamentals[ticker] = {}

    # Determine common trading calendar from the first loaded ticker
    if price_data:
        common_ticker = list(price_data.keys())[0]
        ticker_dates = sorted(price_data[common_ticker].index.date)
    else:
        ticker_dates = []

    return {
        "price_data": price_data,
        "xic": xic,
        "macro_by_date": macro_by_date,
        "fundamentals": fundamentals,
        "ticker_dates": ticker_dates,
        "start_str": start_str,
        "end_str": end_str,
    }


# ── Score distribution calibration scan ──────────────────────────────────────

def scan_score_distribution(
    years: int = 3,
    tickers: Optional[list] = None,
    _preloaded: Optional[dict] = None,
    universe: str = "tsx",
) -> list:
    """
    Scan every eligible ticker on every weekly scan date (no trading) and collect
    the post-overlay composite scores. Used for threshold calibration.
    Returns list of {date, ticker, pre_score, post_score} dicts.
    """
    if _preloaded is not None:
        bundle = _preloaded
    else:
        bundle = load_backtest_data(years, tickers, use_full_signal=True, universe=universe)

    price_data = bundle["price_data"]
    macro_by_date = bundle.get("macro_by_date", {})
    fundamentals = bundle.get("fundamentals", {})
    ticker_dates = bundle["ticker_dates"]

    warmup = 200
    if len(ticker_dates) <= warmup:
        return []

    backtest_start = ticker_dates[warmup]
    scan_dates = [d for d in ticker_dates if d >= backtest_start]

    records = []
    for i, date in enumerate(scan_dates):
        if i % 5 != 0:
            continue
        macro_today = macro_by_date.get(date, _NEUTRAL_MACRO)

        for ticker, df in price_data.items():
            hist_slice = df[df.index.date <= date]
            if len(hist_slice) < 200:
                continue
            try:
                price = float(hist_slice["Close"].iloc[-1])
                avg_vol = float(hist_slice["Volume"].tail(20).mean())
                vol_min = 500_000 if ticker.endswith(".NE") else 100_000
                if price < 5.0 or avg_vol < vol_min:
                    continue
                base = ticker.replace(".TO", "").replace(".V", "").replace(".NE", "").upper()
                is_energy = base in _ENERGY_BASES
                fund = fundamentals.get(ticker, {})
                pre, post = _full_signal_backtest(hist_slice, fund, macro_today, is_energy)
                records.append({
                    "date": str(date),
                    "ticker": ticker,
                    "pre_score": round(pre, 2),
                    "post_score": round(post, 2),
                })
            except Exception:
                continue

    return records


# ── Walk-forward engine ───────────────────────────────────────────────────────

class BacktestResult:
    def __init__(self):
        self.equity_curve: list = []            # [(date, portfolio_value)]
        self.trades: list = []                  # list of trade dicts
        self.benchmark_curve: list = []         # [(date, xic_value)]
        self.stats: dict = {}
        self.signal_mode: str = ""              # "full" or "quick"
        self.macro_overlay_events: list = []    # {date, ticker, pre_score, post_score}


def run_backtest(
    years: int = 3,
    capital_cad: float = 1000.0,
    hold_days: int = 20,
    min_hold_days: int = 10,
    max_positions: int = 4,
    signal_threshold: float = 65.0,
    stop_loss_pct: float = 0.06,
    slippage_pct: float = 0.001,
    tickers: Optional[list] = None,
    seed: int = 42,
    use_full_signal: bool = True,
    _preloaded: Optional[dict] = None,
    universe: str = "tsx",
) -> BacktestResult:
    """
    Walk-forward backtest. Scans tickers every 5 trading days; buys when signal
    exceeds threshold; exits at stop-loss, target (2x risk), or hold_days —
    whichever comes first. Stop-loss exits ignore min_hold_days; time exits respect it.

    Args:
        use_full_signal: True = 5-axis composite (recommended); False = quick
            technical-only fallback (used in unit tests to avoid network calls).
        _preloaded: dict from load_backtest_data(). If provided, skip re-fetching
            — used by sweep_backtests() to share a single data download.
        universe: "tsx", "cdr", or "tsx-cdr". Used only when _preloaded is None.
    """
    np.random.seed(seed)
    result = BacktestResult()
    result.signal_mode = "full" if use_full_signal else "quick"

    if min_hold_days > hold_days:
        logger.warning(
            f"min_hold_days ({min_hold_days}) > hold_days ({hold_days}); "
            "time-exits will never trigger — only stop/target exits possible."
        )

    # ── Data loading ──────────────────────────────────────────────────────────
    if _preloaded is not None:
        price_data = _preloaded["price_data"]
        xic = _preloaded["xic"]
        macro_by_date = _preloaded.get("macro_by_date", {})
        fundamentals = _preloaded.get("fundamentals", {})
        ticker_dates = _preloaded["ticker_dates"]
    else:
        bundle = load_backtest_data(years, tickers, use_full_signal=use_full_signal, universe=universe)
        price_data = bundle["price_data"]
        xic = bundle["xic"]
        macro_by_date = bundle.get("macro_by_date", {})
        fundamentals = bundle.get("fundamentals", {})
        ticker_dates = bundle["ticker_dates"]

    if not price_data:
        raise RuntimeError("No price data loaded — check internet connection")

    if use_full_signal and not macro_by_date:
        logger.warning(
            "Full signal requested but macro data is empty — "
            "macro scores will default to neutral (50). Check VIX/TSX fetch."
        )

    # ── Walk-forward setup ────────────────────────────────────────────────────
    warmup = 200
    if len(ticker_dates) <= warmup:
        raise RuntimeError("Not enough data for 200-bar warmup")

    backtest_start = ticker_dates[warmup]
    scan_dates = [d for d in ticker_dates if d >= backtest_start]

    logger.info(
        f"Walk-forward [{result.signal_mode} signal] from {backtest_start} "
        f"over {len(scan_dates)} days, threshold={signal_threshold}, "
        f"hold={hold_days}, min_hold={min_hold_days}"
    )
    if use_full_signal:
        logger.warning(
            "REMINDER: fundamental scores use current-snapshot values, "
            "not point-in-time historical. See DECISIONS.md."
        )

    cash = capital_cad
    positions: dict = {}
    equity_curve = []
    trades = []

    for date in scan_dates:
        date_str = str(date)
        macro_today = macro_by_date.get(date, _NEUTRAL_MACRO)

        # Mark-to-market
        portfolio_value = cash
        positions_to_close = []

        for ticker, pos in positions.items():
            if ticker not in price_data:
                continue
            df = price_data[ticker]
            day_rows = df[df.index.date == date]
            if day_rows.empty:
                prior = df[df.index.date < date]
                if prior.empty:
                    continue
                price_today = float(prior["Close"].iloc[-1])
            else:
                price_today = float(day_rows["Close"].iloc[-1])

            portfolio_value += pos["shares"] * price_today

            days_held = (date - pos["entry_date"]).days
            stop_hit = price_today <= pos["stop_loss"]
            target_hit = price_today >= pos["target"]
            time_exit = (days_held >= hold_days) and (days_held >= min_hold_days)

            if stop_hit or target_hit or time_exit:
                positions_to_close.append((ticker, price_today, days_held))

        equity_curve.append((date, round(portfolio_value, 2)))

        # Close positions
        for ticker, exit_price, days_held in positions_to_close:
            pos = positions.pop(ticker)
            effective_exit = exit_price * (1 - slippage_pct)
            proceeds = pos["shares"] * effective_exit
            cost_basis = pos["shares"] * pos["entry_price"]
            pnl = proceeds - cost_basis
            pnl_pct = pnl / cost_basis * 100

            cash += proceeds
            exit_reason = (
                "STOP" if exit_price <= pos["stop_loss"] else
                "TARGET" if exit_price >= pos["target"] else "TIME"
            )
            trades.append({
                "ticker": ticker,
                "entry_date": str(pos["entry_date"]),
                "exit_date": date_str,
                "entry_price": round(pos["entry_price"], 4),
                "exit_price": round(effective_exit, 4),
                "shares": pos["shares"],
                "pnl_cad": round(pnl, 2),
                "pnl_pct": round(pnl_pct, 2),
                "days_held": days_held,
                "exit_reason": exit_reason,
                "score": round(pos.get("score", 0.0), 1),
            })

        # Scan for new entries every 5 trading days
        if scan_dates.index(date) % 5 != 0:
            continue

        if len(positions) >= max_positions or cash < 50:
            continue

        scored = []
        for ticker, df in price_data.items():
            if ticker in positions:
                continue
            hist_slice = df[df.index.date <= date]
            if len(hist_slice) < 200:
                continue
            try:
                price = float(hist_slice["Close"].iloc[-1])
                avg_vol = float(hist_slice["Volume"].tail(20).mean())
                # CDR tickers (.NE) have a higher volume minimum (500k vs 100k)
                vol_min = 500_000 if ticker.endswith(".NE") else 100_000
                if price < 5.0 or avg_vol < vol_min:
                    continue

                if use_full_signal:
                    base = ticker.replace(".TO", "").replace(".V", "").replace(".NE", "").upper()
                    is_energy = base in _ENERGY_BASES
                    fund = fundamentals.get(ticker, {})
                    pre_score, post_score = _full_signal_backtest(
                        hist_slice, fund, macro_today, is_energy
                    )
                    if pre_score != post_score:
                        result.macro_overlay_events.append({
                            "date": str(date),
                            "ticker": ticker,
                            "pre_score": round(pre_score, 1),
                            "post_score": round(post_score, 1),
                        })
                    score = post_score
                else:
                    score = _quick_signal(hist_slice)

                if score >= signal_threshold:
                    scored.append((ticker, score, price, avg_vol))
            except Exception:
                continue

        scored.sort(key=lambda x: x[1], reverse=True)

        for ticker, score, price, avg_vol in scored[:2]:
            if len(positions) >= max_positions or cash < 50:
                break

            position_budget = min(cash, capital_cad / max_positions)
            effective_entry = price * (1 + slippage_pct)
            shares = max(1, int(position_budget / effective_entry))
            cost = shares * effective_entry

            if cost > cash:
                continue

            stop = effective_entry * (1 - stop_loss_pct)
            target = effective_entry * (1 + stop_loss_pct * 2)

            cash -= cost
            positions[ticker] = {
                "entry_price": effective_entry,
                "shares": shares,
                "entry_date": date,
                "stop_loss": stop,
                "target": target,
                "score": score,
            }
            logger.debug(
                f"  BUY {ticker} @ ${effective_entry:.2f} x {shares} = "
                f"${cost:.0f} (score {score:.1f})"
            )

    # Close all remaining positions at last available price
    last_date = scan_dates[-1]
    for ticker, pos in positions.items():
        if ticker not in price_data:
            continue
        df = price_data[ticker]
        last_row = df[df.index.date <= last_date]
        if last_row.empty:
            continue
        exit_price = float(last_row["Close"].iloc[-1]) * (1 - slippage_pct)
        pnl = (exit_price - pos["entry_price"]) * pos["shares"]
        cash += pos["shares"] * exit_price
        trades.append({
            "ticker": ticker,
            "entry_date": str(pos["entry_date"]),
            "exit_date": str(last_date),
            "entry_price": round(pos["entry_price"], 4),
            "exit_price": round(exit_price, 4),
            "shares": pos["shares"],
            "pnl_cad": round(pnl, 2),
            "pnl_pct": round(pnl / (pos["entry_price"] * pos["shares"]) * 100, 2),
            "days_held": (last_date - pos["entry_date"]).days,
            "exit_reason": "END",
            "score": round(pos.get("score", 0.0), 1),
        })

    result.equity_curve = equity_curve
    result.trades = trades

    # Build benchmark (XIC.TO buy-and-hold with 1.5% FX drag on initial purchase)
    if xic is not None and not xic.empty:
        xic_dates = [d for d in xic.index.date if d >= backtest_start]
        if xic_dates:
            xic_start_price = float(xic[xic.index.date == xic_dates[0]]["Close"].iloc[0])
            xic_shares = (capital_cad * 0.985) / xic_start_price
            benchmark_curve = []
            for d in xic_dates:
                rows = xic[xic.index.date == d]
                if not rows.empty:
                    val = xic_shares * float(rows["Close"].iloc[-1])
                    benchmark_curve.append((d, round(val, 2)))
            result.benchmark_curve = benchmark_curve

    result.stats = _compute_stats(equity_curve, trades, capital_cad, result.benchmark_curve)
    result.stats["signal_mode"] = result.signal_mode
    result.stats["min_hold_days"] = min_hold_days
    result.stats["signal_threshold"] = signal_threshold
    result.stats["macro_overlay_fire_count"] = len(result.macro_overlay_events)
    return result


# ── Parameter sweep ───────────────────────────────────────────────────────────

def sweep_backtests(
    years: int = 3,
    capital_cad: float = 1000.0,
    thresholds: tuple = (65, 75),
    min_holds: tuple = (5, 15),
    hold_days: int = 20,
    tickers: Optional[list] = None,
    _preloaded: Optional[dict] = None,
    universe: str = "tsx",
) -> list:
    """
    Run a parameter sweep (entry_threshold × min_hold_days).
    Downloads data once; passes the bundle to all runs.
    Returns list of {threshold, min_hold, result} dicts.
    """
    logger.info(
        f"Starting sweep: thresholds={thresholds}, min_holds={min_holds}, "
        f"hold_days={hold_days}, years={years}, universe={universe}"
    )
    bundle = _preloaded if _preloaded is not None else load_backtest_data(years, tickers, use_full_signal=True, universe=universe)

    results = []
    for threshold in thresholds:
        for min_hold in min_holds:
            logger.info(f"  Running threshold={threshold}, min_hold={min_hold}...")
            result = run_backtest(
                years=years,
                capital_cad=capital_cad,
                hold_days=hold_days,
                min_hold_days=min_hold,
                signal_threshold=float(threshold),
                use_full_signal=True,
                _preloaded=bundle,
            )
            results.append({
                "threshold": threshold,
                "min_hold": min_hold,
                "result": result,
            })

    return results


def save_sweep_results(sweep: list, output_dir: Optional[Path] = None) -> Path:
    """Save sweep comparison table (CSV + text) and per-run trade logs."""
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out = output_dir or _DATA_DIR / f"sweep_{ts}"
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for item in sweep:
        s = item["result"].stats
        rows.append({
            "threshold": item["threshold"],
            "min_hold_days": item["min_hold"],
            "cagr_pct": s.get("cagr_pct", 0),
            "sharpe": s.get("sharpe", 0),
            "sortino": s.get("sortino", 0),
            "max_drawdown_pct": s.get("max_drawdown_pct", 0),
            "win_rate_pct": s.get("win_rate_pct", 0),
            "total_trades": s.get("total_trades", 0),
            "total_return_pct": s.get("total_return_pct", 0),
            "benchmark_xic_cagr_pct": s.get("benchmark_xic_cagr_pct", 0),
            "beats_benchmark": s.get("beats_benchmark", False),
        })

    with open(out / "sweep_comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    # Save per-run trade logs
    for item in sweep:
        tag = f"t{item['threshold']}_h{item['min_hold']}"
        trades = item["result"].trades
        if trades:
            with open(out / f"trades_{tag}.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=trades[0].keys())
                w.writeheader()
                w.writerows(trades)
        with open(out / f"summary_{tag}.json", "w") as f:
            json.dump(item["result"].stats, f, indent=2)

    logger.info(f"Sweep results saved to {out}")
    return out


# ── Stats ─────────────────────────────────────────────────────────────────────

def _compute_stats(equity_curve: list, trades: list, capital_cad: float, benchmark_curve: list) -> dict:
    if not equity_curve:
        return {}

    dates = [e[0] for e in equity_curve]
    values = [e[1] for e in equity_curve]

    returns = pd.Series(values).pct_change().dropna()
    total_days = (dates[-1] - dates[0]).days
    years = total_days / 365.25

    final_value = values[-1]
    cagr = ((final_value / capital_cad) ** (1 / years) - 1) * 100 if years > 0 else 0

    rf_daily = 0.04 / 252
    excess = returns - rf_daily
    sharpe = float(excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else 0

    neg_returns = returns[returns < 0]
    sortino = (
        float(excess.mean() / neg_returns.std() * np.sqrt(252))
        if len(neg_returns) > 0 and neg_returns.std() > 0
        else 0
    )

    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_dd = float(drawdown.min()) * 100

    if trades:
        wins = [t for t in trades if t["pnl_cad"] > 0]
        losses = [t for t in trades if t["pnl_cad"] <= 0]
        win_rate = len(wins) / len(trades) * 100
        avg_win = np.mean([t["pnl_cad"] for t in wins]) if wins else 0
        avg_loss = abs(np.mean([t["pnl_cad"] for t in losses])) if losses else 0
        profit_factor = (
            (avg_win * len(wins)) / (avg_loss * len(losses))
            if losses and avg_loss > 0
            else float("inf")
        )
        avg_rr = (
            np.mean([abs(t["pnl_pct"]) for t in wins]) /
            np.mean([abs(t["pnl_pct"]) for t in losses])
            if losses else 0
        )
    else:
        win_rate = profit_factor = avg_rr = 0

    bm_cagr = 0
    bm_total_return = 0
    if benchmark_curve and len(benchmark_curve) > 1:
        bm_end = benchmark_curve[-1][1]
        bm_total_return = (bm_end - capital_cad) / capital_cad * 100
        bm_cagr = ((bm_end / capital_cad) ** (1 / years) - 1) * 100 if years > 0 else 0

    return {
        "start_date": str(dates[0]),
        "end_date": str(dates[-1]),
        "years": round(years, 2),
        "capital_cad": capital_cad,
        "final_value_cad": round(final_value, 2),
        "total_return_pct": round((final_value - capital_cad) / capital_cad * 100, 2),
        "cagr_pct": round(cagr, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_drawdown_pct": round(max_dd, 2),
        "win_rate_pct": round(win_rate, 1),
        "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else "inf",
        "avg_rr": round(avg_rr, 2),
        "total_trades": len(trades),
        "benchmark_xic_cagr_pct": round(bm_cagr, 2),
        "benchmark_xic_total_return_pct": round(bm_total_return, 2),
        "beats_benchmark": cagr > bm_cagr,
    }


# ── Save results ──────────────────────────────────────────────────────────────

def save_results(result: BacktestResult, output_dir: Optional[Path] = None) -> Path:
    """Save equity curve PNG, monthly heatmap PNG, stats CSV, trade log CSV."""
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out = output_dir or _DATA_DIR / f"backtest_{ts}"
    out.mkdir(parents=True, exist_ok=True)

    if result.equity_curve:
        dates_e = [e[0] for e in result.equity_curve]
        values_e = [e[1] for e in result.equity_curve]
        dates_b = [b[0] for b in result.benchmark_curve]
        values_b = [b[1] for b in result.benchmark_curve]

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(dates_e, values_e, label="Sterling Strategy", linewidth=2, color="#2ecc71")
        if dates_b:
            ax.plot(dates_b, values_b, label="XIC.TO (B&H + FX drag)", linewidth=2,
                    color="#3498db", linestyle="--")
        mode_label = result.stats.get("signal_mode", "")
        ax.set_title(
            f"Sterling Strategy vs XIC.TO Buy-and-Hold "
            f"[signal={mode_label}, threshold={result.stats.get('signal_threshold','?')}, "
            f"min_hold={result.stats.get('min_hold_days','?')}d]",
            fontsize=12,
        )
        ax.set_xlabel("Date")
        ax.set_ylabel("Portfolio Value (CAD)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.xticks(rotation=45)
        plt.tight_layout()
        fig.savefig(out / "equity_curve.png", dpi=150)
        plt.close(fig)

    if result.equity_curve and len(result.equity_curve) > 30:
        df_eq = pd.DataFrame(result.equity_curve, columns=["date", "value"])
        df_eq["date"] = pd.to_datetime(df_eq["date"])
        df_eq = df_eq.set_index("date")
        monthly = df_eq["value"].resample("ME").last().pct_change().dropna() * 100

        monthly_df = pd.DataFrame({
            "Year": monthly.index.year,
            "Month": monthly.index.month,
            "Return": monthly.values,
        })
        if not monthly_df.empty:
            pivot = monthly_df.pivot(index="Year", columns="Month", values="Return")
            pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                             "Jul","Aug","Sep","Oct","Nov","Dec"][:len(pivot.columns)]
            fig, ax = plt.subplots(figsize=(14, max(4, len(pivot) * 0.8)))
            sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn",
                        center=0, linewidths=0.5, ax=ax, cbar_kws={"label": "Return (%)"})
            ax.set_title("Monthly Returns Heatmap (%)", fontsize=13)
            plt.tight_layout()
            fig.savefig(out / "monthly_heatmap.png", dpi=150)
            plt.close(fig)

    with open(out / "stats.csv", "w", newline="") as f:
        w = csv.writer(f)
        for k, v in result.stats.items():
            w.writerow([k, v])

    if result.trades:
        with open(out / "trades.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=result.trades[0].keys())
            w.writeheader()
            w.writerows(result.trades)

    with open(out / "summary.json", "w") as f:
        json.dump(result.stats, f, indent=2)

    logger.info(f"Backtest results saved to {out}")
    return out
