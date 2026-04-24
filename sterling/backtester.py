"""Walk-forward backtest on TSX 60 constituents vs XIC.TO buy-and-hold."""

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


# ── Simple signal generator (uses same technical scoring logic) ───────────────

def _quick_signal(hist: pd.DataFrame) -> float:
    """Fast signal score [0, 100] using RSI + MACD + 50/200 SMA."""
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


# ── Walk-forward engine ───────────────────────────────────────────────────────

class BacktestResult:
    def __init__(self):
        self.equity_curve: list = []     # [(date, portfolio_value)]
        self.trades: list = []           # list of trade dicts
        self.benchmark_curve: list = []  # [(date, xic_value)]
        self.stats: dict = {}


def run_backtest(
    years: int = 3,
    capital_cad: float = 1000.0,
    hold_days: int = 20,
    max_positions: int = 4,
    signal_threshold: float = 65.0,
    stop_loss_pct: float = 0.06,
    slippage_pct: float = 0.001,
    tickers: Optional[list] = None,
    seed: int = 42,
) -> BacktestResult:
    """
    Walk-forward backtest. Scans tickers weekly; buys when signal > threshold;
    exits at stop-loss, target (2x risk), or hold_days — whichever comes first.
    """
    np.random.seed(seed)
    result = BacktestResult()

    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=years * 365 + 60)  # extra for warmup
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    tickers = tickers or TSX60_TICKERS
    logger.info(f"Fetching {len(tickers)} tickers from {start_str} to {end_str}...")

    # Load all price history upfront
    price_data: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        df = _get_history_safe(ticker, start_str, end_str)
        if df is not None:
            price_data[ticker] = df
            logger.info(f"  Loaded {ticker}: {len(df)} bars")
        else:
            logger.warning(f"  Skipped {ticker}: insufficient data")

    if not price_data:
        raise RuntimeError("No price data loaded — check internet connection")

    # Load benchmark (XIC.TO)
    xic = _get_history_safe("XIC.TO", start_str, end_str)
    if xic is None:
        logger.warning("Could not load XIC.TO benchmark — using first ticker as proxy")
        xic = list(price_data.values())[0]

    # Align all dates to common trading calendar
    all_dates = sorted(set().union(*[set(df.index.date) for df in price_data.values()]))
    # Start backtest after warmup period (200 bars for SMA200)
    warmup = 200
    common_ticker = list(price_data.keys())[0]
    ticker_dates = sorted(price_data[common_ticker].index.date)
    if len(ticker_dates) <= warmup:
        raise RuntimeError("Not enough data for 200-bar warmup")

    backtest_start = ticker_dates[warmup]

    cash = capital_cad
    positions: dict = {}  # ticker → {entry_price, shares, entry_date, stop_loss, target}
    equity_curve = []
    trades = []

    # Walk forward week by week
    current_scan_idx = warmup
    scan_dates = [d for d in ticker_dates if d >= backtest_start]

    logger.info(f"Running walk-forward from {backtest_start} over {len(scan_dates)} trading days...")

    for date in scan_dates:
        date_str = str(date)

        # Mark-to-market portfolio value
        portfolio_value = cash
        positions_to_close = []

        for ticker, pos in positions.items():
            if ticker not in price_data:
                continue
            df = price_data[ticker]
            day_rows = df[df.index.date == date]
            if day_rows.empty:
                # Use last known price
                prior = df[df.index.date < date]
                if prior.empty:
                    continue
                price_today = float(prior["Close"].iloc[-1])
            else:
                price_today = float(day_rows["Close"].iloc[-1])

            portfolio_value += pos["shares"] * price_today

            # Check exit conditions
            days_held = (date - pos["entry_date"]).days
            if (price_today <= pos["stop_loss"] or
                    price_today >= pos["target"] or
                    days_held >= hold_days):
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
            })

        # Scan for new entries weekly (every 5 trading days)
        if scan_dates.index(date) % 5 != 0:
            continue

        if len(positions) >= max_positions or cash < 50:
            continue

        # Score available tickers
        scored = []
        for ticker, df in price_data.items():
            if ticker in positions:
                continue
            hist_slice = df[df.index.date <= date]
            if len(hist_slice) < 200:
                continue
            try:
                score = _quick_signal(hist_slice)
                price = float(hist_slice["Close"].iloc[-1])
                avg_vol = float(hist_slice["Volume"].tail(20).mean())
                if price >= 5.0 and avg_vol >= 100_000 and score >= signal_threshold:
                    scored.append((ticker, score, price, avg_vol))
            except Exception:
                continue

        scored.sort(key=lambda x: x[1], reverse=True)

        for ticker, score, price, avg_vol in scored[:2]:  # buy up to 2 per scan
            if len(positions) >= max_positions or cash < 50:
                break

            position_budget = min(cash, capital_cad / max_positions)
            effective_entry = price * (1 + slippage_pct)
            shares = max(1, int(position_budget / effective_entry))
            cost = shares * effective_entry

            if cost > cash:
                continue

            stop = effective_entry * (1 - stop_loss_pct)
            target = effective_entry * (1 + stop_loss_pct * 2)  # 2:1 R:R minimum

            cash -= cost
            positions[ticker] = {
                "entry_price": effective_entry,
                "shares": shares,
                "entry_date": date,
                "stop_loss": stop,
                "target": target,
                "score": score,
            }
            logger.debug(f"  BUY {ticker} @ ${effective_entry:.2f} x {shares} = ${cost:.0f} (score {score:.0f})")

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
        })

    result.equity_curve = equity_curve
    result.trades = trades

    # Build benchmark (XIC.TO buy-and-hold with 1.5% FX drag on initial purchase)
    xic_dates = [d for d in xic.index.date if d >= backtest_start]
    if xic_dates:
        xic_start_price = float(xic[xic.index.date == xic_dates[0]]["Close"].iloc[0])
        xic_shares = (capital_cad * 0.985) / xic_start_price  # 1.5% FX drag
        benchmark_curve = []
        for d in xic_dates:
            rows = xic[xic.index.date == d]
            if not rows.empty:
                val = xic_shares * float(rows["Close"].iloc[-1])
                benchmark_curve.append((d, round(val, 2)))
        result.benchmark_curve = benchmark_curve

    # Compute stats
    result.stats = _compute_stats(equity_curve, trades, capital_cad, result.benchmark_curve)
    return result


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

    rf_daily = 0.04 / 252  # risk-free rate assumption
    excess = returns - rf_daily
    sharpe = float(excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else 0

    neg_returns = returns[returns < 0]
    sortino = float(excess.mean() / neg_returns.std() * np.sqrt(252)) if len(neg_returns) > 0 and neg_returns.std() > 0 else 0

    # Max drawdown
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_dd = float(drawdown.min()) * 100

    # Trade stats
    if trades:
        wins = [t for t in trades if t["pnl_cad"] > 0]
        losses = [t for t in trades if t["pnl_cad"] <= 0]
        win_rate = len(wins) / len(trades) * 100

        avg_win = np.mean([t["pnl_cad"] for t in wins]) if wins else 0
        avg_loss = abs(np.mean([t["pnl_cad"] for t in losses])) if losses else 0
        profit_factor = (avg_win * len(wins)) / (avg_loss * len(losses)) if losses and avg_loss > 0 else float("inf")

        avg_rr = np.mean([abs(t["pnl_pct"]) for t in wins]) / np.mean([abs(t["pnl_pct"]) for t in losses]) if losses else 0
    else:
        win_rate = profit_factor = avg_rr = 0

    # Benchmark stats
    bm_cagr = 0
    bm_total_return = 0
    if benchmark_curve and len(benchmark_curve) > 1:
        bm_start = benchmark_curve[0][1]
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
        "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else "∞",
        "avg_rr": round(avg_rr, 2),
        "total_trades": len(trades),
        "benchmark_xic_cagr_pct": round(bm_cagr, 2),
        "benchmark_xic_total_return_pct": round(bm_total_return, 2),
        "beats_benchmark": cagr > bm_cagr,
    }


def save_results(result: BacktestResult, output_dir: Optional[Path] = None) -> Path:
    """Save equity curve PNG, monthly heatmap PNG, stats CSV, trade log CSV."""
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out = output_dir or _DATA_DIR / f"backtest_{ts}"
    out.mkdir(parents=True, exist_ok=True)

    # Equity curve chart
    if result.equity_curve:
        dates_e = [e[0] for e in result.equity_curve]
        values_e = [e[1] for e in result.equity_curve]
        dates_b = [b[0] for b in result.benchmark_curve]
        values_b = [b[1] for b in result.benchmark_curve]

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(dates_e, values_e, label="Sterling Strategy", linewidth=2, color="#2ecc71")
        if dates_b:
            ax.plot(dates_b, values_b, label="XIC.TO (B&H + FX drag)", linewidth=2, color="#3498db", linestyle="--")
        ax.set_title("Sterling Strategy vs XIC.TO Buy-and-Hold", fontsize=14)
        ax.set_xlabel("Date")
        ax.set_ylabel("Portfolio Value (CAD)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.xticks(rotation=45)
        plt.tight_layout()
        fig.savefig(out / "equity_curve.png", dpi=150)
        plt.close(fig)

    # Monthly returns heatmap
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
            pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][:len(pivot.columns)]

            fig, ax = plt.subplots(figsize=(14, max(4, len(pivot) * 0.8)))
            sns.heatmap(
                pivot, annot=True, fmt=".1f", cmap="RdYlGn",
                center=0, linewidths=0.5, ax=ax, cbar_kws={"label": "Return (%)"}
            )
            ax.set_title("Monthly Returns Heatmap (%)", fontsize=13)
            plt.tight_layout()
            fig.savefig(out / "monthly_heatmap.png", dpi=150)
            plt.close(fig)

    # Stats CSV
    with open(out / "stats.csv", "w", newline="") as f:
        w = csv.writer(f)
        for k, v in result.stats.items():
            w.writerow([k, v])

    # Trade log CSV
    if result.trades:
        with open(out / "trades.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=result.trades[0].keys())
            w.writeheader()
            w.writerows(result.trades)

    # JSON summary
    with open(out / "summary.json", "w") as f:
        json.dump(result.stats, f, indent=2)

    logger.info(f"Backtest results saved to {out}")
    return out
