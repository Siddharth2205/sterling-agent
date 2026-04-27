"""Post-hoc diagnostic analysis of a completed backtest run."""

import csv
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


_DATA_DIR = Path(__file__).parent.parent / "data"

SLIPPAGE_PCT = 0.001  # matches backtester default


# ── Loader ────────────────────────────────────────────────────────────────────

def find_latest_backtest(data_dir: Optional[Path] = None) -> Optional[Path]:
    """Return the most recent data/backtest_*/ directory, or None."""
    root = data_dir or _DATA_DIR
    candidates = sorted(root.glob("backtest_*/"), reverse=True)
    return candidates[0] if candidates else None


def load_backtest(bt_dir: Path) -> tuple[dict, pd.DataFrame]:
    """Load summary stats and trade log from a backtest directory."""
    summary_path = bt_dir / "summary.json"
    stats_path = bt_dir / "stats.csv"
    trades_path = bt_dir / "trades.csv"

    stats: dict = {}
    if summary_path.exists():
        with open(summary_path) as f:
            stats = json.load(f)
    elif stats_path.exists():
        with open(stats_path, newline="") as f:
            reader = csv.reader(f)
            for k, v in reader:
                try:
                    stats[k] = float(v)
                except (ValueError, TypeError):
                    stats[k] = v

    trades_df = pd.DataFrame()
    if trades_path.exists():
        trades_df = pd.read_csv(trades_path)
        if not trades_df.empty:
            trades_df["entry_date"] = pd.to_datetime(trades_df["entry_date"]).dt.date
            trades_df["exit_date"] = pd.to_datetime(trades_df["exit_date"]).dt.date

    return stats, trades_df


# ── Metric 1: cash exposure ───────────────────────────────────────────────────

def cash_days_analysis(trades_df: pd.DataFrame, start_date: date, end_date: date) -> dict:
    """
    For each business day in [start_date, end_date], check whether any trade
    was open (entry_date <= day <= exit_date-1 so exit day counts as closed).
    Returns fraction of days fully in cash.
    """
    all_days = pd.bdate_range(start=str(start_date), end=str(end_date))
    total = len(all_days)

    if trades_df.empty or total == 0:
        return {
            "total_trading_days": total,
            "days_in_cash": total,
            "days_with_positions": 0,
            "pct_in_cash": 100.0,
            "pct_with_positions": 0.0,
            "peak_simultaneous_positions": 0,
        }

    cash_count = 0
    peak = 0

    for day in all_days:
        d = day.date()
        # A trade is "open" on day D if entry_date <= D < exit_date
        # (on exit_date the position is liquidated at market open/close)
        open_mask = (trades_df["entry_date"] <= d) & (trades_df["exit_date"] > d)
        n_open = int(open_mask.sum())
        if n_open == 0:
            cash_count += 1
        if n_open > peak:
            peak = n_open

    days_with = total - cash_count
    return {
        "total_trading_days": total,
        "days_in_cash": cash_count,
        "days_with_positions": days_with,
        "pct_in_cash": round(cash_count / total * 100, 1) if total else 0.0,
        "pct_with_positions": round(days_with / total * 100, 1) if total else 0.0,
        "peak_simultaneous_positions": peak,
    }


# ── Metric 2: confidence score distribution ───────────────────────────────────

def score_distribution(trades_df: pd.DataFrame) -> Optional[dict]:
    """
    Quartile stats + ASCII histogram of entry confidence scores.
    Returns None if 'score' column is not present (pre-fix backtests).
    """
    if trades_df.empty or "score" not in trades_df.columns:
        return None

    scores = trades_df["score"].dropna()
    if scores.empty:
        return None

    q1, median, q3 = np.percentile(scores, [25, 50, 75])

    # Build ASCII histogram with 10-point bins
    bins = list(range(0, 101, 10))
    counts, edges = np.histogram(scores, bins=bins)
    max_count = max(counts) if counts.max() > 0 else 1
    bar_width = 20
    rows = []
    for i, count in enumerate(counts):
        label = f"{edges[i]:>3.0f}-{edges[i+1]:.0f}"
        bar_len = int(count / max_count * bar_width)
        bar = "#" * bar_len
        rows.append(f"  {label} | {bar:<{bar_width}} {count}")
    histogram = "\n".join(rows)

    return {
        "count": len(scores),
        "min": round(float(scores.min()), 1),
        "q1": round(float(q1), 1),
        "median": round(float(median), 1),
        "q3": round(float(q3), 1),
        "max": round(float(scores.max()), 1),
        "mean": round(float(scores.mean()), 1),
        "histogram": histogram,
    }


# ── Metric 3: macro overlay demotions ────────────────────────────────────────

def macro_overlay_analysis(trades_df: pd.DataFrame) -> dict:
    """
    Count BUY signals demoted to HOLD by apply_macro_overlay.
    In the current backtest, _quick_signal() is purely technical and never
    calls apply_macro_overlay(), so demotions are structurally 0.
    If 'pre_overlay_score' is present (future backtest versions), compute
    demotion count from the score delta.
    """
    has_overlay_col = (
        not trades_df.empty
        and "pre_overlay_score" in trades_df.columns
        and "score" in trades_df.columns
    )

    if has_overlay_col:
        overlay_applied = trades_df["pre_overlay_score"] > trades_df["score"]
        demoted_to_hold = overlay_applied & (trades_df["pre_overlay_score"] >= 75) & (trades_df["score"] < 75)
        return {
            "has_overlay_data": True,
            "overlay_applied_count": int(overlay_applied.sum()),
            "demoted_buy_to_lower": int(demoted_to_hold.sum()),
            "note": None,
        }

    return {
        "has_overlay_data": False,
        "overlay_applied_count": 0,
        "demoted_buy_to_lower": 0,
        "note": (
            "The backtest engine uses _quick_signal() — a technical-only scorer "
            "that never calls apply_macro_overlay(). Macro overlay demotions are "
            "structurally absent from all backtest runs. This is a fidelity gap: "
            "the live analyst path applies a -15pt downgrade when VIX > 25 or "
            "TSX intraday < -2%, which would suppress entries during drawdown "
            "periods — potentially helping OR hurting depending on mean-reversion "
            "vs momentum regime."
        ),
    }


# ── Metric 4: gross vs net return (cost drag) ────────────────────────────────

def cost_drag_analysis(trades_df: pd.DataFrame, slippage_pct: float = SLIPPAGE_PCT) -> dict:
    """
    Reconstruct gross PnL (pre-slippage) from trade records.

    In the backtester:
      effective_entry = raw_price * (1 + slippage_pct)
      effective_exit  = raw_exit  * (1 - slippage_pct)

    So:
      raw_entry = effective_entry / (1 + slippage_pct)
      raw_exit  = effective_exit  / (1 - slippage_pct)
      gross_pnl = (raw_exit - raw_entry) * shares

    FX drag (1.5%) is applied only to the XIC benchmark, not to individual
    trades, so there is no per-trade FX cost to strip out.
    """
    if trades_df.empty:
        return {
            "net_pnl_cad": 0.0,
            "gross_pnl_cad": 0.0,
            "total_slippage_cost_cad": 0.0,
            "slippage_as_pct_of_gross": 0.0,
            "fx_drag_on_trades_cad": 0.0,
            "note": "No trades found.",
        }

    factor_entry = 1 + slippage_pct
    factor_exit = 1 - slippage_pct

    raw_entries = trades_df["entry_price"] / factor_entry
    raw_exits = trades_df["exit_price"] / factor_exit
    gross_pnl_series = (raw_exits - raw_entries) * trades_df["shares"]

    gross_pnl = float(gross_pnl_series.sum())
    net_pnl = float(trades_df["pnl_cad"].sum())
    slippage_cost = net_pnl - gross_pnl  # negative — slippage hurts

    round_trip_drag_pct = slippage_pct * 2 * 100  # per-trade %
    total_trades = len(trades_df)

    return {
        "net_pnl_cad": round(net_pnl, 2),
        "gross_pnl_cad": round(gross_pnl, 2),
        "total_slippage_cost_cad": round(slippage_cost, 2),
        "slippage_as_pct_of_gross": round(slippage_cost / gross_pnl * 100, 2) if gross_pnl != 0 else 0.0,
        "round_trip_drag_pct": round(round_trip_drag_pct, 3),
        "total_trades": total_trades,
        "implied_slippage_per_trade": round(slippage_cost / total_trades, 2) if total_trades else 0.0,
        "fx_drag_on_trades_cad": 0.0,
        "fx_drag_note": (
            "FX drag (1.5% Wealthsimple spread) is applied to the XIC.TO benchmark "
            "to make the comparison conservative, but is NOT deducted from individual "
            "trade entries in this backtest. USD-listed cross-listings would face this "
            "cost in live trading."
        ),
    }


# ── Master runner ──────────────────────────────────────────────────────────────

def run_diagnostics(bt_dir: Path) -> dict:
    """Run all four diagnostic checks against a backtest directory."""
    stats, trades_df = load_backtest(bt_dir)

    start_date = None
    end_date = None
    try:
        start_date = date.fromisoformat(str(stats.get("start_date", "")))
        end_date = date.fromisoformat(str(stats.get("end_date", "")))
    except (ValueError, TypeError):
        pass

    cash = {}
    if start_date and end_date:
        cash = cash_days_analysis(trades_df, start_date, end_date)

    scores = score_distribution(trades_df)
    macro = macro_overlay_analysis(trades_df)
    drag = cost_drag_analysis(trades_df)

    return {
        "bt_dir": str(bt_dir),
        "stats": stats,
        "cash_exposure": cash,
        "score_distribution": scores,
        "macro_overlay": macro,
        "cost_drag": drag,
    }
