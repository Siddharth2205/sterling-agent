"""Evaluate paper-trade signal outcomes against actual price history.

Reads data/paper_trades.csv, fetches post-signal price data via yfinance,
and determines whether each signal hit its target, stop, or expired.
Read-only analysis — does not modify analyst.py or scoring logic.
"""

import csv
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from sterling.paper_log import _CSV_PATH, FIELDNAMES

_OUT_PATH = Path(__file__).parent.parent / "data" / "paper_trades_with_outcomes.csv"
_HOLD_DAYS = 30

OUT_FIELDNAMES = FIELDNAMES + ["entry_price", "outcome", "outcome_date", "exit_price", "realized_pnl_pct"]


def _dedup_signals(records: list[dict]) -> list[dict]:
    """Keep first signal per ticker per calendar date."""
    seen: set[tuple[str, str]] = set()
    deduped = []
    for r in records:
        date_str = r["timestamp"][:10]
        key = (r["ticker"], date_str)
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    return deduped


def _fetch_prices(ticker: str, start: datetime, end: datetime) -> Optional[pd.DataFrame]:
    start_str = (start - timedelta(days=1)).strftime("%Y-%m-%d")
    end_str = (end + timedelta(days=5)).strftime("%Y-%m-%d")
    try:
        t = yf.Ticker(ticker)
        df = t.history(start=start_str, end=end_str, auto_adjust=True)
        if df.empty:
            return None
        df.index = df.index.tz_localize(None)
        return df
    except Exception:
        return None


def evaluate_signals(csv_path: Optional[Path] = None) -> list[dict]:
    """Evaluate all paper-trade signals and return enriched records."""
    path = Path(csv_path) if csv_path else _CSV_PATH
    if not path.exists():
        return []

    with open(path, newline="", encoding="utf-8") as f:
        raw = list(csv.DictReader(f))

    if not raw:
        return []

    signals = _dedup_signals(raw)

    tickers = sorted(set(r["ticker"] for r in signals))
    earliest = min(r["timestamp"][:10] for r in signals)
    start_dt = datetime.strptime(earliest, "%Y-%m-%d")
    end_dt = datetime.now(timezone.utc).replace(tzinfo=None)

    price_cache: dict[str, Optional[pd.DataFrame]] = {}
    for ticker in tickers:
        price_cache[ticker] = _fetch_prices(ticker, start_dt, end_dt)

    results = []
    for sig in signals:
        entry_low = float(sig["entry_zone_low"]) if sig["entry_zone_low"] else 0
        entry_high = float(sig["entry_zone_high"]) if sig["entry_zone_high"] else 0
        entry_price = round((entry_low + entry_high) / 2, 2) if (entry_low and entry_high) else 0
        stop = float(sig["stop_loss"]) if sig["stop_loss"] else 0
        target = float(sig["target"]) if sig["target"] else 0

        sig_date = datetime.strptime(sig["timestamp"][:10], "%Y-%m-%d")
        df = price_cache.get(sig["ticker"])

        outcome = "error"
        outcome_date = ""
        exit_price = 0.0
        pnl_pct = 0.0

        if df is not None and not df.empty and entry_price > 0:
            future = df[df.index > sig_date]

            if len(future) == 0:
                outcome = "open"
            else:
                window = future.iloc[:_HOLD_DAYS]
                resolved = False

                for i, (dt_idx, row) in enumerate(window.iterrows()):
                    hit_stop = stop > 0 and row["Low"] <= stop
                    hit_target = target > 0 and row["High"] >= target

                    if hit_stop and hit_target:
                        # Both on same day — conservative: check open
                        if row["Open"] <= stop:
                            outcome = "stop"
                            exit_price = stop
                        else:
                            outcome = "target"
                            exit_price = target
                        outcome_date = dt_idx.strftime("%Y-%m-%d")
                        resolved = True
                        break
                    elif hit_stop:
                        outcome = "stop"
                        exit_price = stop
                        outcome_date = dt_idx.strftime("%Y-%m-%d")
                        resolved = True
                        break
                    elif hit_target:
                        outcome = "target"
                        exit_price = target
                        outcome_date = dt_idx.strftime("%Y-%m-%d")
                        resolved = True
                        break

                if not resolved:
                    if len(future) < _HOLD_DAYS:
                        outcome = "open"
                    else:
                        outcome = "expired"
                        exit_price = round(float(window.iloc[-1]["Close"]), 2)
                        outcome_date = window.index[-1].strftime("%Y-%m-%d")

        if entry_price > 0 and exit_price > 0:
            pnl_pct = round((exit_price - entry_price) / entry_price * 100, 2)

        row = dict(sig)
        row["entry_price"] = entry_price
        row["outcome"] = outcome
        row["outcome_date"] = outcome_date
        row["exit_price"] = exit_price if exit_price else ""
        row["realized_pnl_pct"] = pnl_pct if outcome not in ("open", "error") else ""
        results.append(row)

    return results


def save_outcomes(results: list[dict], out_path: Optional[Path] = None) -> Path:
    path = Path(out_path) if out_path else _OUT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=OUT_FIELDNAMES)
        w.writeheader()
        w.writerows(results)
    return path


def compute_summary(results: list[dict]) -> dict:
    """Aggregate outcome statistics."""
    total = len(results)
    if total == 0:
        return {"total": 0}

    by_outcome = defaultdict(int)
    wins = []
    losses = []

    for r in results:
        by_outcome[r["outcome"]] += 1
        pnl = r.get("realized_pnl_pct")
        if pnl == "" or pnl is None:
            continue
        pnl = float(pnl)
        if r["outcome"] == "target":
            wins.append(pnl)
        elif r["outcome"] in ("stop", "expired"):
            losses.append(pnl)

    resolved = len(wins) + len(losses)
    gross_profit = sum(w for w in wins) if wins else 0
    gross_loss = abs(sum(l for l in losses)) if losses else 0

    return {
        "total": total,
        "target": by_outcome.get("target", 0),
        "stop": by_outcome.get("stop", 0),
        "expired": by_outcome.get("expired", 0),
        "open": by_outcome.get("open", 0),
        "error": by_outcome.get("error", 0),
        "resolved": resolved,
        "win_rate_pct": round(len(wins) / resolved * 100, 1) if resolved else 0,
        "avg_win_pct": round(np.mean(wins), 2) if wins else 0,
        "avg_loss_pct": round(np.mean(losses), 2) if losses else 0,
        "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss > 0 else float("inf") if gross_profit > 0 else 0,
        "expectancy_pct": round((sum(wins) + sum(losses)) / resolved, 2) if resolved else 0,
    }


def compute_by_confidence(results: list[dict]) -> list[dict]:
    """Break down outcomes by confidence bucket."""
    buckets = [(60, 62, "60-62"), (62, 64, "62-64"), (64, 66, "64-66"), (66, 100, "66+")]
    rows = []

    for lo, hi, label in buckets:
        subset = [r for r in results if lo <= float(r.get("confidence", 0)) < hi]
        if not subset:
            continue
        s = compute_summary(subset)
        s["bucket"] = label
        rows.append(s)

    return rows
