"""Append-only CSV log for paper-trade signals.

Every actionable signal (BUY/ACCUMULATE/SELL/TRIM) emitted during
--paper mode is recorded here. The log is the source of truth for the
60-day paper-trading evaluation; it is committed to the repo by the
GitHub Actions workflow after each run.
"""

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_CSV_PATH = Path(__file__).parent.parent / "data" / "paper_trades.csv"

FIELDNAMES = [
    "timestamp",
    "ticker",
    "recommendation",
    "confidence",
    "entry_zone_low",
    "entry_zone_high",
    "stop_loss",
    "target",
    "thesis",
    "hypothetical_position_size_cad",
]

# Mirrors live account constraints from analyst.py / CLAUDE.md
_CAPITAL_CAD = 1_000.0
_MAX_POSITIONS = 4
POSITION_SIZE_CAD = round(_CAPITAL_CAD / _MAX_POSITIONS, 2)  # $250 per slot


def log_signal(signal: dict, csv_path: Optional[Path] = None) -> Path:
    """Append one paper-trade record to the CSV log.

    Creates the file with a header row if it does not yet exist.
    Safe to call concurrently from sequential CI runs — each run
    appends; no record is ever overwritten.

    Returns the path written to.
    """
    path = Path(csv_path) if csv_path else _CSV_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    write_header = not path.exists() or path.stat().st_size == 0

    entry = signal.get("entry_zone") or []
    ts_raw = signal.get("timestamp") or datetime.now(timezone.utc).isoformat()
    # Normalise to "YYYY-MM-DD HH:MM:SSZ" regardless of input format
    ts_clean = str(ts_raw)[:19].replace("T", " ") + "Z"

    row = {
        "timestamp": ts_clean,
        "ticker": signal.get("ticker", ""),
        "recommendation": signal.get("recommendation", ""),
        "confidence": signal.get("confidence", 0),
        "entry_zone_low": entry[0] if len(entry) > 0 else "",
        "entry_zone_high": entry[1] if len(entry) > 1 else "",
        "stop_loss": signal.get("stop_loss", ""),
        "target": signal.get("target", ""),
        "thesis": signal.get("thesis", ""),
        "hypothetical_position_size_cad": POSITION_SIZE_CAD,
    }

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    return path


def read_log(csv_path: Optional[Path] = None) -> list[dict]:
    """Return all paper-trade records as a list of dicts. Empty list if no log exists."""
    path = Path(csv_path) if csv_path else _CSV_PATH
    if not path.exists() or path.stat().st_size == 0:
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))
