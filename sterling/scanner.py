"""Market scanner -- read-only awareness tool, independent of the paper-trading pipeline.

Scans ~190 tickers across TSX 60, mid-caps, CIBC CDRs, and ETFs.
Produces a ranked top-15 table by confidence score.  Optionally sends
a Telegram digest.  Never writes to paper_trades.csv, the notifications
throttle DB, or any part of the analyze / backtest pipeline.
"""

import html as html_mod
import logging
import time
from typing import Callable, Optional

import requests
import yfinance as yf

logger = logging.getLogger(__name__)

DEFAULT_PACE_SECONDS = 2.0


def validate_universe(tickers: list[str]) -> tuple[list[str], list[str]]:
    """Batch-validate tickers via a quick yfinance download.

    Returns (valid, invalid).  On network error assumes all valid so the
    scan can still attempt them individually.
    """
    if not tickers:
        return [], []

    try:
        data = yf.download(tickers, period="5d", progress=False, threads=True)
    except Exception:
        return list(tickers), []

    if data.empty:
        return [], list(tickers)

    valid: list[str] = []
    invalid: list[str] = []

    if len(tickers) == 1:
        if not data["Close"].dropna().empty:
            valid.append(tickers[0])
        else:
            invalid.append(tickers[0])
        return valid, invalid

    for ticker in tickers:
        try:
            col = data["Close"][ticker]
            if col.dropna().empty:
                invalid.append(ticker)
            else:
                valid.append(ticker)
        except (KeyError, TypeError):
            invalid.append(ticker)

    return valid, invalid


def _top_axis(signals: dict) -> str:
    """Return the name of the highest-scoring weighted axis."""
    from sterling.analyst import WEIGHTS
    scored = {k: signals.get(k, 0) for k in WEIGHTS}
    return max(scored, key=scored.get)


def run_scan(
    finnhub_key: str,
    pace_seconds: float = DEFAULT_PACE_SECONDS,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    validated_universe: Optional[list[str]] = None,
) -> list[dict]:
    """Scan the full universe with rate-limit pacing.

    Returns all successful results sorted by confidence descending.
    Respects Finnhub free tier (60 calls/min) by sleeping *pace_seconds*
    between tickers.  Each ticker may trigger up to 2 Finnhub calls
    (sentiment + insider); 2.0 s gap keeps us at ~60 calls/min worst-case.

    Google Trends calls are disabled in the scan path to avoid 429 rate
    limits.  Retries are reduced to 1 (vs 3 for the trading path).

    Pass *validated_universe* to supply a pre-validated ticker list
    (skips the built-in validation step).
    """
    from sterling.scan_universe import get_scan_universe
    from sterling import data_feed

    if validated_universe is not None:
        universe = validated_universe
    else:
        universe = get_scan_universe()
        valid, invalid = validate_universe(universe)
        if invalid:
            logger.info("Skipping %d invalid tickers: %s", len(invalid), ", ".join(invalid))
        universe = valid

    data_feed.set_scan_mode(True)
    try:
        return _run_scan_inner(universe, finnhub_key, pace_seconds, progress_cb)
    finally:
        data_feed.set_scan_mode(False)


def _run_scan_inner(
    universe: list[str],
    finnhub_key: str,
    pace_seconds: float,
    progress_cb: Optional[Callable[[int, int], None]],
) -> list[dict]:
    from sterling import analyst, data_feed

    try:
        macro = data_feed.get_macro_indicators()
    except Exception:
        macro = {"risk_off": False, "oil_crash": False}

    energy_bases = {
        "ENB", "SU", "CVE", "CNQ", "TRP", "PPL", "WCP", "ARX",
        "PEY", "TVE", "XOM", "IMO", "BIR",
        "BTE", "OVV", "TOU", "VET", "FRU",
    }

    results: list[dict] = []
    total = len(universe)

    for i, ticker in enumerate(universe, 1):
        base = ticker.replace(".TO", "").replace(".V", "").replace(".NE", "")
        is_energy = base.upper() in energy_bases

        try:
            result = analyst.analyze(
                ticker, finnhub_key, macro=macro,
                is_energy=is_energy, skip_trends=True,
            )
            results.append(result)
        except Exception as e:
            logger.warning("Scan failed for %s: %s", ticker, e)

        if progress_cb and i % 10 == 0:
            progress_cb(i, total)

        if i < total and pace_seconds > 0:
            time.sleep(pace_seconds)

    if progress_cb:
        progress_cb(total, total)

    return sorted(results, key=lambda x: x.get("confidence", 0), reverse=True)


def top_setups(results: list, n: int = 15) -> list[dict]:
    """Extract top *n* setups with scan-specific display fields."""
    from sterling.cdr_mapping import is_cdr

    top: list[dict] = []
    for r in results[:n]:
        if "error" in r:
            continue
        signals = r.get("signals", {})
        axis = _top_axis(signals)
        thesis = r.get("thesis", "")
        note = thesis.split(". ")[0][:60] if thesis else ""

        top.append({
            "ticker": r["ticker"],
            "score": r.get("confidence", 0),
            "recommendation": r.get("recommendation", "HOLD"),
            "top_axis": axis,
            "note": note,
            "is_cdr": is_cdr(r["ticker"]),
        })
    return top


def format_scan_digest(top: list[dict], date_str: str) -> str:
    """Format a Telegram HTML digest message for the top setups."""
    esc = html_mod.escape

    lines = [
        f"\U0001f4ca <b>Sterling Scan — Top 15 setups [{esc(date_str)}]</b>",
        "",
        "⚠️ <b>RESEARCH ONLY — not paper-trade signals</b>",
        "",
    ]

    for i, entry in enumerate(top, 1):
        ticker = esc(entry["ticker"])
        rec = esc(entry["recommendation"])
        score = entry["score"]
        axis = esc(entry["top_axis"])
        note = esc(entry["note"])
        cdr_tag = " [CAD-settled CDR]" if entry.get("is_cdr") else ""

        lines.append(
            f"{i}. <b>{ticker}</b> — <code>{rec}</code> ({score:.1f}) — {axis}{cdr_tag}"
        )
        if note:
            lines.append(f"   <i>{note}</i>")

    lines += [
        "",
        f"\U0001f550 <code>{esc(date_str)}</code>",
    ]

    return "\n".join(lines)


def send_scan_digest(message: str, bot_token: str, chat_id: str) -> bool:
    """Send a scan digest via Telegram (HTML parse mode).

    Completely independent of notifier.send_signal -- does NOT read or
    write the notifications throttle DB.
    """
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "HTML",
    }
    try:
        resp = requests.post(url, json=payload, timeout=15)
        resp.raise_for_status()
        return resp.json().get("ok", False)
    except Exception as e:
        logger.error("Failed to send scan digest: %s", e)
        return False
