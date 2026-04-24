"""Telegram notification with SQLite-backed throttle."""

import json
import logging
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

_DB_PATH = Path(__file__).parent.parent / "data" / "notifications.db"
_THROTTLE_HOURS = 4
_CONFIDENCE_DELTA_OVERRIDE = 15


def _init_db() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            ticker      TEXT NOT NULL,
            sent_at     REAL NOT NULL,
            confidence  REAL NOT NULL,
            rec         TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


def _should_send(ticker: str, confidence: float) -> bool:
    """True if no alert for this ticker in last 4h, or confidence delta > 15 pts."""
    conn = _init_db()
    try:
        cutoff = time.time() - _THROTTLE_HOURS * 3600
        row = conn.execute(
            "SELECT sent_at, confidence FROM alerts WHERE ticker = ? ORDER BY sent_at DESC LIMIT 1",
            (ticker,)
        ).fetchone()

        if row is None:
            return True

        last_sent, last_confidence = row
        if last_sent < cutoff:
            return True

        delta = abs(confidence - last_confidence)
        if delta >= _CONFIDENCE_DELTA_OVERRIDE:
            logger.info(f"Confidence delta {delta:.1f} pts for {ticker} — overriding throttle")
            return True

        return False
    finally:
        conn.close()


def _record_sent(ticker: str, confidence: float, rec: str) -> None:
    conn = _init_db()
    try:
        conn.execute(
            "INSERT INTO alerts (ticker, sent_at, confidence, rec) VALUES (?, ?, ?, ?)",
            (ticker, time.time(), confidence, rec)
        )
        conn.commit()
    finally:
        conn.close()


def _format_message(signal: dict) -> str:
    """Format a signal dict into a Telegram-ready message (Markdown V2)."""
    ticker = signal.get("ticker", "???")
    rec = signal.get("recommendation", "HOLD")
    conf = signal.get("confidence", 0)
    thesis = signal.get("thesis", "No thesis available.")
    entry = signal.get("entry_zone")
    stop = signal.get("stop_loss")
    target = signal.get("target")
    rr = signal.get("risk_reward")
    price = signal.get("current_price")
    ts = signal.get("timestamp", datetime.now(timezone.utc).isoformat())
    fx_warn = signal.get("fx_warning")

    rec_emoji = {
        "BUY": "🟢",
        "ACCUMULATE": "🔵",
        "HOLD": "⚪",
        "TRIM": "🟡",
        "SELL": "🔴",
    }.get(rec, "⚪")

    scores = signal.get("signals", {})
    score_line = (
        f"T:{scores.get('technical', 0):.0f} "
        f"F:{scores.get('fundamental', 0):.0f} "
        f"S:{scores.get('sentiment', 0):.0f} "
        f"M:{scores.get('macro', 0):.0f} "
        f"I:{scores.get('insider', 0):.0f}"
    )

    lines = [
        f"*{rec_emoji} STERLING SIGNAL*",
        f"",
        f"*{ticker}* — `{rec}` | Confidence: *{conf:.0f}/100*",
        f"",
        f"📊 Axes: `{score_line}`",
        f"",
    ]

    if price:
        lines.append(f"💵 Price: `${price:.2f}`")
    if entry:
        lines.append(f"🎯 Entry zone: `${entry[0]:.2f} – ${entry[1]:.2f}`")
    if stop:
        lines.append(f"🛑 Stop: `${stop:.2f}`")
    if target:
        lines.append(f"🏁 Target: `${target:.2f}`")
    if rr:
        lines.append(f"⚖️  R:R: `{rr:.1f}x`")

    lines += [
        f"",
        f"📝 _{thesis}_",
    ]

    if fx_warn:
        lines += [f"", f"⚠️  _{fx_warn}_"]

    lines += [
        f"",
        f"🕐 `{ts[:16].replace('T', ' ')} UTC`",
        f"",
        f"_Signal, not advice\\. You decide\\._",
    ]

    return "\n".join(lines)


def send_signal(
    signal: dict,
    bot_token: str,
    chat_id: str,
    force: bool = False,
) -> bool:
    """
    Send a signal alert via Telegram.
    Returns True if sent, False if throttled or failed.
    """
    ticker = signal.get("ticker", "UNKNOWN")
    confidence = signal.get("confidence", 0)
    rec = signal.get("recommendation", "HOLD")

    if not force and not _should_send(ticker, confidence):
        logger.info(f"Throttled: {ticker} alert suppressed (within 4h window)")
        return False

    message = _format_message(signal)

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "MarkdownV2",
    }

    try:
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        result = resp.json()
        if result.get("ok"):
            _record_sent(ticker, confidence, rec)
            logger.info(f"Alert sent for {ticker} ({rec}, {confidence:.0f})")
            return True
        else:
            logger.error(f"Telegram API error: {result}")
            return False
    except Exception as e:
        logger.error(f"Failed to send Telegram alert for {ticker}: {e}")
        return False


def send_text(message: str, bot_token: str, chat_id: str) -> bool:
    """Send a plain text message (for status pings, errors, etc)."""
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}
    try:
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json().get("ok", False)
    except Exception as e:
        logger.error(f"Failed to send text message: {e}")
        return False


def get_my_chat_id(bot_token: str) -> Optional[str]:
    """Poll getUpdates to find the chat ID of the first message to the bot."""
    url = f"https://api.telegram.org/bot{bot_token}/getUpdates"
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        updates = data.get("result", [])
        if updates:
            return str(updates[-1]["message"]["chat"]["id"])
        return None
    except Exception as e:
        logger.error(f"getUpdates failed: {e}")
        return None
