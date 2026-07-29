"""Telegram notifications for the live paper book.

Reuses your existing bot: reads TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID from .env.
Never raises — a missing key or network hiccup just skips the message.
"""

from __future__ import annotations

import logging
import os

import requests

logger = logging.getLogger(__name__)


def _creds() -> tuple[str | None, str | None]:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass
    return os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")


def send(text: str) -> bool:
    """Send an HTML message to your chat. Returns True on success, False if skipped/failed."""
    token, chat = _creds()
    if not token or not chat:
        logger.warning("Telegram creds missing (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID) — skipping")
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat, "text": text, "parse_mode": "HTML",
                  "disable_web_page_preview": True},
            timeout=10,
        )
        if r.status_code != 200:
            logger.warning(f"Telegram API {r.status_code}: {r.text[:150]}")
        return r.status_code == 200
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Telegram send failed: {e}")
        return False


def format_book(asof, book, top_n: int = 15) -> str:
    """Render a paper book as a mobile-friendly Telegram message."""
    lines = [
        f"📈 <b>Sterling paper book — {asof}</b>",
        f"{len(book)} names · market-neutral · sector-neutral · inverse-vol",
        "",
        f"<b>Top {min(top_n, len(book))} holdings:</b>",
    ]
    top = book.sort_values("weight", ascending=False).head(top_n)
    for r in top.itertuples(index=False):
        lines.append(f"• <b>{r.ticker}</b> ({r.sector}) — {r.weight * 100:.2f}%")
    lines += ["", "<i>Paper only — research, not advice.</i>"]
    return "\n".join(lines)


def format_evaluation(result: dict) -> str:
    """Render the ledger evaluation as a Telegram message."""
    books = result.get("books", [])
    if not books:
        return "📊 <b>Sterling</b> — no paper books to evaluate yet."
    lines = ["📊 <b>Sterling paper track record</b>", ""]
    for b in books:
        lines.append(f"• {b['rebalance_date']}: {b['book_return_pct']:+.2f}% "
                     f"({b['names']} names)")
    return "\n".join(lines)
