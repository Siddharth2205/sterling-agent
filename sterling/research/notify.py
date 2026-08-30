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
    asof = result.get("as_of")
    lines = [f"📊 <b>Sterling paper track record</b>"
             + (f" (to {asof})" if asof else ""), ""]
    for b in books:
        ew = b.get("equal_weight_pct")
        held = b.get("days_held")
        head = f"• <b>{b['rebalance_date']}</b>"
        head += f" ({held}d)" if held is not None else ""
        head += f": book <b>{b['book_return_pct']:+.2f}%</b>"
        if ew is not None:
            head += f"  vs equal-wt {ew:+.2f}%"
        lines.append(head)
    lines += ["", "<i>Return since each book's log date. Equal-wt = the same names, "
              "equally weighted — the honest baseline. Paper only, not advice.</i>"]
    return "\n".join(lines)


def format_experiment(report: dict, board=None) -> str:
    """Telegram summary of the autonomous search: trials so far + honest hold-out verdict."""
    if "error" in report:
        return f"🔬 <b>Sterling search</b>\n{report['error']}"
    lines = ["🔬 <b>Sterling — autonomous strategy search</b>",
             f"Trials so far: <b>{report.get('trials')}</b>"]
    hs, ht = report.get("holdout_sharpe"), report.get("holdout_t")
    ha = report.get("holdout_alpha_pct_yr")
    lines += [
        f"Best (by development score) on the <b>untouched hold-out</b>:",
        f"  • return/yr: <b>{ha}%</b>" if ha is not None else "  • return/yr: n/a",
        f"  • Sharpe: <b>{hs}</b>   • t-stat: <b>{ht}</b>",
        ("✅ <b>Honest deployable edge found</b>" if report.get("honest_deployable_edge")
         else "❌ <b>No deployable edge yet</b> (needs Sharpe&gt;0.5 &amp; t&gt;2 on hold-out)"),
        "",
        "<i>Hold-out = data the search never selected on. With many trials, only t&gt;2 there counts.</i>",
    ]
    return "\n".join(lines)
