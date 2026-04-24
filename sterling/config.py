"""Loads and validates environment configuration from .env."""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root (two levels up from this file)
_ROOT = Path(__file__).parent.parent
load_dotenv(_ROOT / ".env")


def _require(key: str) -> str:
    val = os.getenv(key, "").strip()
    if not val:
        print(f"[Sterling] ERROR: Required environment variable '{key}' is missing or empty.", file=sys.stderr)
        print(f"           Copy .env.example to .env and fill in all required values.", file=sys.stderr)
        sys.exit(1)
    return val


def _optional(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


# Resolved configuration — import these throughout the project
FINNHUB_API_KEY: str = ""
TELEGRAM_BOT_TOKEN: str = ""
TELEGRAM_CHAT_ID: str = ""
NEWSAPI_KEY: str = ""

_validated = False


def validate() -> None:
    """Call once at startup to load and validate all required env vars."""
    global FINNHUB_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, NEWSAPI_KEY, _validated
    FINNHUB_API_KEY = _require("FINNHUB_API_KEY")
    TELEGRAM_BOT_TOKEN = _require("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID = _require("TELEGRAM_CHAT_ID")
    NEWSAPI_KEY = _optional("NEWSAPI_KEY")
    _validated = True


def validate_optional() -> dict:
    """Load env vars without requiring Telegram/Finnhub — for backtest/offline use."""
    global FINNHUB_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, NEWSAPI_KEY, _validated
    FINNHUB_API_KEY = _optional("FINNHUB_API_KEY")
    TELEGRAM_BOT_TOKEN = _optional("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID = _optional("TELEGRAM_CHAT_ID")
    NEWSAPI_KEY = _optional("NEWSAPI_KEY")
    _validated = True
    missing = [k for k, v in {
        "FINNHUB_API_KEY": FINNHUB_API_KEY,
        "TELEGRAM_BOT_TOKEN": TELEGRAM_BOT_TOKEN,
        "TELEGRAM_CHAT_ID": TELEGRAM_CHAT_ID,
    }.items() if not v]
    return {"missing": missing, "ok": len(missing) == 0}
