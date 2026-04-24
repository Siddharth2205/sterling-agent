#!/usr/bin/env python3
"""
Interactive Telegram setup wizard.
Run this once to configure your bot token and chat ID.
"""

import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    print("Run: pip install requests")
    sys.exit(1)

ENV_FILE = Path(__file__).parent / ".env"


def _read_env() -> dict:
    if not ENV_FILE.exists():
        return {}
    result = {}
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            result[k.strip()] = v.strip()
    return result


def _write_env(env: dict) -> None:
    example = Path(__file__).parent / ".env.example"
    header = example.read_text() if example.exists() else ""
    # Build new .env from existing values plus updates
    lines = []
    written_keys = set()
    for line in header.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key in env:
                lines.append(f"{key}={env[key]}")
                written_keys.add(key)
            else:
                lines.append(line)
        else:
            lines.append(line)
    # Append any new keys not in example
    for k, v in env.items():
        if k not in written_keys:
            lines.append(f"{k}={v}")
    ENV_FILE.write_text("\n".join(lines) + "\n")


def _test_bot(token: str) -> bool:
    url = f"https://api.telegram.org/bot{token}/getMe"
    try:
        r = requests.get(url, timeout=10)
        data = r.json()
        if data.get("ok"):
            name = data["result"].get("first_name", "?")
            username = data["result"].get("username", "?")
            print(f"  ✓ Bot verified: {name} (@{username})")
            return True
        print(f"  ✗ Bot API error: {data.get('description', 'unknown')}")
        return False
    except Exception as e:
        print(f"  ✗ Network error: {e}")
        return False


def _poll_chat_id(token: str, timeout_seconds: int = 60) -> str | None:
    url = f"https://api.telegram.org/bot{token}/getUpdates"
    print(f"\n  Polling for your chat ID (you have {timeout_seconds}s)...")
    deadline = time.time() + timeout_seconds
    seen_ids = set()

    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=10)
            updates = r.json().get("result", [])
            for upd in updates:
                msg = upd.get("message") or upd.get("channel_post")
                if msg:
                    chat_id = str(msg["chat"]["id"])
                    if chat_id not in seen_ids:
                        seen_ids.add(chat_id)
                        return chat_id
        except Exception:
            pass
        time.sleep(3)

    return None


def main():
    print("\n" + "=" * 58)
    print("  Sterling — Telegram Setup Wizard")
    print("=" * 58)

    env = _read_env()

    # Step 1: Bot token
    print("""
Step 1 — Create a Telegram bot (takes ~60 seconds):
  1. Open Telegram and search for @BotFather
  2. Send:  /newbot
  3. Choose a name (e.g., "Sterling Signals")
  4. Choose a username ending in 'bot' (e.g., sterling_signals_bot)
  5. BotFather will reply with your token — looks like:
       123456789:ABCdefGHIjklMNOpqrSTUVwxyz
""")

    existing_token = env.get("TELEGRAM_BOT_TOKEN", "")
    if existing_token and not existing_token.startswith("your_"):
        print(f"  Found existing token: {existing_token[:15]}...")
        use_existing = input("  Use this token? [Y/n]: ").strip().lower()
        token = existing_token if use_existing != "n" else ""
    else:
        token = ""

    if not token:
        token = input("  Paste your bot token: ").strip()

    if not token:
        print("  No token provided. Exiting.")
        sys.exit(1)

    print("\n  Verifying bot token...")
    if not _test_bot(token):
        print("  Check the token and try again.")
        sys.exit(1)

    # Step 2: Chat ID
    print("""
Step 2 — Find your chat ID:
  1. Open Telegram
  2. Find your new bot (search by its username)
  3. Send it any message — e.g., "hello"
""")
    input("  Press Enter when you have sent the message...")

    chat_id = _poll_chat_id(token, timeout_seconds=60)

    if not chat_id:
        print("\n  Could not detect chat ID automatically.")
        chat_id = input("  Enter your chat ID manually (check @userinfobot): ").strip()

    if not chat_id:
        print("  No chat ID. Exiting.")
        sys.exit(1)

    print(f"\n  ✓ Chat ID detected: {chat_id}")

    # Step 3: Test message
    print("\nStep 3 — Sending test message...")
    test_url = f"https://api.telegram.org/bot{token}/sendMessage"
    test_payload = {
        "chat_id": chat_id,
        "text": "✅ *Sterling is connected\\.* You'll receive signals here\\.",
        "parse_mode": "MarkdownV2",
    }
    try:
        r = requests.post(test_url, json=test_payload, timeout=10)
        if r.json().get("ok"):
            print("  ✓ Test message sent — check your Telegram!")
        else:
            print(f"  ✗ Send failed: {r.json()}")
    except Exception as e:
        print(f"  ✗ Error: {e}")

    # Step 4: Write to .env
    env["TELEGRAM_BOT_TOKEN"] = token
    env["TELEGRAM_CHAT_ID"] = chat_id
    _write_env(env)
    print(f"\n  ✓ Saved to {ENV_FILE}")

    print("""
Setup complete. Next steps:
  1. Set your FINNHUB_API_KEY in .env (get one free at finnhub.io)
  2. Run:  sterling analyze
  3. Run:  sterling run    (starts the scheduler)
""")


if __name__ == "__main__":
    main()
