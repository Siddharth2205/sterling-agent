"""Standalone Sharadar API key health check.

Run:  python check_api_key.py

Reads the key from .env (NASDAQ_API_KEY or SHARADAR_API_KEY), makes ONE tiny request,
and tells you whether the key is responding. It never prints the key itself.
"""

import os
import sys

import requests

API = "https://api.sharadar.com/v1.0/data/stocks"


def load_key() -> str | None:
    # Try python-dotenv if available, else parse .env by hand.
    try:
        from dotenv import dotenv_values
        vals = dotenv_values(".env")
    except Exception:
        vals = {}
        if os.path.exists(".env"):
            for line in open(".env", encoding="utf-8"):
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    vals[k.strip()] = v.strip().strip('"').strip("'")
    return (vals.get("NASDAQ_API_KEY") or vals.get("SHARADAR_API_KEY")
            or os.getenv("NASDAQ_API_KEY") or os.getenv("SHARADAR_API_KEY"))


def main() -> int:
    key = load_key()
    if not key:
        print("[FAIL] No API key found. Add NASDAQ_API_KEY=your_key to the .env file.")
        return 1
    print(f"- Key found (length {len(key)}, ends '...{key[-4:]}'). Testing...")

    try:
        r = requests.get(API, params={"ticker": "AAPL", "limit": 1,
                                      "format": "json", "api_key": key}, timeout=30)
    except requests.RequestException as e:
        print(f"[FAIL] Could not reach the API (network error): {e}")
        return 1

    print(f"- HTTP status: {r.status_code}")

    if r.status_code == 200:
        rows = r.json().get("data", [])
        if rows:
            print(f"[OK]   API KEY IS WORKING — got {len(rows)} row for AAPL: {rows[0]}")
            return 0
        print("[WARN] Key authenticated but returned no data (plan/permissions?).")
        return 1
    if r.status_code in (401, 403):
        print("[FAIL] Key rejected (401/403) — invalid key or plan doesn't allow this endpoint.")
    elif r.status_code == 429:
        print("[FAIL] Rate-limited (429) — key is valid but temporarily throttled. Wait and retry.")
    else:
        print(f"[FAIL] Unexpected response ({r.status_code}). Body: {r.text[:200]}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
