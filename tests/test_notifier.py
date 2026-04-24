"""Tests for sterling/notifier.py — formatting, throttle logic."""

import sqlite3
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import responses as responses_lib

from sterling import notifier


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr("sterling.notifier._DB_PATH", tmp_path / "notifications.db")
    yield tmp_path / "notifications.db"


SAMPLE_SIGNAL = {
    "ticker": "SHOP.TO",
    "recommendation": "BUY",
    "confidence": 78,
    "thesis": "Strong technical setup with RSI recovering from oversold. Tight stop at 200-day SMA.",
    "entry_zone": [98.50, 101.20],
    "stop_loss": 92.00,
    "target": 118.00,
    "risk_reward": 2.4,
    "signals": {"technical": 80, "fundamental": 70, "sentiment": 65, "macro": 55, "insider": 60},
    "current_price": 100.00,
    "fx_warning": None,
    "timestamp": "2026-04-24T14:30:00+00:00",
}


class TestFormatMessage:
    def test_contains_ticker(self):
        msg = notifier._format_message(SAMPLE_SIGNAL)
        assert "SHOP.TO" in msg

    def test_contains_recommendation(self):
        msg = notifier._format_message(SAMPLE_SIGNAL)
        assert "BUY" in msg

    def test_contains_confidence(self):
        msg = notifier._format_message(SAMPLE_SIGNAL)
        assert "78" in msg

    def test_ends_with_disclaimer(self):
        msg = notifier._format_message(SAMPLE_SIGNAL)
        assert "Signal, not advice" in msg

    def test_fx_warning_included(self):
        signal = {**SAMPLE_SIGNAL, "fx_warning": "USD trade — add 1.5% FX drag."}
        msg = notifier._format_message(signal)
        assert "FX drag" in msg

    def test_entry_zone_included(self):
        msg = notifier._format_message(SAMPLE_SIGNAL)
        assert "98.50" in msg

    def test_stop_loss_included(self):
        msg = notifier._format_message(SAMPLE_SIGNAL)
        assert "92.00" in msg

    def test_sell_signal_formatting(self):
        signal = {**SAMPLE_SIGNAL, "recommendation": "SELL", "confidence": 20}
        msg = notifier._format_message(signal)
        assert "SELL" in msg
        assert "Signal, not advice" in msg


class TestThrottle:
    def test_first_alert_always_sends(self):
        assert notifier._should_send("SHOP.TO", 75) is True

    def test_second_alert_within_4h_throttled(self):
        notifier._record_sent("SHOP.TO", 75, "BUY")
        assert notifier._should_send("SHOP.TO", 76) is False

    def test_different_ticker_not_throttled(self):
        notifier._record_sent("SHOP.TO", 75, "BUY")
        assert notifier._should_send("ENB.TO", 75) is True

    def test_confidence_delta_overrides_throttle(self):
        notifier._record_sent("SHOP.TO", 75, "BUY")
        # Delta of 15+ should override 4h throttle
        assert notifier._should_send("SHOP.TO", 75 + 15) is True

    def test_confidence_delta_below_15_still_throttled(self):
        notifier._record_sent("SHOP.TO", 75, "BUY")
        assert notifier._should_send("SHOP.TO", 75 + 14) is False

    def test_negative_delta_also_overrides(self):
        notifier._record_sent("SHOP.TO", 75, "BUY")
        assert notifier._should_send("SHOP.TO", 75 - 15) is True

    def test_record_sent_persists(self):
        notifier._record_sent("BNS.TO", 65, "ACCUMULATE")
        conn = sqlite3.connect(str(notifier._DB_PATH))
        row = conn.execute("SELECT ticker, confidence, rec FROM alerts").fetchone()
        conn.close()
        assert row == ("BNS.TO", 65, "ACCUMULATE")


class TestSendSignal:
    @responses_lib.activate
    def test_sends_on_first_call(self):
        responses_lib.add(
            responses_lib.POST,
            "https://api.telegram.org/botTEST_TOKEN/sendMessage",
            json={"ok": True, "result": {"message_id": 1}},
        )
        result = notifier.send_signal(SAMPLE_SIGNAL, "TEST_TOKEN", "12345")
        assert result is True
        assert len(responses_lib.calls) == 1

    @responses_lib.activate
    def test_throttled_does_not_call_api(self):
        notifier._record_sent("SHOP.TO", 78, "BUY")
        result = notifier.send_signal(SAMPLE_SIGNAL, "TEST_TOKEN", "12345")
        assert result is False
        assert len(responses_lib.calls) == 0

    @responses_lib.activate
    def test_force_bypasses_throttle(self):
        notifier._record_sent("SHOP.TO", 78, "BUY")
        responses_lib.add(
            responses_lib.POST,
            "https://api.telegram.org/botTEST_TOKEN/sendMessage",
            json={"ok": True, "result": {"message_id": 2}},
        )
        result = notifier.send_signal(SAMPLE_SIGNAL, "TEST_TOKEN", "12345", force=True)
        assert result is True

    @responses_lib.activate
    def test_telegram_api_error_returns_false(self):
        responses_lib.add(
            responses_lib.POST,
            "https://api.telegram.org/botTEST_TOKEN/sendMessage",
            json={"ok": False, "description": "Unauthorized"},
            status=401,
        )
        result = notifier.send_signal(SAMPLE_SIGNAL, "TEST_TOKEN", "12345")
        assert result is False
