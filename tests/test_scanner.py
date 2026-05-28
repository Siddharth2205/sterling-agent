"""Tests for sterling scan -- universe, ranking, digest, pipeline isolation."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import responses as responses_lib

from sterling.scan_universe import (
    get_scan_universe,
    TSX60_TICKERS,
    TSX_MIDCAP_TICKERS,
    TSX_ETF_TICKERS,
)

MOCK_SIGNAL = {
    "ticker": "TEST.TO",
    "confidence": 70,
    "recommendation": "ACCUMULATE",
    "signals": {
        "technical": 75,
        "fundamental": 65,
        "macro": 60,
        "sentiment": 50,
        "insider": 50,
    },
    "thesis": "Strong technical setup with momentum. Macro backdrop is supportive.",
    "current_price": 50.0,
    "entry_zone": [49.0, 51.0],
    "stop_loss": 46.0,
    "target": 58.0,
    "risk_reward": 2.0,
    "fx_warning": None,
    "tradeable": True,
    "timestamp": "2026-05-27T14:00:00+00:00",
}


# ── Universe loading & deduplication ─────────────────────────────────────────

class TestScanUniverse:
    def test_loads_without_error(self):
        universe = get_scan_universe()
        assert isinstance(universe, list)
        assert len(universe) > 0

    def test_no_duplicates(self):
        universe = get_scan_universe()
        assert len(universe) == len(set(universe)), "Duplicate tickers in scan universe"

    def test_approximate_size(self):
        universe = get_scan_universe()
        assert 200 <= len(universe) <= 250, f"Expected ~220, got {len(universe)}"

    def test_contains_tsx60_tickers(self):
        universe = get_scan_universe()
        for t in ["SHOP.TO", "RY.TO", "ENB.TO", "CNR.TO"]:
            assert t in universe, f"TSX 60 ticker {t} missing"

    def test_contains_cdr_tickers(self):
        universe = get_scan_universe()
        for t in ["NVDA.NE", "AAPL.NE", "MSFT.NE"]:
            assert t in universe, f"CDR ticker {t} missing"

    def test_contains_etf_tickers(self):
        universe = get_scan_universe()
        for t in ["XIC.TO", "XIU.TO", "XEG.TO", "ZEB.TO"]:
            assert t in universe, f"ETF ticker {t} missing"

    def test_tsx60_no_internal_duplicates(self):
        assert len(TSX60_TICKERS) == len(set(TSX60_TICKERS))

    def test_midcap_no_internal_duplicates(self):
        assert len(TSX_MIDCAP_TICKERS) == len(set(TSX_MIDCAP_TICKERS))

    def test_etf_no_internal_duplicates(self):
        assert len(TSX_ETF_TICKERS) == len(set(TSX_ETF_TICKERS))

    def test_cdr_catalog_fully_included(self):
        from sterling.cdr_mapping import CDR_TICKERS
        universe = set(get_scan_universe())
        for cdr in CDR_TICKERS:
            assert cdr in universe, f"CDR {cdr} not in scan universe"


# ── Top-15 ranking ───────────────────────────────────────────────────────────

class TestTopSetups:
    def test_ranking_by_confidence(self):
        from sterling.scanner import top_setups

        results = [
            {**MOCK_SIGNAL, "ticker": "A.TO", "confidence": 60},
            {**MOCK_SIGNAL, "ticker": "B.TO", "confidence": 80},
            {**MOCK_SIGNAL, "ticker": "C.TO", "confidence": 70},
        ]
        sorted_results = sorted(results, key=lambda x: x["confidence"], reverse=True)
        top = top_setups(sorted_results, 15)

        assert [t["ticker"] for t in top] == ["B.TO", "C.TO", "A.TO"]

    def test_limits_to_n(self):
        from sterling.scanner import top_setups

        results = [
            {**MOCK_SIGNAL, "ticker": f"T{i}.TO", "confidence": 90 - i}
            for i in range(20)
        ]
        top = top_setups(results, 5)
        assert len(top) == 5

    def test_cdr_flagged(self):
        from sterling.scanner import top_setups

        results = [{**MOCK_SIGNAL, "ticker": "NVDA.NE"}]
        top = top_setups(results, 15)
        assert top[0]["is_cdr"] is True

    def test_tsx_not_flagged_as_cdr(self):
        from sterling.scanner import top_setups

        results = [{**MOCK_SIGNAL, "ticker": "SHOP.TO"}]
        top = top_setups(results, 15)
        assert top[0]["is_cdr"] is False

    def test_top_axis_identification(self):
        from sterling.scanner import top_setups

        results = [{
            **MOCK_SIGNAL,
            "signals": {"technical": 90, "fundamental": 60, "macro": 50,
                        "sentiment": 50, "insider": 50},
        }]
        top = top_setups(results, 15)
        assert top[0]["top_axis"] == "technical"

    def test_fundamental_axis_wins_when_highest(self):
        from sterling.scanner import top_setups

        results = [{
            **MOCK_SIGNAL,
            "signals": {"technical": 50, "fundamental": 90, "macro": 50,
                        "sentiment": 50, "insider": 50},
        }]
        top = top_setups(results, 15)
        assert top[0]["top_axis"] == "fundamental"

    def test_skips_error_entries(self):
        from sterling.scanner import top_setups

        results = [
            {**MOCK_SIGNAL, "ticker": "GOOD.TO", "confidence": 80},
            {"ticker": "BAD.TO", "error": "fetch failed", "confidence": 0},
        ]
        top = top_setups(results, 15)
        assert len(top) == 1
        assert top[0]["ticker"] == "GOOD.TO"


# ── CDR redirection in scan context ──────────────────────────────────────────

class TestCDRRedirectionInScan:
    def test_all_cdrs_in_universe(self):
        from sterling.cdr_mapping import CDR_TICKERS
        universe = set(get_scan_universe())
        for cdr in CDR_TICKERS:
            assert cdr in universe

    def test_cdr_underlying_resolution(self):
        from sterling.cdr_mapping import get_underlying
        assert get_underlying("NVDA.NE") == "NVDA"
        assert get_underlying("AAPL.NE") == "AAPL"
        assert get_underlying("SHOP.TO") == "SHOP.TO"


# ── Rate-limit pacing ────────────────────────────────────────────────────────

class TestRateLimitPacing:
    def test_sleeps_between_tickers(self):
        from sterling.scanner import run_scan

        sleep_calls = []
        small_universe = ["A.TO", "B.TO", "C.TO"]

        with patch("sterling.scanner.time.sleep",
                   side_effect=lambda s: sleep_calls.append(s)), \
             patch("sterling.scan_universe.get_scan_universe",
                   return_value=small_universe), \
             patch("sterling.analyst.analyze", return_value=MOCK_SIGNAL), \
             patch("sterling.data_feed.get_macro_indicators",
                   return_value={"risk_off": False, "oil_crash": False}):
            run_scan("fake_key", pace_seconds=2.0)

        assert len(sleep_calls) == 2  # n-1 sleeps for n tickers
        assert all(s == 2.0 for s in sleep_calls)

    def test_no_sleep_after_last_ticker(self):
        from sterling.scanner import run_scan

        sleep_calls = []

        with patch("sterling.scanner.time.sleep",
                   side_effect=lambda s: sleep_calls.append(s)), \
             patch("sterling.scan_universe.get_scan_universe",
                   return_value=["ONLY.TO"]), \
             patch("sterling.analyst.analyze", return_value=MOCK_SIGNAL), \
             patch("sterling.data_feed.get_macro_indicators",
                   return_value={"risk_off": False, "oil_crash": False}):
            run_scan("fake_key", pace_seconds=2.0)

        assert len(sleep_calls) == 0

    def test_progress_callback_fires(self):
        from sterling.scanner import run_scan

        progress_log = []
        universe = [f"T{i}.TO" for i in range(25)]

        with patch("sterling.scanner.time.sleep"), \
             patch("sterling.scan_universe.get_scan_universe",
                   return_value=universe), \
             patch("sterling.analyst.analyze", return_value=MOCK_SIGNAL), \
             patch("sterling.data_feed.get_macro_indicators",
                   return_value={"risk_off": False, "oil_crash": False}):
            run_scan("fake_key", pace_seconds=0,
                     progress_cb=lambda done, total: progress_log.append((done, total)))

        assert (10, 25) in progress_log
        assert (20, 25) in progress_log
        assert (25, 25) in progress_log  # final


# ── Digest message ───────────────────────────────────────────────────────────

class TestDigestMessage:
    def test_contains_research_only_prefix(self):
        from sterling.scanner import format_scan_digest

        top = [{"ticker": "SHOP.TO", "score": 80.0, "recommendation": "BUY",
                "top_axis": "technical", "note": "Strong setup", "is_cdr": False}]
        msg = format_scan_digest(top, "2026-05-27")
        assert "RESEARCH ONLY" in msg

    def test_contains_not_paper_trade_signals(self):
        from sterling.scanner import format_scan_digest

        top = [{"ticker": "SHOP.TO", "score": 80.0, "recommendation": "BUY",
                "top_axis": "technical", "note": "Strong setup", "is_cdr": False}]
        msg = format_scan_digest(top, "2026-05-27")
        assert "not paper-trade signals" in msg

    def test_contains_date(self):
        from sterling.scanner import format_scan_digest

        top = [{"ticker": "SHOP.TO", "score": 80.0, "recommendation": "BUY",
                "top_axis": "technical", "note": "Strong setup", "is_cdr": False}]
        msg = format_scan_digest(top, "2026-05-27")
        assert "2026-05-27" in msg

    def test_cdr_flagged_in_digest(self):
        from sterling.scanner import format_scan_digest

        top = [{"ticker": "NVDA.NE", "score": 80.0, "recommendation": "BUY",
                "top_axis": "technical", "note": "Strong setup", "is_cdr": True}]
        msg = format_scan_digest(top, "2026-05-27")
        assert "CAD-settled CDR" in msg

    def test_uses_html_tags(self):
        from sterling.scanner import format_scan_digest

        top = [{"ticker": "SHOP.TO", "score": 80.0, "recommendation": "BUY",
                "top_axis": "technical", "note": "Strong setup", "is_cdr": False}]
        msg = format_scan_digest(top, "2026-05-27")
        assert "<b>" in msg
        assert "<code>" in msg

    @responses_lib.activate
    def test_send_digest_uses_html_parse_mode(self):
        from sterling.scanner import send_scan_digest

        captured = []

        def capture(request):
            captured.append(json.loads(request.body))
            return (200, {}, '{"ok": true, "result": {"message_id": 1}}')

        responses_lib.add_callback(
            responses_lib.POST,
            "https://api.telegram.org/botTOK/sendMessage",
            callback=capture,
            content_type="application/json",
        )
        send_scan_digest("<b>Test</b>", "TOK", "123")

        assert len(captured) == 1
        assert captured[0]["parse_mode"] == "HTML"

    @responses_lib.activate
    def test_send_digest_returns_true_on_success(self):
        from sterling.scanner import send_scan_digest

        responses_lib.add(
            responses_lib.POST,
            "https://api.telegram.org/botTOK/sendMessage",
            json={"ok": True, "result": {"message_id": 1}},
        )
        assert send_scan_digest("test", "TOK", "123") is True

    @responses_lib.activate
    def test_send_digest_returns_false_on_failure(self):
        from sterling.scanner import send_scan_digest

        responses_lib.add(
            responses_lib.POST,
            "https://api.telegram.org/botTOK/sendMessage",
            json={"ok": False},
            status=400,
        )
        assert send_scan_digest("test", "TOK", "123") is False


# ── Pipeline isolation ───────────────────────────────────────────────────────

class TestScanIsolation:
    """Confirm scan never touches paper_trades.csv or the notification throttle DB."""

    def test_scanner_module_does_not_import_paper_log(self):
        import sterling.scanner as mod
        source = Path(mod.__file__).read_text(encoding="utf-8")
        assert "import paper_log" not in source
        assert "from sterling.paper_log" not in source
        assert "from sterling import paper_log" not in source

    def test_scanner_module_does_not_import_notifier(self):
        import sterling.scanner as mod
        source = Path(mod.__file__).read_text(encoding="utf-8")
        assert "import notifier" not in source
        assert "from sterling.notifier" not in source
        assert "from sterling import notifier" not in source
        assert "_record_sent(" not in source
        assert "_should_send(" not in source

    def test_scan_does_not_write_paper_trades(self, tmp_path):
        from sterling.scanner import run_scan

        csv_path = tmp_path / "paper_trades.csv"
        csv_path.write_text("sentinel\n")

        with patch("sterling.scan_universe.get_scan_universe",
                   return_value=["TEST.TO"]), \
             patch("sterling.analyst.analyze", return_value=MOCK_SIGNAL), \
             patch("sterling.data_feed.get_macro_indicators",
                   return_value={"risk_off": False, "oil_crash": False}), \
             patch("sterling.scanner.time.sleep"), \
             patch("sterling.paper_log._CSV_PATH", csv_path):
            run_scan("fake_key", pace_seconds=0)

        assert csv_path.read_text() == "sentinel\n"

    def test_scan_does_not_create_notifications_db(self, tmp_path):
        from sterling.scanner import run_scan

        db_path = tmp_path / "notifications.db"

        with patch("sterling.scan_universe.get_scan_universe",
                   return_value=["TEST.TO"]), \
             patch("sterling.analyst.analyze", return_value=MOCK_SIGNAL), \
             patch("sterling.data_feed.get_macro_indicators",
                   return_value={"risk_off": False, "oil_crash": False}), \
             patch("sterling.scanner.time.sleep"), \
             patch("sterling.notifier._DB_PATH", db_path):
            run_scan("fake_key", pace_seconds=0)

        assert not db_path.exists()
