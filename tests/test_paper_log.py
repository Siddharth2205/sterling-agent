"""Tests for sterling/paper_log.py and the --paper flag in sterling analyze."""

import csv
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import responses as responses_lib

from sterling import paper_log, notifier


# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_SIGNAL = {
    "ticker": "SHOP.TO",
    "recommendation": "BUY",
    "confidence": 78.5,
    "entry_zone": [98.50, 101.20],
    "stop_loss": 92.00,
    "target": 118.00,
    "risk_reward": 2.4,
    "thesis": "Strong technical setup. Tight stop at 200-day SMA.",
    "signals": {"technical": 80, "fundamental": 70, "sentiment": 65, "macro": 55, "insider": 60},
    "current_price": 100.00,
    "fx_warning": None,
    "timestamp": "2026-04-27T13:30:00+00:00",
}

SAMPLE_CDR_SIGNAL = {
    **SAMPLE_SIGNAL,
    "ticker": "NVDA.NE",
    "recommendation": "ACCUMULATE",
    "confidence": 65.0,
    "fx_warning": "CAD-settled CDR (FX-hedged). No FX drag; check CDR premium/discount to NAV.",
}

SAMPLE_TRIM_SIGNAL = {
    **SAMPLE_SIGNAL,
    "ticker": "BCE.TO",
    "recommendation": "TRIM",
    "confidence": 35.0,
    "entry_zone": None,
    "stop_loss": None,
    "target": None,
}


# ── paper_log.log_signal ─────────────────────────────────────────────────────

class TestLogSignal:
    def test_creates_file_with_header(self, tmp_path):
        path = tmp_path / "paper_trades.csv"
        paper_log.log_signal(SAMPLE_SIGNAL, csv_path=path)
        assert path.exists()
        with open(path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert set(rows[0].keys()) == set(paper_log.FIELDNAMES)

    def test_appends_without_duplicate_header(self, tmp_path):
        path = tmp_path / "paper_trades.csv"
        paper_log.log_signal(SAMPLE_SIGNAL, csv_path=path)
        paper_log.log_signal(SAMPLE_SIGNAL, csv_path=path)
        with open(path) as f:
            content = f.read()
        # Header should appear exactly once
        assert content.count("timestamp") == 1
        rows = list(csv.DictReader(content.splitlines()))
        assert len(rows) == 2

    def test_correct_column_values(self, tmp_path):
        path = tmp_path / "paper_trades.csv"
        paper_log.log_signal(SAMPLE_SIGNAL, csv_path=path)
        with open(path) as f:
            row = list(csv.DictReader(f))[0]
        assert row["ticker"] == "SHOP.TO"
        assert row["recommendation"] == "BUY"
        assert float(row["confidence"]) == pytest.approx(78.5)
        assert float(row["entry_zone_low"]) == pytest.approx(98.50)
        assert float(row["entry_zone_high"]) == pytest.approx(101.20)
        assert float(row["stop_loss"]) == pytest.approx(92.00)
        assert float(row["target"]) == pytest.approx(118.00)
        assert row["thesis"] == SAMPLE_SIGNAL["thesis"]

    def test_position_size_is_250(self, tmp_path):
        path = tmp_path / "paper_trades.csv"
        paper_log.log_signal(SAMPLE_SIGNAL, csv_path=path)
        with open(path) as f:
            row = list(csv.DictReader(f))[0]
        assert float(row["hypothetical_position_size_cad"]) == pytest.approx(250.0)

    def test_timestamp_normalised(self, tmp_path):
        path = tmp_path / "paper_trades.csv"
        paper_log.log_signal(SAMPLE_SIGNAL, csv_path=path)
        with open(path) as f:
            row = list(csv.DictReader(f))[0]
        ts = row["timestamp"]
        assert ts.endswith("Z")
        assert "T" not in ts  # normalised to space separator

    def test_missing_entry_zone_logs_empty(self, tmp_path):
        path = tmp_path / "paper_trades.csv"
        paper_log.log_signal(SAMPLE_TRIM_SIGNAL, csv_path=path)
        with open(path) as f:
            row = list(csv.DictReader(f))[0]
        assert row["entry_zone_low"] == ""
        assert row["entry_zone_high"] == ""
        assert row["stop_loss"] == ""
        assert row["target"] == ""

    def test_cdr_ticker_logs_correctly(self, tmp_path):
        path = tmp_path / "paper_trades.csv"
        paper_log.log_signal(SAMPLE_CDR_SIGNAL, csv_path=path)
        with open(path) as f:
            row = list(csv.DictReader(f))[0]
        assert row["ticker"] == "NVDA.NE"
        assert row["recommendation"] == "ACCUMULATE"

    def test_returns_path(self, tmp_path):
        path = tmp_path / "paper_trades.csv"
        result = paper_log.log_signal(SAMPLE_SIGNAL, csv_path=path)
        assert result == path

    def test_creates_parent_directories(self, tmp_path):
        nested = tmp_path / "a" / "b" / "paper_trades.csv"
        paper_log.log_signal(SAMPLE_SIGNAL, csv_path=nested)
        assert nested.exists()

    def test_no_timestamp_in_signal_uses_current_time(self, tmp_path):
        path = tmp_path / "paper_trades.csv"
        signal = {**SAMPLE_SIGNAL}
        del signal["timestamp"]
        paper_log.log_signal(signal, csv_path=path)
        with open(path) as f:
            row = list(csv.DictReader(f))[0]
        assert row["timestamp"].endswith("Z")
        assert len(row["timestamp"]) == 20  # "YYYY-MM-DD HH:MM:SSZ"


# ── paper_log.read_log ────────────────────────────────────────────────────────

class TestReadLog:
    def test_empty_when_file_absent(self, tmp_path):
        result = paper_log.read_log(tmp_path / "missing.csv")
        assert result == []

    def test_returns_all_rows(self, tmp_path):
        path = tmp_path / "paper_trades.csv"
        paper_log.log_signal(SAMPLE_SIGNAL, csv_path=path)
        paper_log.log_signal(SAMPLE_CDR_SIGNAL, csv_path=path)
        rows = paper_log.read_log(path)
        assert len(rows) == 2
        assert rows[0]["ticker"] == "SHOP.TO"
        assert rows[1]["ticker"] == "NVDA.NE"

    def test_empty_file_returns_empty_list(self, tmp_path):
        path = tmp_path / "paper_trades.csv"
        path.touch()
        assert paper_log.read_log(path) == []


# ── notifier — paper mode formatting ─────────────────────────────────────────

class TestPaperFormatMessage:
    def test_paper_prefix_in_message(self):
        msg = notifier._format_message(SAMPLE_SIGNAL, paper=True)
        assert "PAPER TRADE" in msg

    def test_no_paper_prefix_without_flag(self):
        msg = notifier._format_message(SAMPLE_SIGNAL, paper=False)
        assert "PAPER TRADE" not in msg

    def test_paper_message_still_contains_ticker(self):
        msg = notifier._format_message(SAMPLE_SIGNAL, paper=True)
        assert "SHOP.TO" in msg

    def test_paper_message_still_contains_disclaimer(self):
        msg = notifier._format_message(SAMPLE_SIGNAL, paper=True)
        assert "Signal, not advice" in msg


class TestSendSignalPaperMode:
    @pytest.fixture(autouse=True)
    def temp_db(self, tmp_path, monkeypatch):
        monkeypatch.setattr("sterling.notifier._DB_PATH", tmp_path / "notifications.db")

    @responses_lib.activate
    def test_paper_send_uses_paper_message(self):
        captured = []

        def capture(request):
            import json
            body = json.loads(request.body)
            captured.append(body["text"])
            return (200, {}, '{"ok": true, "result": {"message_id": 1}}')

        responses_lib.add_callback(
            responses_lib.POST,
            "https://api.telegram.org/botTEST/sendMessage",
            callback=capture,
            content_type="application/json",
        )
        notifier.send_signal(SAMPLE_SIGNAL, "TEST", "123", paper=True)
        assert len(captured) == 1
        assert "PAPER TRADE" in captured[0]

    @responses_lib.activate
    def test_live_send_has_no_paper_prefix(self):
        captured = []

        def capture(request):
            import json
            body = json.loads(request.body)
            captured.append(body["text"])
            return (200, {}, '{"ok": true, "result": {"message_id": 2}}')

        responses_lib.add_callback(
            responses_lib.POST,
            "https://api.telegram.org/botTEST/sendMessage",
            callback=capture,
            content_type="application/json",
        )
        notifier.send_signal(SAMPLE_SIGNAL, "TEST", "123", paper=False)
        assert len(captured) == 1
        assert "PAPER TRADE" not in captured[0]


# ── CLI --paper flag ──────────────────────────────────────────────────────────

class TestAnalyzePaperFlag:
    """Integration-level tests: mock analyst + notifier, verify paper_log is called."""

    @pytest.fixture
    def mock_portfolio(self):
        with patch("sterling.portfolio.get_portfolio",
                   return_value={"holdings": {}, "watchlist": ["SHOP.TO"]}):
            yield

    @pytest.fixture
    def mock_analyze(self):
        with patch("sterling.analyst.analyze_portfolio",
                   return_value=[SAMPLE_SIGNAL]) as m:
            yield m

    @pytest.fixture
    def mock_config(self, monkeypatch):
        monkeypatch.setenv("FINNHUB_API_KEY", "test-key")
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "")

    def test_paper_flag_calls_log_signal(self, tmp_path, mock_portfolio, mock_analyze, mock_config):
        from click.testing import CliRunner
        from sterling.cli import cli

        csv_path = tmp_path / "paper_trades.csv"
        with patch("sterling.paper_log._CSV_PATH", csv_path):
            runner = CliRunner()
            result = runner.invoke(cli, ["analyze", "--paper", "--no-notify"])

        assert result.exit_code == 0, result.output
        assert csv_path.exists()
        rows = paper_log.read_log(csv_path)
        assert len(rows) == 1
        assert rows[0]["ticker"] == "SHOP.TO"

    def test_no_paper_flag_does_not_create_csv(self, tmp_path, mock_portfolio, mock_analyze, mock_config):
        from click.testing import CliRunner
        from sterling.cli import cli

        csv_path = tmp_path / "paper_trades.csv"
        with patch("sterling.paper_log._CSV_PATH", csv_path):
            runner = CliRunner()
            result = runner.invoke(cli, ["analyze", "--no-paper", "--no-notify"])

        assert result.exit_code == 0, result.output
        assert not csv_path.exists()

    def test_paper_output_includes_paper_mode_label(self, tmp_path, mock_portfolio, mock_analyze, mock_config):
        from click.testing import CliRunner
        from sterling.cli import cli

        csv_path = tmp_path / "paper_trades.csv"
        with patch("sterling.paper_log._CSV_PATH", csv_path):
            runner = CliRunner()
            result = runner.invoke(cli, ["analyze", "--paper", "--no-notify"])

        assert "PAPER MODE" in result.output or "PAPER" in result.output

    def test_hold_signal_not_logged(self, tmp_path, mock_portfolio, mock_config):
        """HOLD signals should not appear in the paper log."""
        from click.testing import CliRunner
        from sterling.cli import cli

        hold_signal = {**SAMPLE_SIGNAL, "recommendation": "HOLD", "confidence": 50.0}
        csv_path = tmp_path / "paper_trades.csv"
        with patch("sterling.analyst.analyze_portfolio", return_value=[hold_signal]):
            with patch("sterling.paper_log._CSV_PATH", csv_path):
                runner = CliRunner()
                runner.invoke(cli, ["analyze", "--paper", "--no-notify"])

        assert not csv_path.exists()


# ── CLI --watchlist-from-env flag ─────────────────────────────────────────────

class TestWatchlistFromEnv:
    """Tests for --watchlist-from-env: parses valid input, errors cleanly when
    the env var is absent, handles whitespace and mixed case."""

    @pytest.fixture
    def mock_config(self, monkeypatch):
        monkeypatch.setenv("FINNHUB_API_KEY", "test-key")
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "")

    def _run_with_watchlist(self, watchlist_value, mock_config, tmp_path, *, extra_args=None):
        from click.testing import CliRunner
        from sterling.cli import cli

        csv_path = tmp_path / "paper_trades.csv"
        captured_tickers = []

        def fake_analyze(tickers, api_key):
            captured_tickers.extend(tickers)
            return []

        env = {}
        if watchlist_value is not None:
            env["STERLING_WATCHLIST"] = watchlist_value

        args = ["analyze", "--paper", "--no-notify", "--watchlist-from-env"]
        if extra_args:
            args.extend(extra_args)

        with patch("sterling.analyst.analyze_portfolio", side_effect=fake_analyze):
            with patch("sterling.paper_log._CSV_PATH", csv_path):
                runner = CliRunner()
                result = runner.invoke(cli, args, env=env)

        return result, captured_tickers

    def test_parses_comma_separated_tickers(self, mock_config, tmp_path):
        result, tickers = self._run_with_watchlist(
            "SHOP.TO,ENB.TO,NVDA.NE", mock_config, tmp_path
        )
        assert result.exit_code == 0, result.output
        assert tickers == ["SHOP.TO", "ENB.TO", "NVDA.NE"]

    def test_strips_whitespace_around_commas(self, mock_config, tmp_path):
        result, tickers = self._run_with_watchlist(
            " SHOP.TO , ENB.TO , NVDA.NE ", mock_config, tmp_path
        )
        assert result.exit_code == 0, result.output
        assert tickers == ["SHOP.TO", "ENB.TO", "NVDA.NE"]

    def test_normalises_to_uppercase(self, mock_config, tmp_path):
        result, tickers = self._run_with_watchlist(
            "shop.to,enb.to", mock_config, tmp_path
        )
        assert result.exit_code == 0, result.output
        assert tickers == ["SHOP.TO", "ENB.TO"]

    def test_skips_empty_tokens_from_trailing_comma(self, mock_config, tmp_path):
        result, tickers = self._run_with_watchlist(
            "SHOP.TO,ENB.TO,", mock_config, tmp_path
        )
        assert result.exit_code == 0, result.output
        assert tickers == ["SHOP.TO", "ENB.TO"]

    def test_single_ticker(self, mock_config, tmp_path):
        result, tickers = self._run_with_watchlist("RY.TO", mock_config, tmp_path)
        assert result.exit_code == 0, result.output
        assert tickers == ["RY.TO"]

    def test_missing_env_var_exits_nonzero(self, mock_config, tmp_path):
        result, _ = self._run_with_watchlist(None, mock_config, tmp_path)
        assert result.exit_code != 0

    def test_missing_env_var_prints_clear_error(self, mock_config, tmp_path):
        result, _ = self._run_with_watchlist(None, mock_config, tmp_path)
        assert "STERLING_WATCHLIST" in result.output

    def test_empty_env_var_exits_nonzero(self, mock_config, tmp_path):
        result, _ = self._run_with_watchlist("", mock_config, tmp_path)
        assert result.exit_code != 0

    def test_whitespace_only_env_var_exits_nonzero(self, mock_config, tmp_path):
        result, _ = self._run_with_watchlist("   ", mock_config, tmp_path)
        assert result.exit_code != 0

    def test_env_watchlist_takes_precedence_over_portfolio(self, mock_config, tmp_path):
        """When --watchlist-from-env is set, portfolio.json is not consulted."""
        captured_tickers = []

        def fake_analyze(tickers, api_key):
            captured_tickers.extend(tickers)
            return []

        csv_path = tmp_path / "paper_trades.csv"
        with patch("sterling.analyst.analyze_portfolio", side_effect=fake_analyze):
            with patch("sterling.paper_log._CSV_PATH", csv_path):
                with patch("sterling.portfolio.get_portfolio",
                           return_value={"holdings": {"BCE.TO": {}}, "watchlist": ["T.TO"]}):
                    from click.testing import CliRunner
                    from sterling.cli import cli
                    runner = CliRunner()
                    result = runner.invoke(
                        cli,
                        ["analyze", "--paper", "--no-notify", "--watchlist-from-env"],
                        env={"STERLING_WATCHLIST": "SHOP.TO", "FINNHUB_API_KEY": "x"},
                    )

        assert result.exit_code == 0, result.output
        assert captured_tickers == ["SHOP.TO"]
        assert "BCE.TO" not in captured_tickers
        assert "T.TO" not in captured_tickers
