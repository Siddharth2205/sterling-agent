"""Tests for sterling/diagnostics.py."""

import json
import csv
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sterling import diagnostics


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_trades(rows: list[dict]) -> pd.DataFrame:
    """Build a trades DataFrame with properly typed date columns."""
    df = pd.DataFrame(rows)
    if not df.empty:
        df["entry_date"] = pd.to_datetime(df["entry_date"]).dt.date
        df["exit_date"] = pd.to_datetime(df["exit_date"]).dt.date
    return df


def _make_bt_dir(tmp_path, stats: dict, trades: list[dict]) -> Path:
    """Write a minimal backtest directory for load_backtest tests."""
    bt = tmp_path / "backtest_test"
    bt.mkdir()
    with open(bt / "summary.json", "w") as f:
        json.dump(stats, f)
    if trades:
        with open(bt / "trades.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=trades[0].keys())
            w.writeheader()
            w.writerows(trades)
    return bt


# ── find_latest_backtest ──────────────────────────────────────────────────────

class TestFindLatestBacktest:
    def test_returns_most_recent(self, tmp_path):
        (tmp_path / "backtest_20240101_120000").mkdir()
        (tmp_path / "backtest_20240202_120000").mkdir()
        result = diagnostics.find_latest_backtest(tmp_path)
        assert result.name == "backtest_20240202_120000"

    def test_returns_none_when_empty(self, tmp_path):
        result = diagnostics.find_latest_backtest(tmp_path)
        assert result is None

    def test_single_dir(self, tmp_path):
        (tmp_path / "backtest_20230101_000000").mkdir()
        result = diagnostics.find_latest_backtest(tmp_path)
        assert result is not None


# ── load_backtest ─────────────────────────────────────────────────────────────

class TestLoadBacktest:
    def test_loads_stats_and_trades(self, tmp_path):
        stats = {"cagr_pct": 5.08, "total_trades": 135, "start_date": "2023-01-01", "end_date": "2026-01-01"}
        trades = [
            {"ticker": "RY.TO", "entry_date": "2023-03-01", "exit_date": "2023-03-20",
             "entry_price": 120.0, "exit_price": 125.0, "shares": 2,
             "pnl_cad": 10.0, "pnl_pct": 4.2, "days_held": 19, "exit_reason": "TARGET", "score": 70.5},
        ]
        bt = _make_bt_dir(tmp_path, stats, trades)
        loaded_stats, df = diagnostics.load_backtest(bt)
        assert loaded_stats["cagr_pct"] == pytest.approx(5.08)
        assert len(df) == 1
        assert df["ticker"].iloc[0] == "RY.TO"

    def test_empty_trades(self, tmp_path):
        bt = tmp_path / "bt"
        bt.mkdir()
        with open(bt / "summary.json", "w") as f:
            json.dump({"cagr_pct": 0.0}, f)
        _, df = diagnostics.load_backtest(bt)
        assert df.empty

    def test_dates_parsed(self, tmp_path):
        stats = {"start_date": "2023-01-03", "end_date": "2023-12-29"}
        trades = [
            {"ticker": "TD.TO", "entry_date": "2023-03-01", "exit_date": "2023-04-01",
             "entry_price": 80.0, "exit_price": 82.0, "shares": 3,
             "pnl_cad": 6.0, "pnl_pct": 2.5, "days_held": 31, "exit_reason": "TIME", "score": 68.0},
        ]
        bt = _make_bt_dir(tmp_path, stats, trades)
        _, df = diagnostics.load_backtest(bt)
        assert isinstance(df["entry_date"].iloc[0], date)
        assert isinstance(df["exit_date"].iloc[0], date)


# ── cash_days_analysis ────────────────────────────────────────────────────────

class TestCashDaysAnalysis:
    def test_fully_in_cash_no_trades(self):
        df = _make_trades([])
        start = date(2024, 1, 2)
        end = date(2024, 1, 31)
        result = diagnostics.cash_days_analysis(df, start, end)
        assert result["pct_in_cash"] == 100.0
        assert result["days_with_positions"] == 0

    def test_fully_invested_all_month(self):
        # One trade spanning the whole test period
        df = _make_trades([{
            "ticker": "BNS.TO", "entry_date": "2024-01-01", "exit_date": "2024-02-01",
            "entry_price": 60.0, "exit_price": 62.0, "shares": 4,
            "pnl_cad": 8.0, "pnl_pct": 3.3, "days_held": 22, "exit_reason": "TARGET",
        }])
        start = date(2024, 1, 2)
        end = date(2024, 1, 31)
        result = diagnostics.cash_days_analysis(df, start, end)
        assert result["pct_in_cash"] == 0.0
        assert result["days_with_positions"] == result["total_trading_days"]

    def test_peak_positions_counted(self):
        df = _make_trades([
            {"ticker": "A", "entry_date": "2024-01-02", "exit_date": "2024-01-20",
             "entry_price": 10.0, "exit_price": 11.0, "shares": 1, "pnl_cad": 1.0, "pnl_pct": 10.0, "days_held": 13, "exit_reason": "TARGET"},
            {"ticker": "B", "entry_date": "2024-01-02", "exit_date": "2024-01-20",
             "entry_price": 20.0, "exit_price": 21.0, "shares": 1, "pnl_cad": 1.0, "pnl_pct": 5.0, "days_held": 13, "exit_reason": "TARGET"},
        ])
        start = date(2024, 1, 2)
        end = date(2024, 1, 31)
        result = diagnostics.cash_days_analysis(df, start, end)
        assert result["peak_simultaneous_positions"] == 2

    def test_partial_coverage(self):
        # Trade covers only first half of month
        df = _make_trades([{
            "ticker": "ENB.TO", "entry_date": "2024-01-02", "exit_date": "2024-01-12",
            "entry_price": 50.0, "exit_price": 51.0, "shares": 2, "pnl_cad": 2.0, "pnl_pct": 2.0, "days_held": 8, "exit_reason": "TIME",
        }])
        start = date(2024, 1, 2)
        end = date(2024, 1, 31)
        result = diagnostics.cash_days_analysis(df, start, end)
        assert result["days_with_positions"] > 0
        assert result["days_in_cash"] > 0
        assert result["total_trading_days"] == result["days_in_cash"] + result["days_with_positions"]


# ── score_distribution ────────────────────────────────────────────────────────

class TestScoreDistribution:
    def test_returns_none_without_score_column(self):
        df = _make_trades([
            {"ticker": "X", "entry_date": "2024-01-02", "exit_date": "2024-01-10",
             "entry_price": 10.0, "exit_price": 11.0, "shares": 1, "pnl_cad": 1.0, "pnl_pct": 10.0, "days_held": 8, "exit_reason": "TARGET"},
        ])
        assert diagnostics.score_distribution(df) is None

    def test_returns_none_for_empty_df(self):
        assert diagnostics.score_distribution(pd.DataFrame()) is None

    def test_correct_quartiles(self):
        scores = [65, 70, 72, 75, 80, 85, 66, 68]
        rows = [
            {"ticker": "T", "entry_date": "2024-01-02", "exit_date": "2024-01-10",
             "entry_price": 10.0, "exit_price": 11.0, "shares": 1, "pnl_cad": 1.0, "pnl_pct": 10.0, "days_held": 8, "exit_reason": "TARGET", "score": s}
            for s in scores
        ]
        df = _make_trades(rows)
        result = diagnostics.score_distribution(df)
        assert result is not None
        assert result["count"] == 8
        assert result["min"] == min(scores)
        assert result["max"] == max(scores)
        assert result["q1"] <= result["median"] <= result["q3"]

    def test_histogram_included(self):
        rows = [
            {"ticker": "T", "entry_date": "2024-01-02", "exit_date": "2024-01-10",
             "entry_price": 10.0, "exit_price": 11.0, "shares": 1, "pnl_cad": 1.0, "pnl_pct": 10.0, "days_held": 8, "exit_reason": "TARGET", "score": 70.0},
        ]
        df = _make_trades(rows)
        result = diagnostics.score_distribution(df)
        assert "histogram" in result
        assert isinstance(result["histogram"], str)
        assert len(result["histogram"]) > 0


# ── macro_overlay_analysis ────────────────────────────────────────────────────

class TestMacroOverlayAnalysis:
    def test_no_overlay_col_returns_zero_with_note(self):
        df = _make_trades([
            {"ticker": "A", "entry_date": "2024-01-02", "exit_date": "2024-01-10",
             "entry_price": 10.0, "exit_price": 11.0, "shares": 1, "pnl_cad": 1.0, "pnl_pct": 10.0, "days_held": 8, "exit_reason": "TARGET", "score": 70.0},
        ])
        result = diagnostics.macro_overlay_analysis(df)
        assert result["has_overlay_data"] is False
        assert result["overlay_applied_count"] == 0
        assert result["demoted_buy_to_lower"] == 0
        assert result["note"] is not None
        assert len(result["note"]) > 10

    def test_empty_df_returns_zero(self):
        result = diagnostics.macro_overlay_analysis(pd.DataFrame())
        assert result["overlay_applied_count"] == 0

    def test_with_overlay_columns_counts_correctly(self):
        rows = [
            # Was BUY (80), demoted to 65 → crossed below 75
            {"ticker": "A", "entry_date": "2024-01-02", "exit_date": "2024-01-10",
             "entry_price": 10.0, "exit_price": 11.0, "shares": 1, "pnl_cad": 1.0, "pnl_pct": 10.0, "days_held": 8, "exit_reason": "TARGET",
             "score": 65.0, "pre_overlay_score": 80.0},
            # Was ACCUMULATE (70), demoted to 55 → crossed below 75 but started below — not a BUY demotion
            {"ticker": "B", "entry_date": "2024-01-02", "exit_date": "2024-01-10",
             "entry_price": 20.0, "exit_price": 21.0, "shares": 1, "pnl_cad": 1.0, "pnl_pct": 5.0, "days_held": 8, "exit_reason": "TIME",
             "score": 55.0, "pre_overlay_score": 70.0},
            # No overlay fired
            {"ticker": "C", "entry_date": "2024-01-02", "exit_date": "2024-01-10",
             "entry_price": 30.0, "exit_price": 31.0, "shares": 1, "pnl_cad": 1.0, "pnl_pct": 3.3, "days_held": 8, "exit_reason": "TARGET",
             "score": 72.0, "pre_overlay_score": 72.0},
        ]
        df = _make_trades(rows)
        result = diagnostics.macro_overlay_analysis(df)
        assert result["has_overlay_data"] is True
        assert result["overlay_applied_count"] == 2   # A and B had overlay applied
        assert result["demoted_buy_to_lower"] == 1    # only A was pre_overlay >= 75 and post < 75


# ── cost_drag_analysis ────────────────────────────────────────────────────────

class TestCostDragAnalysis:
    def test_empty_df(self):
        result = diagnostics.cost_drag_analysis(pd.DataFrame())
        assert result["net_pnl_cad"] == 0.0
        assert result["gross_pnl_cad"] == 0.0
        assert result["total_slippage_cost_cad"] == 0.0

    def test_slippage_hurts_returns(self):
        # A profitable trade: raw_entry=100, raw_exit=110, 10 shares
        # effective_entry = 100 * 1.001 = 100.1
        # effective_exit = 110 * 0.999 = 109.889
        # net_pnl = (109.889 - 100.1) * 10 = 97.89
        # gross_pnl = (110 - 100) * 10 = 100.00
        df = _make_trades([{
            "ticker": "SHOP.TO", "entry_date": "2024-01-02", "exit_date": "2024-01-20",
            "entry_price": 100.1, "exit_price": 109.889, "shares": 10,
            "pnl_cad": 97.89, "pnl_pct": 9.78, "days_held": 13, "exit_reason": "TARGET",
        }])
        result = diagnostics.cost_drag_analysis(df, slippage_pct=0.001)
        assert result["gross_pnl_cad"] > result["net_pnl_cad"]
        assert result["total_slippage_cost_cad"] < 0  # slippage costs money

    def test_round_trip_drag_pct(self):
        df = _make_trades([{
            "ticker": "RY.TO", "entry_date": "2024-01-02", "exit_date": "2024-01-20",
            "entry_price": 120.12, "exit_price": 125.875, "shares": 2,
            "pnl_cad": 11.51, "pnl_pct": 4.78, "days_held": 13, "exit_reason": "TIME",
        }])
        result = diagnostics.cost_drag_analysis(df, slippage_pct=0.001)
        assert result["round_trip_drag_pct"] == pytest.approx(0.2, abs=0.001)

    def test_fx_drag_zero(self):
        df = _make_trades([{
            "ticker": "ENB.TO", "entry_date": "2024-01-02", "exit_date": "2024-01-20",
            "entry_price": 50.05, "exit_price": 51.0, "shares": 5,
            "pnl_cad": 4.75, "pnl_pct": 1.9, "days_held": 13, "exit_reason": "TIME",
        }])
        result = diagnostics.cost_drag_analysis(df)
        assert result["fx_drag_on_trades_cad"] == 0.0
        assert "FX drag" in result["fx_drag_note"]

    def test_total_trades_count(self):
        trades = [
            {"ticker": f"T{i}", "entry_date": "2024-01-02", "exit_date": "2024-01-10",
             "entry_price": 10.0, "exit_price": 11.0, "shares": 1, "pnl_cad": 1.0, "pnl_pct": 10.0, "days_held": 8, "exit_reason": "TARGET"}
            for i in range(5)
        ]
        df = _make_trades(trades)
        result = diagnostics.cost_drag_analysis(df)
        assert result["total_trades"] == 5


# ── run_diagnostics (integration) ────────────────────────────────────────────

class TestRunDiagnostics:
    def test_full_report_shape(self, tmp_path):
        stats = {
            "start_date": "2023-01-03", "end_date": "2023-06-30",
            "cagr_pct": 4.0, "total_trades": 3, "capital_cad": 1000.0,
            "years": 0.5, "benchmark_xic_cagr_pct": 20.0,
        }
        trades = [
            {"ticker": "BNS.TO", "entry_date": "2023-02-01", "exit_date": "2023-03-01",
             "entry_price": 65.065, "exit_price": 67.933, "shares": 3,
             "pnl_cad": 8.6, "pnl_pct": 4.41, "days_held": 28, "exit_reason": "TIME", "score": 70.0},
            {"ticker": "TD.TO", "entry_date": "2023-04-01", "exit_date": "2023-05-01",
             "entry_price": 85.085, "exit_price": 88.912, "shares": 2,
             "pnl_cad": 7.65, "pnl_pct": 4.5, "days_held": 30, "exit_reason": "TARGET", "score": 72.0},
            {"ticker": "RY.TO", "entry_date": "2023-05-10", "exit_date": "2023-06-10",
             "entry_price": 130.13, "exit_price": 127.873, "shares": 1,
             "pnl_cad": -2.26, "pnl_pct": -1.73, "days_held": 31, "exit_reason": "STOP", "score": 66.5},
        ]
        bt = _make_bt_dir(tmp_path, stats, trades)
        report = diagnostics.run_diagnostics(bt)

        assert "stats" in report
        assert "cash_exposure" in report
        assert "score_distribution" in report
        assert "macro_overlay" in report
        assert "cost_drag" in report

    def test_score_distribution_present_when_column_available(self, tmp_path):
        stats = {
            "start_date": "2023-01-03", "end_date": "2023-06-30",
            "capital_cad": 1000.0, "total_trades": 1,
        }
        trades = [
            {"ticker": "ENB.TO", "entry_date": "2023-02-01", "exit_date": "2023-03-01",
             "entry_price": 50.05, "exit_price": 52.0, "shares": 5,
             "pnl_cad": 9.75, "pnl_pct": 3.9, "days_held": 28, "exit_reason": "TIME", "score": 68.0},
        ]
        bt = _make_bt_dir(tmp_path, stats, trades)
        report = diagnostics.run_diagnostics(bt)
        assert report["score_distribution"] is not None
        assert report["score_distribution"]["count"] == 1

    def test_score_distribution_none_when_column_absent(self, tmp_path):
        stats = {
            "start_date": "2023-01-03", "end_date": "2023-06-30",
            "capital_cad": 1000.0, "total_trades": 1,
        }
        trades = [
            {"ticker": "ENB.TO", "entry_date": "2023-02-01", "exit_date": "2023-03-01",
             "entry_price": 50.05, "exit_price": 52.0, "shares": 5,
             "pnl_cad": 9.75, "pnl_pct": 3.9, "days_held": 28, "exit_reason": "TIME"},
        ]
        bt = _make_bt_dir(tmp_path, stats, trades)
        report = diagnostics.run_diagnostics(bt)
        assert report["score_distribution"] is None


# ── CLI smoke test ────────────────────────────────────────────────────────────
# CliRunner conflicts with the sys.stdout TextIOWrapper in cli.py on Windows
# when pytest's global capture is active; use subprocess for CLI integration.

class TestDiagnoseCLI:
    def test_cli_runs_against_bt_dir(self, tmp_path):
        import subprocess, sys

        stats = {
            "start_date": "2023-01-03", "end_date": "2023-06-30",
            "cagr_pct": 4.5, "total_trades": 2, "capital_cad": 1000.0,
            "years": 0.5, "benchmark_xic_cagr_pct": 18.0,
            "total_return_pct": 2.2,
        }
        trades = [
            {"ticker": "RY.TO", "entry_date": "2023-02-01", "exit_date": "2023-03-01",
             "entry_price": 120.12, "exit_price": 125.875, "shares": 2,
             "pnl_cad": 11.51, "pnl_pct": 4.8, "days_held": 28, "exit_reason": "TARGET", "score": 70.0},
            {"ticker": "BNS.TO", "entry_date": "2023-03-15", "exit_date": "2023-04-15",
             "entry_price": 65.065, "exit_price": 63.935, "shares": 3,
             "pnl_cad": -3.39, "pnl_pct": -1.73, "days_held": 31, "exit_reason": "STOP", "score": 67.5},
        ]
        bt = _make_bt_dir(tmp_path, stats, trades)

        result = subprocess.run(
            [sys.executable, "-m", "sterling.cli", "diagnose", "--dir", str(bt)],
            capture_output=True, text=True, encoding="utf-8",
        )
        combined = result.stdout + result.stderr
        assert result.returncode == 0, combined
        assert "CASH EXPOSURE" in combined
        assert "CONFIDENCE SCORE" in combined
        assert "MACRO OVERLAY" in combined
        assert "GROSS vs NET" in combined

    def test_cli_missing_dir_exits_zero_with_err_message(self, tmp_path):
        import subprocess, sys

        result = subprocess.run(
            [sys.executable, "-m", "sterling.cli", "diagnose", "--dir",
             str(tmp_path / "nonexistent")],
            capture_output=True, text=True, encoding="utf-8",
        )
        combined = result.stdout + result.stderr
        assert result.returncode == 0
        assert "ERR" in combined
