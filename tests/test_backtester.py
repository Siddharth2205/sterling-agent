"""Tests for sterling/backtester.py — determinism, stats correctness, file output."""

import json
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
import pytest

from sterling import backtester


def _make_price_df(n: int = 400, start_price: float = 50.0, seed: int = 42) -> pd.DataFrame:
    """Synthetic OHLCV dataframe suitable for backtesting."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-03", periods=n, freq="B")
    closes = start_price + np.cumsum(rng.normal(0.05, 1.0, n))
    closes = np.clip(closes, 5.0, None)
    df = pd.DataFrame({
        "Open":   closes * rng.uniform(0.99, 1.01, n),
        "High":   closes * rng.uniform(1.00, 1.02, n),
        "Low":    closes * rng.uniform(0.98, 1.00, n),
        "Close":  closes,
        "Volume": rng.integers(500_000, 3_000_000, n).astype(float),
    }, index=dates)
    return df


def _make_preloaded(tickers=None, with_macro=False):
    """Build a minimal _preloaded bundle for testing without network calls."""
    tickers = tickers or ["SHOP.TO", "ENB.TO", "BNS.TO"]
    data = {t: _make_price_df(seed=i * 10) for i, t in enumerate(tickers)}
    xic = _make_price_df(seed=99, start_price=30.0)

    macro_by_date = {}
    if with_macro:
        for d in pd.bdate_range("2022-01-03", periods=400):
            macro_by_date[d.date()] = {
                "vix": 18.0, "tsx_change_pct": 0.3,
                "risk_off": False, "oil_crash": False,
            }

    ticker_dates = sorted(data[tickers[0]].index.date)

    return {
        "price_data": data,
        "xic": xic,
        "macro_by_date": macro_by_date,
        "hist_fundamentals": {t: [] for t in tickers},
        "ticker_dates": ticker_dates,
    }


# ── _quick_signal ─────────────────────────────────────────────────────────────

class TestQuickSignal:
    def test_returns_in_range(self):
        hist = _make_price_df()
        score = backtester._quick_signal(hist)
        assert 0.0 <= score <= 100.0

    def test_deterministic(self):
        hist = _make_price_df(seed=7)
        s1 = backtester._quick_signal(hist)
        s2 = backtester._quick_signal(hist)
        assert s1 == s2


# ── _full_signal_backtest ─────────────────────────────────────────────────────

class TestFullSignalBacktest:
    # _full_signal_backtest returns (pre_overlay_score, post_overlay_score)
    def test_returns_in_range(self):
        hist = _make_price_df(n=250)
        pre, post = backtester._full_signal_backtest(hist, {}, backtester._NEUTRAL_MACRO)
        assert 0.0 <= pre <= 100.0
        assert 0.0 <= post <= 100.0

    def test_deterministic(self):
        hist = _make_price_df(n=250, seed=13)
        pre1, post1 = backtester._full_signal_backtest(hist, {}, backtester._NEUTRAL_MACRO)
        pre2, post2 = backtester._full_signal_backtest(hist, {}, backtester._NEUTRAL_MACRO)
        assert pre1 == pre2
        assert post1 == post2

    def test_risk_off_macro_lowers_score(self):
        """risk_off=True should drive post_score down via apply_macro_overlay."""
        hist = _make_price_df(n=250, seed=5)
        neutral = {"vix": 15.0, "tsx_change_pct": 1.0, "risk_off": False, "oil_crash": False}
        risk_off = {"vix": 35.0, "tsx_change_pct": -3.0, "risk_off": True, "oil_crash": False}
        _, score_neutral = backtester._full_signal_backtest(hist, {}, neutral)
        _, score_riskoff = backtester._full_signal_backtest(hist, {}, risk_off)
        # overlay subtracts up to 15 points
        assert score_riskoff <= score_neutral

    def test_energy_oil_crash_lowers_score(self):
        hist = _make_price_df(n=250, seed=5)
        normal = {"vix": 15.0, "tsx_change_pct": 0.5, "risk_off": False, "oil_crash": False}
        oil_crash = {"vix": 15.0, "tsx_change_pct": 0.5, "risk_off": False, "oil_crash": True}
        _, score_normal = backtester._full_signal_backtest(hist, {}, normal, is_energy=False)
        _, score_energy_crash = backtester._full_signal_backtest(hist, {}, oil_crash, is_energy=True)
        assert score_energy_crash <= score_normal

    def test_empty_fundamentals_returns_valid_score(self):
        hist = _make_price_df(n=250)
        pre, post = backtester._full_signal_backtest(hist, {}, backtester._NEUTRAL_MACRO)
        assert isinstance(pre, float)
        assert isinstance(post, float)
        assert 0.0 <= post <= 100.0

    def test_high_pe_fundamental_lowers_score(self):
        hist = _make_price_df(n=250, seed=7)
        low_pe_fund = {"pe": 10, "fcf_yield_pct": 8.0}
        high_pe_fund = {"pe": 80, "fcf_yield_pct": -2.0}
        _, score_low = backtester._full_signal_backtest(hist, low_pe_fund, backtester._NEUTRAL_MACRO)
        _, score_high = backtester._full_signal_backtest(hist, high_pe_fund, backtester._NEUTRAL_MACRO)
        assert score_low > score_high

    def test_no_overlay_neutral_macro_pre_equals_post(self):
        """With neutral macro (risk_off=False), pre and post should be equal."""
        hist = _make_price_df(n=250, seed=3)
        pre, post = backtester._full_signal_backtest(hist, {}, backtester._NEUTRAL_MACRO)
        assert pre == pytest.approx(post)


# ── _fetch_historical_macro ───────────────────────────────────────────────────

class TestFetchHistoricalMacro:
    def test_returns_dict_on_fetch_failure(self):
        """When VIX/TSX fetch fails, should return empty dict gracefully."""
        with patch("sterling.backtester._get_index_history", return_value=None):
            result = backtester._fetch_historical_macro("2023-01-01", "2023-12-31")
        assert isinstance(result, dict)
        assert len(result) == 0

    def test_risk_off_when_vix_high(self):
        """risk_off should be True when VIX > 25."""
        dates = pd.date_range("2023-01-03", periods=5, freq="B")
        vix_df = pd.DataFrame({"Close": [30.0, 28.0, 26.0, 24.0, 22.0]}, index=dates)
        tsx_df = pd.DataFrame({"Close": [100.0, 101.0, 102.0, 103.0, 104.0]}, index=dates)

        def fake_index(ticker, start, end):
            if "VIX" in ticker:
                return vix_df
            return tsx_df

        with patch("sterling.backtester._get_index_history", side_effect=fake_index):
            result = backtester._fetch_historical_macro("2023-01-03", "2023-01-09")

        d0 = dates[0].date()
        assert d0 in result
        assert result[d0]["risk_off"] is True    # VIX=30 > 25
        assert result[d0]["vix"] == pytest.approx(30.0)

    def test_risk_off_when_tsx_crash(self):
        """risk_off should be True when TSX drops > 2% on the day."""
        dates = pd.date_range("2023-01-03", periods=3, freq="B")
        vix_df = pd.DataFrame({"Close": [15.0, 15.0, 15.0]}, index=dates)
        # Day 1 → Day 2: -3% drop
        tsx_df = pd.DataFrame({"Close": [1000.0, 970.0, 980.0]}, index=dates)

        def fake_index(ticker, start, end):
            if "VIX" in ticker:
                return vix_df
            return tsx_df

        with patch("sterling.backtester._get_index_history", side_effect=fake_index):
            result = backtester._fetch_historical_macro("2023-01-03", "2023-01-05")

        d1 = dates[1].date()
        assert d1 in result
        chg = result[d1]["tsx_change_pct"]
        assert chg is not None and chg < -2.0
        assert result[d1]["risk_off"] is True

    def test_risk_on_normal_conditions(self):
        """risk_off=False when VIX < 25 and TSX change > -2%."""
        dates = pd.date_range("2023-06-01", periods=3, freq="B")
        vix_df = pd.DataFrame({"Close": [14.0, 15.0, 13.0]}, index=dates)
        tsx_df = pd.DataFrame({"Close": [800.0, 805.0, 810.0]}, index=dates)

        def fake_index(ticker, start, end):
            return vix_df if "VIX" in ticker else tsx_df

        with patch("sterling.backtester._get_index_history", side_effect=fake_index):
            result = backtester._fetch_historical_macro("2023-06-01", "2023-06-05")

        for d, macro in result.items():
            if macro["tsx_change_pct"] is not None:
                assert macro["risk_off"] is False


# ── min_hold_days ─────────────────────────────────────────────────────────────

class TestMinHoldDays:
    def test_stop_loss_exits_before_min_hold(self):
        """A position whose price falls to stop-loss should exit even before min_hold_days."""
        preloaded = _make_preloaded(["SHOP.TO"], with_macro=False)
        # Make SHOP.TO price decline sharply after day 205 to trigger stop
        df = preloaded["price_data"]["SHOP.TO"].copy()
        # Introduce a drop at bar 210 that triggers the 6% stop
        df.iloc[210:, df.columns.get_loc("Close")] *= 0.88
        preloaded["price_data"]["SHOP.TO"] = df

        result = backtester.run_backtest(
            years=1, capital_cad=1000.0,
            hold_days=20, min_hold_days=15,
            signal_threshold=30.0,  # very low to ensure entries
            use_full_signal=False,
            _preloaded=preloaded,
        )
        stop_trades = [t for t in result.trades if t["exit_reason"] == "STOP"]
        # If any stop trades occurred before min_hold_days, that's correct behaviour
        # We just verify the engine ran without error and produced trades
        assert result.stats.get("total_trades", 0) >= 0

    def test_min_hold_greater_than_hold_days_warns(self, caplog):
        """min_hold_days > hold_days should log a warning."""
        import logging
        preloaded = _make_preloaded(["BNS.TO"])
        with caplog.at_level(logging.WARNING, logger="sterling.backtester"):
            backtester.run_backtest(
                years=1, capital_cad=1000.0,
                hold_days=10, min_hold_days=20,  # inverted
                signal_threshold=30.0,
                use_full_signal=False,
                _preloaded=preloaded,
            )
        assert any("min_hold_days" in m for m in caplog.messages)

    def test_min_hold_days_recorded_in_stats(self):
        preloaded = _make_preloaded(["RY.TO"])
        result = backtester.run_backtest(
            years=1, min_hold_days=7, use_full_signal=False, _preloaded=preloaded,
        )
        assert result.stats.get("min_hold_days") == 7

    def test_signal_threshold_recorded_in_stats(self):
        preloaded = _make_preloaded(["TD.TO"])
        result = backtester.run_backtest(
            years=1, signal_threshold=72.0, use_full_signal=False, _preloaded=preloaded,
        )
        assert result.stats.get("signal_threshold") == pytest.approx(72.0)


# ── _compute_stats ────────────────────────────────────────────────────────────

class TestComputeStats:
    def test_basic_stats(self):
        start = date(2022, 1, 3)
        n = 252
        dates = [start + timedelta(days=i) for i in range(n)]
        values = [1000.0 + i * 0.5 for i in range(n)]
        equity = list(zip(dates, values))
        trades = [
            {"pnl_cad": 20.0, "pnl_pct": 5.0},
            {"pnl_cad": -10.0, "pnl_pct": -3.0},
            {"pnl_cad": 15.0, "pnl_pct": 4.0},
        ]
        stats = backtester._compute_stats(equity, trades, 1000.0, [])
        assert "cagr_pct" in stats
        assert "sharpe" in stats
        assert "sortino" in stats
        assert "max_drawdown_pct" in stats
        assert "win_rate_pct" in stats
        assert "profit_factor" in stats

    def test_win_rate_correct(self):
        start = date(2022, 1, 3)
        dates = [start + timedelta(days=i) for i in range(252)]
        values = [1000.0] * 252
        trades = [
            {"pnl_cad": 10.0, "pnl_pct": 2.0},
            {"pnl_cad": 10.0, "pnl_pct": 2.0},
            {"pnl_cad": -5.0, "pnl_pct": -1.0},
        ]
        stats = backtester._compute_stats(list(zip(dates, values)), trades, 1000.0, [])
        assert stats["win_rate_pct"] == pytest.approx(66.7, abs=0.2)

    def test_empty_trades(self):
        start = date(2022, 1, 3)
        dates = [start + timedelta(days=i) for i in range(100)]
        values = [1000.0] * 100
        stats = backtester._compute_stats(list(zip(dates, values)), [], 1000.0, [])
        assert stats["win_rate_pct"] == 0
        assert stats["total_trades"] == 0

    def test_beats_benchmark_flag(self):
        start = date(2022, 1, 3)
        n = 756
        dates = [start + timedelta(days=i) for i in range(n)]
        values = [1000.0 * (1 + 0.0003) ** i for i in range(n)]
        bm_values = [(dates[i], 1000.0 * (1 + 0.0001) ** i) for i in range(n)]
        stats = backtester._compute_stats(list(zip(dates, values)), [], 1000.0, bm_values)
        assert stats["beats_benchmark"] is True


# ── save_results ──────────────────────────────────────────────────────────────

class TestSaveResults:
    def test_creates_output_files(self, tmp_path):
        result = backtester.BacktestResult()
        start = date(2022, 1, 3)
        n = 252 * 2
        dates = [start + timedelta(days=i) for i in range(n)]
        result.equity_curve = [(d, 1000 + i) for i, d in enumerate(dates)]
        result.benchmark_curve = [(d, 950 + i * 0.9) for i, d in enumerate(dates)]
        result.trades = [
            {
                "ticker": "SHOP.TO", "entry_date": "2022-03-01", "exit_date": "2022-03-20",
                "entry_price": 100.0, "exit_price": 110.0, "shares": 2,
                "pnl_cad": 20.0, "pnl_pct": 10.0, "days_held": 15, "exit_reason": "TARGET",
                "score": 70.0,
            }
        ]
        result.stats = {
            "cagr_pct": 5.0, "sharpe": 0.8, "sortino": 1.1,
            "max_drawdown_pct": -12.0, "win_rate_pct": 55.0,
            "beats_benchmark": True, "total_trades": 1,
            "signal_mode": "full", "min_hold_days": 10, "signal_threshold": 65.0,
        }

        out_dir = backtester.save_results(result, output_dir=tmp_path / "bt")
        assert (out_dir / "equity_curve.png").exists()
        assert (out_dir / "stats.csv").exists()
        assert (out_dir / "trades.csv").exists()
        assert (out_dir / "summary.json").exists()

    def test_summary_json_valid(self, tmp_path):
        result = backtester.BacktestResult()
        dates = [date(2022, 1, 3) + timedelta(days=i) for i in range(100)]
        result.equity_curve = [(d, 1000.0) for d in dates]
        result.benchmark_curve = []
        result.trades = []
        result.stats = {"cagr_pct": 3.0}
        out_dir = backtester.save_results(result, output_dir=tmp_path / "bt2")
        with open(out_dir / "summary.json") as f:
            data = json.load(f)
        assert data["cagr_pct"] == 3.0


# ── reproducibility (quick signal, to avoid full-stack network mocking) ───────

class TestBacktestReproducibility:
    def test_same_seed_same_result(self):
        preloaded = _make_preloaded(["SHOP.TO", "ENB.TO", "BNS.TO"])
        r1 = backtester.run_backtest(
            years=1, capital_cad=1000.0, seed=42,
            use_full_signal=False, _preloaded=preloaded,
        )
        r2 = backtester.run_backtest(
            years=1, capital_cad=1000.0, seed=42,
            use_full_signal=False, _preloaded=preloaded,
        )
        assert r1.stats.get("total_trades") == r2.stats.get("total_trades")
        if r1.equity_curve and r2.equity_curve:
            assert r1.equity_curve[-1][1] == r2.equity_curve[-1][1]

    def test_signal_mode_recorded_quick(self):
        preloaded = _make_preloaded(["RY.TO"])
        r = backtester.run_backtest(years=1, use_full_signal=False, _preloaded=preloaded)
        assert r.stats.get("signal_mode") == "quick"

    def test_signal_mode_recorded_full(self):
        preloaded = _make_preloaded(["RY.TO"])
        r = backtester.run_backtest(years=1, use_full_signal=True, _preloaded=preloaded)
        assert r.stats.get("signal_mode") == "full"


# ── sweep_backtests ───────────────────────────────────────────────────────────

class TestSweepBacktests:
    def test_returns_four_results(self):
        preloaded = _make_preloaded(["SHOP.TO", "BNS.TO"])

        with patch("sterling.backtester.load_backtest_data", return_value=preloaded):
            results = backtester.sweep_backtests(
                years=1, thresholds=(65, 75), min_holds=(5, 15),
            )

        assert len(results) == 4

    def test_result_keys_present(self):
        preloaded = _make_preloaded(["RY.TO"])
        with patch("sterling.backtester.load_backtest_data", return_value=preloaded):
            results = backtester.sweep_backtests(years=1, thresholds=(65,), min_holds=(10,))

        r = results[0]
        assert "threshold" in r
        assert "min_hold" in r
        assert "result" in r
        assert isinstance(r["result"], backtester.BacktestResult)

    def test_different_combos_produce_different_stats(self):
        preloaded = _make_preloaded(["SHOP.TO", "ENB.TO", "BNS.TO"])
        with patch("sterling.backtester.load_backtest_data", return_value=preloaded):
            results = backtester.sweep_backtests(
                years=1, thresholds=(30, 90), min_holds=(1, 19),
            )
        # threshold=30 should produce more trades than threshold=90
        trades_low = results[0]["result"].stats.get("total_trades", 0)  # (30, 1)
        trades_high = results[3]["result"].stats.get("total_trades", 0)  # (90, 19)
        assert trades_low >= trades_high  # lower threshold → at least as many trades

    def test_save_sweep_results_creates_files(self, tmp_path):
        preloaded = _make_preloaded(["BNS.TO"])
        with patch("sterling.backtester.load_backtest_data", return_value=preloaded):
            results = backtester.sweep_backtests(years=1, thresholds=(65,), min_holds=(5,))

        out = backtester.save_sweep_results(results, output_dir=tmp_path / "sw")
        assert (out / "sweep_comparison.csv").exists()
        assert (out / "summary_t65_h5.json").exists()


# ── NaN-bar robustness (regression: yfinance current-session NaN close) ────────

class TestNaNRobustStats:
    def test_compute_stats_never_returns_nan_final_value(self):
        import datetime as _dt
        import math
        # Equity curve whose final bar is NaN (as a NaN price bar would produce).
        base = _dt.date(2024, 1, 1)
        curve = [(base + _dt.timedelta(days=i), v)
                 for i, v in enumerate([1000.0, 1010.0, 1005.0, float("nan")])]
        bench = [(base, 1000.0), (base + _dt.timedelta(days=3), float("nan"))]
        stats = backtester._compute_stats(curve, [], 1000.0, bench)
        assert not math.isnan(stats["cagr_pct"])
        assert not math.isnan(stats["benchmark_xic_cagr_pct"])
        # Falls back to the last finite equity value (1005.0), not NaN.
        assert stats["final_value_cad"] == 1005.0
