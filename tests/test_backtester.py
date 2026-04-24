"""Tests for sterling/backtester.py — determinism, stats correctness, file output."""

import json
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


@pytest.fixture
def mock_price_data():
    """Patch yfinance to return synthetic data for 3 tickers + XIC."""
    tickers = ["SHOP.TO", "ENB.TO", "BNS.TO"]
    data = {t: _make_price_df(seed=i * 10) for i, t in enumerate(tickers)}
    data["XIC.TO"] = _make_price_df(seed=99, start_price=30.0)

    def _fake_history(ticker, start, end, auto_adjust=True):
        return data.get(ticker, pd.DataFrame())

    with patch("sterling.backtester._get_history_safe") as mock_hist:
        def side_effect(ticker, start, end):
            return data.get(ticker)
        mock_hist.side_effect = side_effect
        yield data, tickers


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


class TestComputeStats:
    def test_basic_stats(self):
        from datetime import date, timedelta
        start = date(2022, 1, 3)
        n = 252
        dates = [start + timedelta(days=i) for i in range(n)]
        # Flat equity curve — known returns
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
        from datetime import date, timedelta
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
        from datetime import date, timedelta
        start = date(2022, 1, 3)
        dates = [start + timedelta(days=i) for i in range(100)]
        values = [1000.0] * 100
        stats = backtester._compute_stats(list(zip(dates, values)), [], 1000.0, [])
        assert stats["win_rate_pct"] == 0
        assert stats["total_trades"] == 0

    def test_beats_benchmark_flag(self):
        from datetime import date, timedelta
        start = date(2022, 1, 3)
        n = 756
        dates = [start + timedelta(days=i) for i in range(n)]
        values = [1000.0 * (1 + 0.0003) ** i for i in range(n)]   # ~10% CAGR
        bm_values = [(dates[i], 1000.0 * (1 + 0.0001) ** i) for i in range(n)]  # ~4% CAGR
        stats = backtester._compute_stats(
            list(zip(dates, values)), [], 1000.0, bm_values
        )
        assert stats["beats_benchmark"] is True


class TestSaveResults:
    def test_creates_output_files(self, tmp_path):
        from datetime import date, timedelta
        from sterling.backtester import BacktestResult
        result = BacktestResult()
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
            }
        ]
        result.stats = {
            "cagr_pct": 5.0, "sharpe": 0.8, "sortino": 1.1,
            "max_drawdown_pct": -12.0, "win_rate_pct": 55.0,
            "beats_benchmark": True, "total_trades": 1,
        }

        out_dir = backtester.save_results(result, output_dir=tmp_path / "bt")
        assert (out_dir / "equity_curve.png").exists()
        assert (out_dir / "stats.csv").exists()
        assert (out_dir / "trades.csv").exists()
        assert (out_dir / "summary.json").exists()

    def test_summary_json_valid(self, tmp_path):
        from datetime import date, timedelta
        from sterling.backtester import BacktestResult
        result = BacktestResult()
        dates = [date(2022, 1, 3) + timedelta(days=i) for i in range(100)]
        result.equity_curve = [(d, 1000.0) for d in dates]
        result.benchmark_curve = []
        result.trades = []
        result.stats = {"cagr_pct": 3.0}
        out_dir = backtester.save_results(result, output_dir=tmp_path / "bt2")
        with open(out_dir / "summary.json") as f:
            data = json.load(f)
        assert data["cagr_pct"] == 3.0


class TestBacktestReproducibility:
    def test_same_seed_same_result(self, mock_price_data):
        data, tickers = mock_price_data
        with patch("sterling.backtester.TSX60_TICKERS", tickers), \
             patch("sterling.backtester._get_history_safe") as mock_hist:
            def side(ticker, start, end):
                return data.get(ticker)
            mock_hist.side_effect = side

            r1 = backtester.run_backtest(years=1, capital_cad=1000.0, seed=42, tickers=tickers)
            r2 = backtester.run_backtest(years=1, capital_cad=1000.0, seed=42, tickers=tickers)

        assert r1.stats.get("total_trades") == r2.stats.get("total_trades")
        if r1.equity_curve and r2.equity_curve:
            assert r1.equity_curve[-1][1] == r2.equity_curve[-1][1]
