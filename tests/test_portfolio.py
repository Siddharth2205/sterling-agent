"""Tests for sterling/portfolio.py."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# Point portfolio at a temp directory so tests don't clobber real data
@pytest.fixture(autouse=True)
def temp_data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("sterling.portfolio._DATA_DIR", tmp_path)
    monkeypatch.setattr("sterling.portfolio._PORTFOLIO_FILE", tmp_path / "portfolio.json")
    yield tmp_path


from sterling import portfolio


class TestAddHolding:
    def test_add_new(self):
        h = portfolio.add_holding("SHOP.TO", 5, 98.50)
        assert h["ticker"] == "SHOP.TO"
        assert h["shares"] == 5
        assert h["avg_cost_cad"] == 98.50
        assert h["currency"] == "CAD"

    def test_normalises_ticker_uppercase(self):
        h = portfolio.add_holding("shop.to", 1, 50.0)
        assert h["ticker"] == "SHOP.TO"

    def test_overwrite_existing(self):
        portfolio.add_holding("ENB.TO", 10, 45.00)
        portfolio.add_holding("ENB.TO", 15, 48.00)
        port = portfolio.get_portfolio()
        assert port["holdings"]["ENB.TO"]["shares"] == 15

    def test_usd_currency_flag(self):
        h = portfolio.add_holding("AAPL", 2, 220.00, currency="USD")
        assert h["currency"] == "USD"


class TestRemoveHolding:
    def test_remove_existing(self):
        portfolio.add_holding("BNS.TO", 8, 60.00)
        assert portfolio.remove_holding("BNS.TO") is True
        assert "BNS.TO" not in portfolio.get_portfolio()["holdings"]

    def test_remove_nonexistent_returns_false(self):
        assert portfolio.remove_holding("FAKE.TO") is False


class TestUpdateHolding:
    def test_update_shares(self):
        portfolio.add_holding("CNR.TO", 3, 175.00)
        updated = portfolio.update_holding("CNR.TO", shares=6)
        assert updated["shares"] == 6
        assert updated["avg_cost_cad"] == 175.00

    def test_update_avg_cost(self):
        portfolio.add_holding("CNR.TO", 3, 175.00)
        updated = portfolio.update_holding("CNR.TO", avg_cost_cad=180.00)
        assert updated["avg_cost_cad"] == 180.00

    def test_update_nonexistent_returns_none(self):
        assert portfolio.update_holding("NOPE.TO") is None


class TestPortfolioSummary:
    def test_empty_portfolio(self):
        s = portfolio.portfolio_summary()
        assert s["positions"] == []
        assert s["total_cost_cad"] == 0.0
        assert s["position_count"] == 0

    def test_pnl_calculation(self):
        portfolio.add_holding("SHOP.TO", 10, 90.00)  # cost basis $900
        s = portfolio.portfolio_summary(current_prices={"SHOP.TO": 100.00})
        pos = s["positions"][0]
        assert pos["cost_basis"] == 900.00
        assert pos["current_value"] == 1000.00
        assert pos["unrealized_pnl"] == 100.00
        assert pos["pnl_pct"] == pytest.approx(11.11, abs=0.01)

    def test_weight_calculation(self):
        portfolio.add_holding("SHOP.TO", 10, 100.00)  # $1000
        portfolio.add_holding("ENB.TO", 10, 50.00)    # $500
        s = portfolio.portfolio_summary()
        weights = {p["ticker"]: p["weight_pct"] for p in s["positions"]}
        assert weights["SHOP.TO"] == pytest.approx(66.7, abs=0.2)
        assert weights["ENB.TO"] == pytest.approx(33.3, abs=0.2)

    def test_total_cost(self):
        portfolio.add_holding("A.TO", 10, 10.00)
        portfolio.add_holding("B.TO", 5, 20.00)
        s = portfolio.portfolio_summary()
        assert s["total_cost_cad"] == 200.00


class TestWatchlist:
    def test_add_to_watchlist(self):
        wl = portfolio.add_to_watchlist("ENB.TO")
        assert "ENB.TO" in wl

    def test_no_duplicates(self):
        portfolio.add_to_watchlist("ENB.TO")
        wl = portfolio.add_to_watchlist("ENB.TO")
        assert wl.count("ENB.TO") == 1

    def test_remove_from_watchlist(self):
        portfolio.add_to_watchlist("ENB.TO")
        wl = portfolio.remove_from_watchlist("ENB.TO")
        assert "ENB.TO" not in wl
