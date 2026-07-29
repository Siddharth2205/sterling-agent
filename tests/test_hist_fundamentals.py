"""Tests for point-in-time fundamentals (look-ahead removal + neutral fallback)."""

import datetime as dt

from sterling import hist_fundamentals as hf
from sterling import analyst


def _record(period_end, available_from, **kw):
    base = {
        "period_end": period_end,
        "source": "annual",
        "available_from": available_from,
        "revenue_ttm": None,
        "revenue_ttm_prior": None,
        "net_income_ttm": None,
        "eps_ttm": None,
        "total_equity": None,
        "total_debt": None,
        "shares": None,
        "fcf_ttm": None,
    }
    base.update(kw)
    return base


class TestAsOfSelection:
    def test_picks_latest_record_available_by_date(self):
        recs = [
            _record("2022-12-31", "2023-03-31", eps_ttm=2.0, revenue_ttm=100, revenue_ttm_prior=90),
            _record("2023-12-31", "2024-03-30", eps_ttm=3.0, revenue_ttm=120, revenue_ttm_prior=100),
        ]
        # Between the two filing dates → must use the OLDER filing, not the newer one.
        f = hf.fundamentals_as_of(recs, dt.date(2024, 1, 15), price=30.0)
        assert f["pe"] == 15.0  # 30 / 2.0 (older eps), NOT 30/3.0

    def test_filing_lag_prevents_lookahead(self):
        # A statement dated 2023-12-31 is NOT usable the day after period end.
        recs = [_record("2023-12-31", "2024-03-30", eps_ttm=3.0)]
        assert hf.fundamentals_as_of(recs, dt.date(2024, 1, 2), price=30.0) == {}
        # ...but IS usable once the filing lag has elapsed.
        assert "pe" in hf.fundamentals_as_of(recs, dt.date(2024, 4, 1), price=30.0)

    def test_no_history_returns_empty_not_snapshot(self):
        assert hf.fundamentals_as_of([], dt.date(2024, 1, 1), price=30.0) == {}

    def test_empty_fundamentals_score_is_neutral(self):
        # The whole point: missing history → neutral 50, never today's snapshot.
        assert analyst.score_fundamental({}) == 50.0


class TestMetricReconstruction:
    def test_negative_eps_yields_penalised_pe(self):
        recs = [_record("2022-12-31", "2023-03-31", eps_ttm=-1.5)]
        f = hf.fundamentals_as_of(recs, dt.date(2023, 6, 1), price=50.0)
        assert f["pe"] < 0  # unprofitable → score_fundamental penalises

    def test_growth_debt_roe_fcf_from_statement(self):
        recs = [_record(
            "2022-12-31", "2023-03-31",
            eps_ttm=4.0, revenue_ttm=120.0, revenue_ttm_prior=100.0,
            net_income_ttm=40.0, total_equity=400.0, total_debt=200.0,
            shares=10.0, fcf_ttm=30.0,
        )]
        f = hf.fundamentals_as_of(recs, dt.date(2023, 6, 1), price=40.0)
        assert round(f["revenue_growth"], 3) == 0.2
        assert round(f["debt_to_equity"], 3) == 0.5
        assert round(f["roe"], 3) == 0.1
        # fcf_yield = 30 / (40 * 10) * 100 = 7.5
        assert round(f["fcf_yield_pct"], 2) == 7.5

    def test_eps_falls_back_to_net_income_over_shares(self):
        recs = [_record(
            "2022-12-31", "2023-03-31",
            eps_ttm=None, net_income_ttm=50.0, shares=25.0,
        )]
        f = hf.fundamentals_as_of(recs, dt.date(2023, 6, 1), price=20.0)
        assert round(f["pe"], 3) == 10.0  # 20 / (50/25 = 2.0)
