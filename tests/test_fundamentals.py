import datetime as dt
from sterling.research.fundamentals import fundamentals_as_of

def _rec(available_from, **kw):
    base = dict(available_from=available_from, eps_ttm=None, revenue_ttm=None,
                revenue_ttm_prior=None, net_income_ttm=None, total_equity=None,
                total_debt=None, shares=None, fcf_ttm=None)
    base.update(kw); return base

class TestFundamentalsAsOf:
    def test_picks_latest_known_and_no_lookahead(self):
        recs = [_rec("2023-03-31", eps_ttm=2.0), _rec("2024-03-30", eps_ttm=3.0)]
        # between filings -> uses the older one
        assert fundamentals_as_of(recs, dt.date(2024,1,15), 30.0)["pe"] == 15.0

    def test_empty_when_nothing_filed_yet(self):
        recs = [_rec("2024-03-30", eps_ttm=3.0)]
        assert fundamentals_as_of(recs, dt.date(2024,1,2), 30.0) == {}

    def test_no_records_is_empty(self):
        assert fundamentals_as_of([], dt.date(2024,1,1), 30.0) == {}

    def test_negative_eps_penalised(self):
        recs = [_rec("2023-03-31", eps_ttm=-1.5)]
        assert fundamentals_as_of(recs, dt.date(2023,6,1), 50.0)["pe"] < 0

    def test_derived_metrics(self):
        recs = [_rec("2023-03-31", eps_ttm=4.0, revenue_ttm=120.0, revenue_ttm_prior=100.0,
                     net_income_ttm=40.0, total_equity=400.0, total_debt=200.0,
                     shares=10.0, fcf_ttm=30.0)]
        f = fundamentals_as_of(recs, dt.date(2023,6,1), 40.0)
        assert round(f["revenue_growth"],3)==0.2 and round(f["debt_to_equity"],3)==0.5
        assert round(f["roe"],3)==0.1 and round(f["fcf_yield_pct"],2)==7.5
