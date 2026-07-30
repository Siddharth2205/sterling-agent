"""Tests for the EDGAR fundamentals connector — the point-in-time rules that make
fundamentals safe to feed the model (earliest-filed dedupe, Q4 derivation, TTM
chaining, available_from stamping)."""

import datetime as dt

import pandas as pd
import pytest

from sterling.research import edgar
from sterling.research.fundamentals import fundamentals_as_of


def _q(metric, start, end, val, filed, tag=None, cik=1):
    return {"cik": cik, "metric": metric, "tag": tag or edgar.TAG_MAP.get(metric, ["t"])[0],
            "start": start, "end": end, "val": float(val), "filed": filed}


def _inst(metric, end, val, filed, tag=None, cik=1):
    return {"cik": cik, "metric": metric, "tag": tag or (edgar.TAG_MAP.get(metric, ["t"])[0]
            if metric != "shares" else edgar.SHARES_TAGS[0][1]),
            "start": None, "end": end, "val": float(val), "filed": filed}


def _year_of_quarters(metric, year, vals, tag=None):
    """Four discrete quarterly facts for one calendar year, filed ~40d after each end."""
    ends = [f"{year}-03-31", f"{year}-06-30", f"{year}-09-30", f"{year}-12-31"]
    starts = [f"{year}-01-01", f"{year}-04-01", f"{year}-07-01", f"{year}-10-01"]
    fileds = [f"{year}-05-10", f"{year}-08-09", f"{year}-11-09", f"{year + 1}-02-15"]
    return [_q(metric, s, e, v, f, tag=tag) for s, e, v, f in zip(starts, ends, vals, fileds)]


class TestQuarterlySeries:
    def test_q4_derived_from_annual(self):
        # Q1-Q3 discrete + FY annual → Q4 = FY − (Q1+Q2+Q3), filed with the 10-K
        rows = _year_of_quarters("revenue", 2020, [100, 110, 120, 130])[:3]
        rows.append(_q("revenue", "2020-01-01", "2020-12-31", 460, "2021-02-15"))
        facts = pd.DataFrame(rows)
        for c in ("start", "end", "filed"):
            facts[c] = pd.to_datetime(facts[c])
        qs = edgar._quarterly_series(facts)
        assert len(qs) == 4
        q4 = qs.iloc[-1]
        assert q4["val"] == 130                      # 460 − 330
        assert q4["filed"] == pd.Timestamp("2021-02-15")

    def test_amendment_does_not_rewrite_history(self):
        # Same period filed twice (original + amendment) → earliest filed wins
        rows = [_q("revenue", "2020-01-01", "2020-03-31", 100, "2020-05-10"),
                _q("revenue", "2020-01-01", "2020-03-31", 999, "2020-09-01")]
        facts = pd.DataFrame(rows)
        for c in ("start", "end", "filed"):
            facts[c] = pd.to_datetime(facts[c])
        qs = edgar._quarterly_series(facts)
        assert len(qs) == 1 and qs.iloc[0]["val"] == 100


class TestRecords:
    def _facts(self, years=(2019, 2020, 2021)):
        rows = []
        for y in years:
            rows += _year_of_quarters("net_income", y, [10, 10, 10, 10])
            rows += _year_of_quarters("revenue", y, [100, 100, 100, 100 + (y - 2019) * 40])
            rows += _year_of_quarters("ocf", y, [15, 15, 15, 15])
            rows += _year_of_quarters("capex", y, [5, 5, 5, 5])
            for qe in (f"{y}-03-31", f"{y}-06-30", f"{y}-09-30", f"{y}-12-31"):
                filed = (pd.Timestamp(qe) + pd.Timedelta(days=40)).date().isoformat()
                rows.append(_inst("equity", qe, 400, filed))
                rows.append(_inst("debt", qe, 200, filed))
                rows.append(_inst("shares", qe, 50, filed))
        return pd.DataFrame(rows)

    def test_ttm_and_available_from(self):
        recs = edgar.records_for_cik(self._facts())
        assert recs, "no records built"
        # A full-year TTM exists once 4 quarters chain; find the record for 2020-Q4
        r = [x for x in recs if x["available_from"] == "2021-02-15"]
        assert r, [x["available_from"] for x in recs]
        r = r[0]
        assert r["net_income_ttm"] == 40
        assert r["revenue_ttm"] == 440               # 100+100+100+140
        assert r["fcf_ttm"] == 40                    # OCF 60 − capex 20
        assert r["total_equity"] == 400 and r["total_debt"] == 200 and r["shares"] == 50
        # available_from must be the LATEST filed among components (the Q4 10-K)
        assert r["available_from"] == "2021-02-15"

    def test_revenue_growth_prior_ttm(self):
        recs = edgar.records_for_cik(self._facts())
        r = [x for x in recs if x["revenue_ttm"] == 480]     # 2021 TTM
        assert r and r[0]["revenue_ttm_prior"] == 440        # vs 2020 TTM

    def test_records_feed_fundamentals_as_of(self):
        recs = edgar.records_for_cik(self._facts())
        out = fundamentals_as_of(recs, dt.date(2021, 6, 1), price=20.0)
        # pe from ni/shares: eps=40/50=0.8 → pe=25 ; roe=40/400 ; d/e=0.5
        assert out["pe"] == pytest.approx(25.0)
        assert out["roe"] == pytest.approx(0.1)
        assert out["debt_to_equity"] == pytest.approx(0.5)
        assert out["fcf_yield_pct"] == pytest.approx(40 / (20.0 * 50) * 100)

    def test_no_lookahead_before_first_filing(self):
        recs = edgar.records_for_cik(self._facts())
        assert fundamentals_as_of(recs, dt.date(2019, 1, 1), price=20.0) == {}

    def test_gap_in_quarters_blocks_ttm(self):
        # Missing 2020-Q2 → no TTM may span the hole
        facts = self._facts(years=(2020,))
        facts = facts[~((facts["metric"] == "net_income") & (facts["end"] == "2020-06-30"))]
        recs = edgar.records_for_cik(facts)
        assert all(r["net_income_ttm"] is None for r in recs)

    def test_factor_metrics(self):
        rows = []
        for y in (2019, 2020):
            rows += _year_of_quarters("net_income", y, [10, 10, 10, 10 + (y - 2019) * 8])
            rows += _year_of_quarters("revenue", y, [100, 100, 100, 100])
            rows += _year_of_quarters("gross_profit", y, [40, 40, 40, 40])
            for qe in (f"{y}-03-31", f"{y}-06-30", f"{y}-09-30", f"{y}-12-31"):
                filed = (pd.Timestamp(qe) + pd.Timedelta(days=40)).date().isoformat()
                rows.append(_inst("assets", qe, 800 + (y - 2019) * 80, filed))
                rows.append(_inst("shares", qe, 50 - (y - 2019) * 2, filed))
        recs = edgar.records_for_cik(pd.DataFrame(rows))
        r = [x for x in recs if x["available_from"] == "2021-02-15"][0]
        from sterling.research.fundamentals import fundamentals_as_of
        f = fundamentals_as_of([r], dt.date(2021, 3, 1), price=20.0)
        assert f["gross_profitability"] == pytest.approx(160 / 880)
        assert f["asset_growth"] == pytest.approx(80 / 800)
        assert f["net_issuance"] == pytest.approx(-2 / 50)      # buyback
        assert f["margin_trend"] == pytest.approx(48 / 400 - 40 / 400)

    def test_gross_profit_falls_back_to_revenue_minus_cost(self):
        rows = (_year_of_quarters("net_income", 2020, [10, 10, 10, 10])
                + _year_of_quarters("revenue", 2020, [100, 100, 100, 100])
                + _year_of_quarters("cost_rev", 2020, [60, 60, 60, 60]))
        recs = edgar.records_for_cik(pd.DataFrame(rows))
        r = [x for x in recs if x["available_from"] == "2021-02-15"][0]
        assert r["gross_profit_ttm"] == 160                     # 400 − 240

    def test_stale_balance_sheet_is_not_used(self):
        # Last balance sheet is >400 days old at the anchor quarter → treated missing.
        rows = _year_of_quarters("net_income", 2021, [10, 10, 10, 10])
        rows.append(_inst("equity", "2019-12-31", 400, "2020-02-15"))
        recs = edgar.records_for_cik(pd.DataFrame(rows))
        assert all(r["total_equity"] is None for r in recs)

    def test_bank_no_capex_fcf_is_ocf(self):
        facts = self._facts()
        facts = facts[facts["metric"] != "capex"]            # bank: no capex tag at all
        recs = edgar.records_for_cik(facts)
        r = [x for x in recs if x["available_from"] == "2021-02-15"][0]
        assert r["fcf_ttm"] == 60                            # OCF, not None

    def test_tag_era_switch_is_merged(self):
        # Company reports under SalesRevenueNet through 2019, then switches to the
        # post-2018-standard tag — the two eras must merge into one full series.
        rows = []
        for y, tag in [(2019, "SalesRevenueNet"),
                       (2020, "RevenueFromContractWithCustomerExcludingAssessedTax")]:
            rows += _year_of_quarters("revenue", y, [100, 100, 100, 100], tag=tag)
            rows += _year_of_quarters("net_income", y, [10, 10, 10, 10])
        recs = edgar.records_for_cik(pd.DataFrame(rows))
        with_rev = [r for r in recs if r["revenue_ttm"] is not None]
        assert with_rev and with_rev[-1]["revenue_ttm"] == 400
        # a TTM window spanning the tag switch also works (2019H2 + 2020H1)
        assert any(r["revenue_ttm"] == 400 and r["available_from"].startswith("2020-08")
                   for r in recs)

    def test_cumulative_ytd_cashflow_is_differenced(self):
        # 10-Q cash-flow statements file YTD cumulatives (3/6/9/12mo), not discrete
        # quarters — Q_n must come from YTD_n − YTD_{n-1}.
        rows = _year_of_quarters("net_income", 2020, [10, 10, 10, 10])
        spans = [("2020-03-31", 15, "2020-05-10"), ("2020-06-30", 30, "2020-08-09"),
                 ("2020-09-30", 45, "2020-11-09"), ("2020-12-31", 60, "2021-02-15")]
        rows += [_q("ocf", "2020-01-01", e, v, f) for e, v, f in spans]
        recs = edgar.records_for_cik(pd.DataFrame(rows))
        r = [x for x in recs if x["available_from"] == "2021-02-15"]
        assert r and r[0]["fcf_ttm"] == 60           # 4 derived quarters of 15, no capex tag


class TestEdgeCases:
    def test_annual_only_metric_does_not_crash(self):
        # A filer with ONLY annual revenue facts (no quarters derivable) plus normal
        # net income — the empty quarterly series must not blow up record building.
        rows = _year_of_quarters("net_income", 2020, [10, 10, 10, 10])
        rows.append(_q("revenue", "2020-01-01", "2020-12-31", 400, "2021-02-15"))
        recs = edgar.records_for_cik(pd.DataFrame(rows))
        assert recs and all(r["revenue_ttm"] is None for r in recs)
        assert any(r["net_income_ttm"] == 40 for r in recs)


class TestExtraction:
    def test_facts_from_json_shape(self):
        obj = {"cik": 320193, "facts": {
            "us-gaap": {"NetIncomeLoss": {"units": {"USD": [
                {"start": "2020-01-01", "end": "2020-03-31", "val": 5,
                 "filed": "2020-05-01", "form": "10-Q"}]}}},
            "dei": {"EntityCommonStockSharesOutstanding": {"units": {"shares": [
                {"end": "2020-03-31", "val": 100, "filed": "2020-05-01"}]}}},
        }}
        rows = edgar._facts_from_json(obj)
        assert {r["metric"] for r in rows} == {"net_income", "shares"}
        assert all(r["cik"] == 320193 for r in rows)
