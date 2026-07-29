"""Tests for the edge validator (information coefficient, buckets, forward returns)."""

import numpy as np
import pandas as pd

from sterling import validate


def _bundle_one_ticker(closes, tk="AAA.TO", start="2024-01-01"):
    idx = pd.date_range(start, periods=len(closes), freq="B")
    df = pd.DataFrame({"Close": closes}, index=idx)
    return {"price_data": {tk: df}, "xic": df.copy()}, [d.date() for d in idx]


class TestForwardReturns:
    def test_fwd_return_matches_price_ratio(self):
        closes = np.linspace(100.0, 200.0, 40)  # strictly rising
        bundle, dates = _bundle_one_ticker(closes)
        records = [{"date": dates[0].isoformat(), "ticker": "AAA.TO", "post_score": 70}]
        df = validate.build_edge_frame(bundle, records, horizon=20)
        assert len(df) == 1
        expected = closes[20] / closes[0] - 1
        assert abs(df["fwd_return"].iloc[0] - expected) < 1e-9

    def test_horizon_beyond_data_is_dropped(self):
        closes = np.linspace(100.0, 110.0, 15)
        bundle, dates = _bundle_one_ticker(closes)
        records = [{"date": dates[0].isoformat(), "ticker": "AAA.TO", "post_score": 70}]
        df = validate.build_edge_frame(bundle, records, horizon=20)  # not enough bars
        assert df.empty


class TestInformationCoefficient:
    def test_positive_when_score_predicts_return(self):
        df = pd.DataFrame({"score": range(50), "fwd_return": [i * 0.001 for i in range(50)]})
        assert validate.information_coefficient(df) > 0.9

    def test_negative_when_score_is_backwards(self):
        df = pd.DataFrame({"score": range(50), "fwd_return": [-i * 0.001 for i in range(50)]})
        assert validate.information_coefficient(df) < -0.9

    def test_near_zero_for_noise(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"score": rng.normal(size=500), "fwd_return": rng.normal(size=500)})
        assert abs(validate.information_coefficient(df)) < 0.15


class TestBuckets:
    def test_monotonic_when_signal_present(self):
        df = pd.DataFrame({"score": range(100), "fwd_return": [i * 0.001 for i in range(100)]})
        b = validate.bucket_returns(df, n=4)
        rets = list(b["mean_fwd_return_pct"])
        assert rets == sorted(rets)          # Q1 < Q2 < Q3 < Q4
        assert rets[-1] > rets[0]


class TestWalkForward:
    def test_splits_into_periods_with_ic(self):
        rng = np.random.default_rng(1)
        n = 200
        dates = pd.date_range("2024-01-01", periods=n, freq="B").date
        df = pd.DataFrame({
            "date": dates,
            "score": rng.normal(size=n),
            "fwd_return": rng.normal(size=n),
        })
        wf = validate.walk_forward_ic(df, n_periods=4)
        assert len(wf) == 4
        assert set(["period", "start", "end", "obs", "ic"]).issubset(wf.columns)
