"""Tests for the research pipeline: universe, dataset, features, labels."""

import numpy as np
import pandas as pd

from sterling.research import universe, dataset, features, labels


def _prices(n=400, seed=0, start="2015-01-02", start_price=50.0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq="B")
    close = start_price + np.cumsum(rng.normal(0.05, 1.0, n)).clip(-40, None)
    close = np.clip(close, 5, None)
    return pd.DataFrame({
        "Open": close, "High": close * 1.01, "Low": close * 0.99,
        "Close": close, "Volume": rng.integers(1e6, 5e6, n).astype(float),
    }, index=idx)


class TestUniverse:
    def test_dedup_and_nonempty(self):
        u = universe.get_universe()
        assert len(u) == len(set(u))
        assert len(u) > 100

    def test_market_of(self):
        assert universe.market_of("AAPL") == "US"
        assert universe.market_of("ENB.TO") == "CA"
        assert universe.market_of("ABX.V") == "CA"


class TestConfig:
    def test_stocks_parquet_falls_back_to_full_history(self, tmp_path, monkeypatch):
        # CI's bootstrap builds only stocks_full.parquet — separate processes (the
        # paper-book step) must find it when the standard 10Y parquet is absent.
        from sterling.research import config
        std, full = tmp_path / "stocks.parquet", tmp_path / "stocks_full.parquet"
        monkeypatch.setattr(config, "STOCKS_PARQUET", std)
        monkeypatch.setattr(config, "STOCKS_FULL_PARQUET", full)
        assert config.stocks_parquet() == std          # neither exists -> default
        full.touch()
        assert config.stocks_parquet() == full         # only full exists -> fallback
        std.touch()
        assert config.stocks_parquet() == std          # standard wins when present


class TestDatasetCoverage:
    def test_coverage_report(self):
        prices = {"AAA": _prices(500), "BBB.TO": _prices(300, seed=1)}
        rep = dataset.coverage_report(prices)
        assert rep["tickers"] == 2
        assert rep["max_bars"] == 500
        assert rep["min_bars"] == 300


class TestFeatures:
    def _bench(self):
        return {"US": _prices(400, seed=9, start_price=30.0),
                "CA": _prices(400, seed=8, start_price=25.0)}

    def test_schema_and_sampling(self):
        prices = {"AAA": _prices(400, seed=2), "BBB.TO": _prices(400, seed=3)}
        fdf = features.build_features(
            prices, self._bench(), universe.market_of,
            hist_fund_fn=None, vix_by_date=None, step=5, warmup=252,
        )
        assert not fdf.empty
        for col in features.ALL_FEATURES:
            assert col in fdf.columns
        assert set(fdf["ticker"].unique()) == {"AAA", "BBB.TO"}
        # Fundamentals absent → those columns are all NaN (model handles it).
        assert fdf["pe"].isna().all()

    def test_no_lookahead_in_price_feature(self):
        df = _prices(400, seed=5)
        feats = features._price_feature_frame(df)
        pos = 300
        expected = df["Close"].iloc[pos] / df["Close"].iloc[pos - 21] - 1
        assert abs(feats["ret_21"].iloc[pos] - expected) < 1e-9


class TestLabels:
    def test_excess_return_label(self):
        prices = {"AAA": _prices(400, seed=2)}
        bench = {"US": _prices(400, seed=9, start_price=30.0)}
        fdf = features.build_features(prices, {"US": bench["US"]}, universe.market_of,
                                      hist_fund_fn=None, step=5, warmup=252)
        out = labels.add_labels(fdf, prices, bench, horizon=21)
        assert "label" in out.columns
        # label == fwd_return - bench_fwd, exactly.
        assert np.allclose(out["label"], out["fwd_return"] - out["bench_fwd"])
        # Every labelled row has a real forward window.
        assert out["label"].notna().all()


# ── Delisting-aware forward return (survivorship-free safety logic) ────────────

class TestDelistingReturn:
    def _series(self, n=100, price=100.0):
        import numpy as np
        dates = list(pd.date_range("2020-01-01", periods=n, freq="B").date)
        closes = np.full(n, price, dtype=float)
        return closes, dates

    def test_normal_forward_return(self):
        from sterling.research import labels
        closes, dates = self._series()
        closes[:] = [100 + i for i in range(len(closes))]
        r = labels.forward_return_with_delisting(closes, dates, pos=0, horizon=10)
        assert abs(r - (closes[10] / closes[0] - 1)) < 1e-9

    def test_bankruptcy_books_total_loss_not_dropped(self):
        # Name delists inside the window with no deal return → -100%, NOT None.
        from sterling.research import labels
        closes, dates = self._series()
        r = labels.forward_return_with_delisting(
            closes, dates, pos=0, horizon=10, delist_date=dates[5], delist_return=None)
        assert r is not None
        assert abs(r - (-1.0)) < 1e-9

    def test_buyout_books_deal_return(self):
        from sterling.research import labels
        closes, dates = self._series()
        # flat price, bought out at +30% on delisting
        r = labels.forward_return_with_delisting(
            closes, dates, pos=0, horizon=10, delist_date=dates[5], delist_return=0.30)
        assert abs(r - 0.30) < 1e-9

    def test_delisting_after_window_is_normal(self):
        from sterling.research import labels
        closes, dates = self._series()
        closes[:] = [100 + i for i in range(len(closes))]
        # delists at day 50, but we only look 10 days out → normal return
        r = labels.forward_return_with_delisting(
            closes, dates, pos=0, horizon=10, delist_date=dates[50], delist_return=-1.0)
        assert abs(r - (closes[10] / closes[0] - 1)) < 1e-9
