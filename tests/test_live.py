import numpy as np, pandas as pd
from sterling.research import live, config

def _fake_features(tmp, n_tickers=60):
    rng = np.random.default_rng(0)
    rows=[]
    for t in range(n_tickers):
        rows.append({"date": pd.Timestamp("2026-07-21").date(), "ticker": f"T{t}",
                     "vol_63": abs(rng.normal(0.02,0.01))+0.005, "pred": rng.normal()})
    return pd.DataFrame(rows)

class TestEvaluate:
    def test_book_and_equal_weight_returns(self, tmp_path, monkeypatch):
        idx = pd.date_range("2026-07-24", periods=40, freq="B")   # series starts on the book date
        def frame(p0, p1):
            v = np.linspace(p0, p1, len(idx))
            return pd.DataFrame({"Close": v}, index=idx)
        # AAA +20% (heavy weight), BBB -10% (light weight)
        prices = {"AAA": frame(100, 120), "BBB": frame(100, 90)}
        led = pd.DataFrame({"rebalance_date": ["2026-07-24", "2026-07-24"],
                            "ticker": ["AAA", "BBB"], "sector": ["Tech", "Energy"],
                            "pred": [1.0, 0.5], "weight": [0.75, 0.25]})
        p = tmp_path / "ledger.csv"; led.to_csv(p, index=False)
        monkeypatch.setattr(live, "LEDGER", p)
        monkeypatch.setattr(live.store, "load_prices", lambda tks: prices)
        res = live.evaluate(fresh=False)
        b = res["books"][0]
        assert b["names"] == 2
        # book return: 0.75*20% + 0.25*(-10%) = 12.5% ; equal-weight: (20-10)/2 = 5%
        assert abs(b["book_return_pct"] - 12.5) < 0.3
        assert abs(b["equal_weight_pct"] - 5.0) < 0.3
        assert abs(b["weighting_alpha_pct"] - 7.5) < 0.3

    def test_fresh_uses_yfinance_source(self, tmp_path, monkeypatch):
        # fresh=True must route through _refresh_prices (yfinance), not the stale parquet
        idx = pd.date_range("2026-07-20", periods=40, freq="B")
        prices = {"AAA": pd.DataFrame({"Close": np.linspace(100, 110, len(idx))}, index=idx)}
        led = pd.DataFrame({"rebalance_date": ["2026-07-24"], "ticker": ["AAA"],
                            "sector": ["Tech"], "pred": [1.0], "weight": [1.0]})
        p = tmp_path / "ledger.csv"; led.to_csv(p, index=False)
        monkeypatch.setattr(live, "LEDGER", p)
        called = {"parquet": False, "fresh": False}
        def boom(tks): called["parquet"] = True; return {}
        def fresh(tks, years=2): called["fresh"] = True; return prices
        monkeypatch.setattr(live.store, "load_prices", boom)
        monkeypatch.setattr(live, "_refresh_prices", fresh)
        res = live.evaluate(fresh=True)
        assert called["fresh"] and not called["parquet"]
        assert res["books"][0]["names"] == 1


class TestBook:
    def test_book_weights_sum_to_one(self, tmp_path, monkeypatch):
        # stub model + tickers so generate_book runs offline
        df = _fake_features(tmp_path)
        pkl = tmp_path/"f.pkl"; df.to_pickle(pkl)
        class Stub:
            def predict(self, X): return X["pred"].to_numpy() if "pred" in X else np.zeros(len(X))
        monkeypatch.setattr(live, "_load_model", lambda: (Stub(), ["pred"]))
        monkeypatch.setattr(live.store, "load_tickers",
                            lambda: pd.DataFrame({"ticker":[f"T{t}" for t in range(60)],
                                                  "sector":["Tech"]*30+["Energy"]*30}))
        # feature frame needs vol_63 + pred columns for the stub predict/weights
        asof, book = live.generate_book(today=df, long_frac=0.2)
        assert abs(book["weight"].sum() - 1.0) < 1e-3
        assert len(book) == int(60*0.2)
        assert set(["ticker","sector","weight"]).issubset(book.columns)

    def test_nan_vol_still_gets_finite_weight(self, tmp_path, monkeypatch):
        df = _fake_features(tmp_path)
        df.loc[df.index[:5], "vol_63"] = np.nan     # a few names missing vol
        class Stub:
            def predict(self, X): return X["pred"].to_numpy()
        monkeypatch.setattr(live, "_load_model", lambda: (Stub(), ["pred"]))
        monkeypatch.setattr(live.store, "load_tickers",
                            lambda: pd.DataFrame({"ticker":[f"T{t}" for t in range(60)],
                                                  "sector":["Tech"]*60}))
        _, book = live.generate_book(today=df, long_frac=0.5)
        assert book["weight"].notna().all()
        assert abs(book["weight"].sum() - 1.0) < 1e-3

    def test_log_book_empty_is_noop(self, tmp_path, monkeypatch):
        ledger = tmp_path / "ledger.csv"
        monkeypatch.setattr(live, "LEDGER", ledger)
        import datetime as dt
        assert live.log_book(dt.date(2026, 7, 30), pd.DataFrame()) == 0
        assert not ledger.exists()
