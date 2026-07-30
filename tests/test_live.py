import numpy as np, pandas as pd
from sterling.research import live, config

def _fake_features(tmp, n_tickers=60):
    rng = np.random.default_rng(0)
    rows=[]
    for t in range(n_tickers):
        rows.append({"date": pd.Timestamp("2026-07-21").date(), "ticker": f"T{t}",
                     "vol_63": abs(rng.normal(0.02,0.01))+0.005, "pred": rng.normal()})
    return pd.DataFrame(rows)

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
