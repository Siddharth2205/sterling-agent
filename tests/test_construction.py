import numpy as np, pandas as pd
from sterling.research import construction as C

def _panel(n_dates=10, n_tickers=80, signal=0.05, seed=0):
    rng = np.random.default_rng(seed); base = pd.Timestamp("2015-01-01")
    secs = ["Tech","Energy","Health","Fin"]; rows=[]
    for k in range(n_dates):
        d = (base + pd.Timedelta(days=95*k)).date()
        for t in range(n_tickers):
            pred = rng.normal(); fwd = signal*pred + rng.normal(0,0.05)
            rows.append(dict(date=d, ticker=f"T{t}", pred=pred, fwd_return=fwd,
                             sector=secs[t%4], vol_63=abs(rng.normal(0.02,0.008))+0.005))
    return pd.DataFrame(rows)

class TestConstruction:
    def test_long_only_runs_with_risk_stats(self):
        r = C.simulate_neutral(_panel(seed=1), long_frac=0.2)
        s = r["strategy_net"]
        assert set(["cagr_pct","sharpe","max_drawdown_pct"]).issubset(s)
        assert "long-only" in r["mode"]

    def test_market_neutral_mode_and_positive_signal(self):
        r = C.simulate_neutral(_panel(signal=0.06, seed=2), long_frac=0.2, short_frac=0.2)
        assert "mkt-neutral" in r["mode"]
        assert r["strategy_net"]["total_return_pct"] > 0   # spread captured

    def test_prepare_panel_attaches_vol_and_sector(self):
        base = pd.DataFrame({"date":["2015-01-01"],"ticker":["T0"],"pred":[1.0],"fwd_return":[0.1]})
        feats = pd.DataFrame({"date":["2015-01-01"],"ticker":["T0"],"vol_63":[0.02]})
        p = C.prepare_panel(base, feats, {"T0":"Tech"})
        assert p.loc[0,"vol_63"]==0.02 and p.loc[0,"sector"]=="Tech"
